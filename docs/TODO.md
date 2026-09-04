Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

---

# HParSearch config-parser drift (fix FIRST — blocks all future searches)

From the `sava_small_v081_s2_stability` post-mortem
([EXPERIMENTS.md](EXPERIMENTS.md) → *bug 1*): the intended `grad_clip` search
arm never executed. All 8 runs trained at the default `grad_clip = 1.0` even
though the config `search_space` requested `0.1`. Root cause: the hparsearch
path builds its own `TrainSettings`/`TrainingStrategy` inline, and this
constructor has **drifted out of sync** with the single-run parser in
[src/run.jl](../src/run.jl). Any `[train]`/`[train.strategy]` key the inline
constructor omits is silently written into the config dict but **never consumed**
— it defaults instead. This is a code bug, not a config error, and it silently
invalidates search arms.

## 0. Eliminate the hparsearch ↔ run.jl config-parsing drift

- **Evidence.** [src/hparsearch.jl](../src/hparsearch.jl) line ~242 builds
  `TrainSettings(...)` and line ~237 builds `TrainingStrategy(...)` inline. The
  single-run parser `parse_run_config` in [src/run.jl](../src/run.jl) (lines
  ~100–139) is the correct reference. Diffing the two, the search path currently
  **drops** the following keys (they can never be varied in a search):
  - `TrainSettings`: `grad_clip`, `lr_warmup_epochs`, `lr_peak_decay` (the three
    flagged in the post-mortem).
  - `TrainingStrategy`: `loss_type`, `peak_delta`, `peak_lambda`, `peak_gamma`,
    `peak_w_max` — so the **peak-weighted-Huber workstream (§2) is also
    un-searchable** until this is fixed.
  - Inconsistent default: `lr_steps` is a hard `td["lr_steps"]` (required) in
    hparsearch vs `get(td, "lr_steps", 10)` in run.jl.
- [x] **Preferred fix — remove the duplication, not patch it.** Refactor
      `parse_run_config` so the dict→`(ds, ms, ts)` construction lives in a
      shared helper that takes an already-loaded TOML `Dict` (+ resolve dir),
      e.g. `settings_from_config(d, toml_dir)`. Have both `parse_run_config`
      (path → load TOML → helper) and `run_hparsearch` (mutated dict → helper)
      call it. This makes future config keys land in both paths automatically
      and prevents the class of bug recurring.
- [ ] **Minimum fix (if the refactor is deferred).** Add the missing keys to the
      inline `TrainSettings`/`TrainingStrategy` in `src/hparsearch.jl`
      (`grad_clip`, `lr_warmup_epochs`, `lr_peak_decay`; `loss_type`,
      `peak_delta`, `peak_lambda`, `peak_gamma`, `peak_w_max`) and switch
      `lr_steps` to `get(td, "lr_steps", 10)` to match run.jl.
- [x] **Regression guard.** Add a test that a `search_space` overriding
      `train.grad_clip` (and one `train.strategy.*` key) is actually reflected in
      the constructed `TrainSettings`/`TrainingStrategy` for each combo — i.e.
      assert the override reaches the settings object, not just the config dict.
      Extend [test/test_strategy.jl](../test/test_strategy.jl) or
      [test/test_training.jl](../test/test_training.jl).
- [x] **Verify.** Re-ran a tiny 2-combo parser-level hparsearch harness varying
      `grad_clip = {1.0, 0.1}` and confirmed each run's saved
      `model/train_settings.toml` shows the intended value (`1.0` for one
      combo and `0.10000000149011612` for the other).
- Touch: `src/hparsearch.jl`, `src/run.jl` (extract shared parser), test file.

---

## 1. Distribution- & physics-aware feature scaling

- [ ] Keep physics channels (`q`, `h`, `inwater`) on **linear z-score** (nonlinear
      transforms would need inverting inside the MB layer).
- [ ] Area-normalise `river_inwater` like `river_q` (fold into `σ/μ_inwater`).
- [ ] `log1p`+z-score the skewed statics (`river_slope`, `width`, `depth`,
      `length`); plain z-score for `river_manning_n`.
- [ ] Optional `log(upstream_area)` static feature.
- [ ] Fit all stats on the **training split only**; add a normalized-tail
      diagnostic dump.
- Touch: `src/preprocess.jl` (`VAR_SCALERS`), `src/gnn.jl`, `src/schema.jl`.

## 2. Peak-weighted Huber loss (toggle vs. current MSE)

- [x] Strategy-gated loss selector; MSE stays default.
- [x] Huber base error with tunable `δ`; per-node peak weights
      `w_i,t = min(1 + λ·[max(0,(y-u_i)/s_i)]^γ, w_max)`; weights from the
      untransformed target; normalise by `Σw`; apply to the `h` channel too.
- [x] Precompute per-node `u_i`/`s_i` (train split). `peak_node_stats` in
      `src/preprocess.jl` computes real per-node (98th-pct / IQR) stats from
      `graphs[1:n_train]` (the training-period slice, sized by
      `ds.train_frac` — never val/test). `run_wflow_gnn` computes it once
      (when `strategy.loss_type == :huber`) and threads it through
      `train_model!`/`loss_function` as the new `peak_stats` keyword;
      `peak_weighted_huber_loss` now accepts `u`/`s` as either the original
      per-channel `AbstractVector` or a per-node `AbstractMatrix`
      (`(nvar, nnode)`). When `peak_stats` is not supplied (e.g. existing
      scripts/tests), `loss_function` keeps the old coarse per-batch scalar
      fallback for backward compatibility.
- **2a. Parameter estimation (avoid blind grid search):** compute `u_i` (≈98th
      pct), `s_i` (IQR); set `δ` from baseline residuals; set `h_loss_weight` by
      gradient-norm balancing; sweep only `λ` (guided by `C_peak`); leave `γ=1`;
      set `w_max` from the weight distribution.
- Touch: `src/strategy.jl`, `src/preprocess.jl`, `src/training.jl`, config in
  `src/run.jl`.

## 2b. Validation metrics needed for tuning

- [x] **Tier 1 (with the loss):** `C_peak`, peak-gradient-fraction, per-channel
  (q,h) grad norms, `RMSE_high`/`MAE_high`, weight summary stats.
  - [x] Weight summary stats (`peak_loss_summary`: `w_mean`/`w_max`/`w_min`) — helper only, not logged per-epoch in `src/training.jl`.
  - [x] `C_peak`, peak-gradient-fraction, per-channel grad norms, `RMSE_high`/`MAE_high` —
    implemented as `peak_epoch_diagnostics` (`src/training.jl`, using `peak_weight_matrix`
    from `src/strategy.jl`) and wired into `train_model!`'s per-epoch history/progress
    display (only meaningful under `loss_type == :huber`; NaN-filled under `:mse`).
- [x] **Tier 2 (model selection §14):** KGE + r/α/β, MAE, gauge-level PBIAS,
  event peak error + FHV, per-gauge (not pooled) reporting.
  - [x] `kge_metrics`, `mae_metric`, `pbias`, `event_peak_metrics` implemented as standalone helpers in `src/postprocess.jl` (unit-tested, not wired into training/rollout/run).
  - [x] Wired into the run pipeline via `river_q_performance_metrics` (`src/postprocess.jl`),
    called from `evaluate_and_write`/`write_run_metrics_toml` in `src/run.jl` and written to
    the run's metrics TOML output.
  - [ ] Per-gauge (not pooled) reporting — blocked: no "gauge" concept exists in the codebase yet; needs a design decision (which nodes count as gauges) before implementing. `river_q_performance_metrics` currently reports pooled (all-river-node) metrics only.
- [ ] **Tier 3 (only if a detection head is added):** POD / false-alarm ratio / CSI.
  - [x] `event_detection_metrics` implemented as a standalone helper (unit-tested, not wired in).
  - Deferred: no detection/exceedance head exists in the codebase yet; wiring this in is out of scope until one is added.
- Touch: `src/training.jl`, `src/postprocess.jl`, `src/rollout.jl`, `src/run.jl` — Tier 1 and
  Tier 2 helpers are now wired in; Tier 3 (`event_detection_metrics`) remains helper-only.

## 2c. Per-anchor fixed-horizon rollout diagnostic (highest-priority diagnostic)

- **Why (from [EXPERIMENTS.md](EXPERIMENTS.md) → `s2_stability` RECOMMENDATION
  #2 / HYPOTHESES).** The fixed-horizon metric currently collapses all `B`
  anchors into two **global** scalars, so a single diverging anchor dominates
  `val_peak_ratio` / `val_fixed_rmse` and the regime that diverges is invisible.
  The post-mortem established (by inference) that the MB rollout operator is
  **conditionally unstable** — bounded from benign low-flow starts, geometrically
  unstable (`peak_ratio ≈ amp^H`, `amp ≈ 12`) from stiff high-flow starts. This
  diagnostic converts that from strong inference to **direct evidence** and
  pinpoints the exact regime the `mb_theta` / peak-weighted-loss stability work
  must target. **Prediction to test:** divergence concentrates in
  high-flow-start anchors while low-flow anchors stay bounded.
- **Current code.** `fixed_horizon_metrics(model, fh)` in
  [src/rollout.jl](../src/rollout.jl) already runs each anchor independently
  (block-diagonal batch, `fh.B` anchors, arrays shaped `(N, H, B)`) but then
  reduces with `mean`/`maximum` over **all** anchors into `(rmse_q, peak_ratio)`.
  The per-anchor information exists and is simply discarded before the reduction.
- [x] **Return per-anchor arrays, not just the two scalars.** Have
      `fixed_horizon_metrics` also produce length-`B` vectors: per-anchor RMSE
      (`sqrt(mean(abs2, ...))` reduced over `(N, H)` only) and per-anchor peak
      ratio (`maximum(abs, pred)/max(true_peak_a, eps)` with a **per-anchor**
      truth peak). Keep the existing two aggregate scalars for backward
      compatibility (existing history fields / plot in
      [src/plot.jl](../src/plot.jl)).
- [x] **Tag each anchor with its start-state flow percentile.** In
      `build_fixed_horizon_eval` ([src/rollout.jl](../src/rollout.jl)) the anchor
      `starts` and per-anchor initial `states0` are known; compute each anchor's
      start-state discharge summary (e.g. basin-mean or basin-max physical `q` at
      step 0) and its percentile within the val-split flow distribution, and
      store it on `FixedHorizonEval` (new field) so the diagnostic can be keyed
      by flow regime without recomputation.
- [x] **Persist it for one full-curriculum run.** Write the per-anchor table
      (`anchor_index`, `start_time`/`start_step`, `start_flow_percentile`,
      `fixed_rmse`, `peak_ratio`) to the run's `metrics/` dir (CSV/TOML, mirror
      the existing `plot_fixed_horizon` CSV writer in
      [src/plot.jl](../src/plot.jl)). Implemented as `metrics/fixed_horizon_anchors.csv`.
      Per-epoch logging of the full table is not
      required — a final-epoch (or best-epoch) dump is enough to test the
      prediction; keep the two aggregate scalars in the per-epoch history.
- [x] **Guard against the reduction masking divergence.** Added per-epoch
      anti-masking aggregates in training history: `val_peak_ratio_frac_gt2`
      (fraction of anchors with `peak_ratio > 2`) and
      `val_fixed_rmse_highflow` (mean fixed-horizon RMSE over anchors with
      `start_flow_percentile >= 0.8`), and surfaced them in fixed-horizon CSV /
      metrics TOML outputs.
      Consider also logging a
      cheap aggregate that is *not* max-dominated (e.g. fraction of anchors with
      `peak_ratio > threshold`, or a high-flow-anchor-only RMSE) as a first-class
      per-epoch signal, per RECOMMENDATION #3 ("report a high-flow-anchor rollout
      metric, not the single benign date-range trajectory").
- **Validation.** On a full-curriculum `increment` run (e.g. reproduce hps4),
  confirm the per-anchor `peak_ratio` rises monotonically with
  `start_flow_percentile` and that low-percentile anchors stay `O(1)`. This
  directly confirms/refutes the conditional-instability hypothesis.
- Touch: `src/rollout.jl` (`FixedHorizonEval`, `build_fixed_horizon_eval`,
  `fixed_horizon_metrics`), `src/training.jl` (thread/persist the per-anchor
  table), `src/plot.jl` (writer), config toggle if per-epoch persistence is
  wanted.

## 3. Pushforward / detached-rollout trick (toggle)

- [ ] `rollout_grad = :full_bptt | :pushforward | :detached`, default full-BPTT.
- [ ] Detached (truncation-length-1) variant preferred: per-step loss, backprop
      one step, O(1) tape depth — should make gradient checkpointing unnecessary.
- [ ] Benchmark step time & peak memory vs. full-BPTT at `steps=10`.
- **Note:** fixes rollout-depth instability & training time, *not* the epoch-2
  single-step blow-up (that is items 1 & 2). Pair it with them.
- Touch: `src/strategy.jl`, `src/training.jl`, config.

---

# Computational performance

Distilled from the 02-09-2026 benchmark log in
[PERFORMANCE.md](PERFORMANCE.md) (STATUS AS OF 02-09-2026) and its persisted
metrics
([experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml](../experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml)).
Each item should land behind a toggle / be A/B-tested with the existing
`scripts/typecheck_forward.jl`, `scripts/diag_overhead.jl`,
`scripts/benchmark_inference_vs_train.jl`, `scripts/benchmark_rollout.jl`.

## 4. Fix type instability in the hot forward path (highest-confidence win)

- **Evidence:** `fieldtype(SparseConv, :A)` and
  `fieldtype(MassBalanceLayer, :A_routing)` are `AbstractMatrix{Float32}`
  (abstract) though the stored value is concrete `SparseMatrixCSC`.
  `@code_warntype` shows `Body::ANY` with red
  `getproperty(l, :A)::ABSTRACTMATRIX{FLOAT32}` propagating `::ANY` through
  `neigh`/`net_flux`/`q_new`/`h_new`/`Δ`. One `SparseConv` forward measures
  **18 allocs / 331 KB / ~289 µs** (323-node CPU graph). Paid every layer,
  every step, in both rollout and training.
- [ ] Parameterise the adjacency field on `SparseConv`:
      `struct SparseConv{M,V,F,SA<:AbstractMatrix{Float32},SB} … A::SA;
      A_batched::SB end` (SB covers `Nothing`/concrete). Update the
      `Flux.gpu`/`Flux.cpu` overloads and `precompute_batched` to preserve the
      concrete types.
- [ ] Same treatment for `MassBalanceLayer.A_routing` / `A_routing_batched`
      (currently `AbstractMatrix{Float32}` / `Union{Nothing,AbstractMatrix}`).
- [ ] Verify the fix: `@code_warntype` red→blue (no `::ANY` on the layer and
      top-level `WflowGNN` forwards) and BenchmarkTools allocs drop, via
      `scripts/typecheck_forward.jl` before/after.
- [ ] Re-run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` to quantify the wall-clock gain
      (the cuSPARSE/BLAS SpMM may still dominate; record the delta regardless).
- Touch: `src/gnn.jl` (`SparseConv`, `MassBalanceLayer` structs + `gpu`/`cpu`/
  `precompute_batched`).

## 5. Gate per-batch training diagnostics to once-per-epoch

- **Evidence:** `training_diagnostic_overhead` — the logging-only forwards
  (`one_step_loss` unconditionally + `loss_components` + `mb_amplification`
  when a mass balance is present) cost **62.4 ms CPU / 25.6 ms GPU** vs a real
  train step of **61.9 / 15.5 ms**, i.e. **50.2% (CPU) / 62.2% (GPU)** of every
  batch at horizon 1. Fraction shrinks at deeper BPTT horizons but dominates the
  many horizon-1 epochs.
- [ ] In the `train_model!` batch loop, stop calling `one_step_loss`,
      `loss_components`, `mb_amplification` every batch; compute them **once per
      epoch on a single fixed batch** (mirror the pattern already documented for
      `peak_epoch_diagnostics`). Keep the per-epoch history fields populated.
- [ ] Put it behind a `TrainSettings` toggle (e.g. `diagnostics::Bool = true`)
      so the per-batch behaviour can be restored for debugging and A/B'd.
- [ ] Confirm ~1.3–2× shallow-horizon training speedup with
      `scripts/diag_overhead.jl` (and an end-to-end `@elapsed` on one epoch).
- Touch: `src/training.jl` (batch loop + `TrainSettings`), config in
  `src/run.jl`.

## 6. Re-baseline on the production model / refresh smoke model

- **Evidence:** the 02-09 numbers are a 323-node / 8-layer proxy on a 4 GB
  laptop GPU; `sava_v081` (8235 nodes) does not fit locally, and
  `test_sava_small` fails to load (4-field `WflowGNN` state predates the
  `augment_mb` field, `5b75540`).
- [ ] Run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` on `sava_v081` (8235 nodes) on a
      ≥16 GB GPU to get the real surrogate-vs-Wflow inference number; persist to
      that run's `metrics/performance.toml` and add a STATUS entry to
      PERFORMANCE.md.
- [ ] Re-train / re-save `test_sava_small` with the current `WflowGNN` struct so
      a lightweight current-struct smoke model exists for future benchmarks.
- Touch: `scripts/*`, `docs/PERFORMANCE.md` (no source change).

## 7. (Lower priority) Fuse MassBalanceLayer broadcasts

- **Evidence (partly speculative):** GPU is launch-latency bound at ~323 nodes
  (single forward 1.71 ms → 0.25 ms/member under B=16). The MB layer issues many
  small separate-kernel broadcasts (`q_phys_new`, `upstream_q`, `net_flux`,
  `h_phys_new`, re-normalise).
- [ ] Fuse the elementwise broadcasts in the `MassBalanceLayer` forward into
      fewer kernels; keep the θ=1 fast path bit-identical.
- [ ] Quantify launch-count/time change with `CUDA.@profile` before committing.
- Touch: `src/gnn.jl` (`MassBalanceLayer` forward).