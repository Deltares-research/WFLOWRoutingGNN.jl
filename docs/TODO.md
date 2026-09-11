Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

---

# HParSearch config-parser drift — RESOLVED 2026-09-04

From the `sava_small_v081_s2_stability` post-mortem
([EXPERIMENTS.md](EXPERIMENTS.md) → *bug 1*): the intended `grad_clip` search
arm never executed. All 8 runs trained at the default `grad_clip = 1.0` even
though the config `search_space` requested `0.1`. Root cause: the hparsearch
path built its own `TrainSettings`/`TrainingStrategy` inline, and this
constructor had **drifted out of sync** with the single-run parser in
[src/run.jl](../src/run.jl). Any `[train]`/`[train.strategy]` key the inline
constructor omitted was silently written into the config dict but **never
consumed** — it defaulted instead. This was a code bug, not a config error, and
it silently invalidated search arms.

**Status: FIXED (preferred refactor taken).** Verified live by the E3 sweep,
which correctly varied `peak_lambda` and honoured `loss_type=huber` /
`grad_clip=1.0` (the per-cell `peak_loss.*` diagnostics only emit under Huber).

## 0. Eliminate the hparsearch ↔ run.jl config-parsing drift

- [x] **Preferred fix — remove the duplication, not patch it.** Done: the
      dict→`(ds, ms, ts)` construction now lives in a shared
      `settings_from_config(d, toml_dir)` in [src/run.jl](../src/run.jl#L66);
      `parse_run_config` ([src/run.jl](../src/run.jl#L153)) and the hparsearch
      path ([src/hparsearch.jl](../src/hparsearch.jl#L211)) both call it, so
      future config keys land in both paths automatically.
- [x] **Regression guard.** Added in
      [test/test_hparsearch.jl](../test/test_hparsearch.jl) — asserts a
      `search_space` override reaches the constructed settings object, not just
      the config dict.
- [x] **Verify.** Confirmed via the E3 sweep (grad_clip / loss_type / peak_* all
      propagated per cell) and documented in
      [docs/CHANGELOG.md](CHANGELOG.md).
- Touched: `src/hparsearch.jl`, `src/run.jl` (shared parser), `test/test_hparsearch.jl`.

---

# Autotune ignores the configured loss — tunes on MSE even for Huber — RESOLVED 2026-09-11

Fixed in commit `4731de3` ("fix autotune with huber loss"): `make_model_loader`
now builds its probe strategy via `probe_training_strategy`, which forwards the
full loss config (`loss_type`, `peak_delta`, `peak_lambda`, `peak_gamma`,
`peak_w_max`); the effective loss is logged, and `test/test_lr_autotune.jl`
asserts inheritance. The documented teacher-forced / horizon-1 limitation below
still stands (treat autotuned LR as an upper bound).

From the E3 (`sava_small_v081_e3_huber_lambda`) post-mortem
([EXPERIMENTS.md](EXPERIMENTS.md) → E3): the whole point of re-running Step-0
autotune for E3 was that "MSE → Huber changes the gradient scale". But the LR
range test **never sees the Huber loss** — it silently tunes on MSE, so the
`lr_start = 0.012` it recommended was an MSE-scaled value applied to a Huber run.
Combined with the ~90× jump from the E1/E2 LR (1.3e-4), this is a prime suspect
for E3's rollout **collapse-to-zero** (see EXPERIMENTS.md).

- **Root cause.** [scripts/lr_range_test.jl](../scripts/lr_range_test.jl)
  `make_model_loader` (~line 124) builds its own strategy and forwards **only**
  `noise_scale` and `h_loss_weight`:
  ```julia
  strategy = TrainingStrategy([horizon], [1], ts.strategy.noise_scale;
                              h_loss_weight = ts.strategy.h_loss_weight)
  ```
  `loss_type`, `peak_delta`, `peak_lambda`, `peak_gamma`, `peak_w_max` are
  dropped, so the constructor defaults (`loss_type = :mse`, `peak_delta = 1`,
  `peak_lambda = 0`) take over and `loss_function` computes **MSE** regardless of
  the config.
- [x] **Fix.** Forward the full loss config from `ts.strategy` into the
      `TrainingStrategy` that `make_model_loader` constructs (`loss_type`,
      `peak_delta`, `peak_lambda`, `peak_gamma`, `peak_w_max`). One-line change;
      no trained model needed (still a horizon-1, teacher-forced test).
      - Note: keep `peak_lambda` at the base (0) is fine for a λ-swept box search
        (one tune for the whole grid), **but `loss_type = :huber` and
        `peak_delta` must be forwarded** so the gradient scale matches the run.
- [x] **Guard.** Log the effective `loss_type`/`peak_delta` used by the range
      test so a mismatch with the config is visible in the autotune output.
- [x] **Regression test.** Assert the strategy built by `make_model_loader`
      inherits `loss_type` from `ts.strategy` (extend `test/test_lr_autotune.jl`).
- **NOT in scope (documented limitation, do not "fix").** The range test is
  teacher-forced / horizon-1 and therefore **blind to multi-step rollout
  collapse** — doing it "properly" would need a pre-trained model, which defeats
  the purpose of tuning *before* training. Mitigation is to treat the autotuned
  LR as an **upper bound** for rollout-curriculum runs and hand-set lower if a
  run collapses, not to change the range test.
- Touch: `scripts/lr_range_test.jl` (`make_model_loader`), `scripts/autotune_train.jl`
  (logging), `test/test_lr_autotune.jl`.

---

# Enforce non-negative predicted discharge (physics floor)

From E3: the free-rollout produced **unphysical negative discharge** (daterange
`river_q_pred` min ≈ −6 m³/s across cells) as the model collapsed toward zero.
River discharge cannot be negative.

- **Current code (partial floor already exists — do NOT re-implement it).** The
  MB layer already floors its *internal* physical discharge at zero
  (`q_phys_new = max.(0f0, …)`, [src/gnn.jl](../src/gnn.jl#L281), since commit
  `5cb4922`), and `rollout_mb_diagnostics` reports that floored copy. The gap is
  that this floored value does **not** propagate: the MB layer returns only `h`,
  so the **state/decoder q that carries to the next step is un-floored**, and
  `fixed_horizon_metrics` reconstructs its reported q directly from `q_norm`
  **without** a floor ([src/rollout.jl](../src/rollout.jl#L483)) — hence E3's
  negative anchor/daterange q.
- [ ] Floor the **propagating** predicted `river_q` (the decoder/state q, not
      just the internal h-path copy) so negative flow cannot feed the next step —
      e.g. `softplus`/`relu`/`max(·,0)` on the reconstructed absolute flow, chosen
      so it does not break the MB water-balance derivation of `river_h` or its
      gradients.
- [ ] Floor the reported q in `fixed_horizon_metrics`
      ([src/rollout.jl](../src/rollout.jl#L483)) consistently with the state floor.
- [ ] Verify it does not mask instability (a floored-but-still-collapsing model
      should still be detectable via PBIAS / peak_ratio, not hidden by clamping).
- Touch: `src/gnn.jl` (decoder / state q), `src/rollout.jl`
  (`fixed_horizon_metrics`), a test asserting `q ≥ 0`.

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
  [src/rollout.jl](../src/rollout.jl) now returns both the global scalars and
  per-anchor arrays (`rmse_q_anchor`, `peak_ratio_anchor`); `FixedHorizonEval`
  stores anchor start-flow summaries/percentiles; and the final run writes a
  per-anchor table via `write_fixed_horizon_anchor_table`.
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
      [src/plot.jl](../src/plot.jl)). Per-epoch logging of the full table is not
      required — a final-epoch (or best-epoch) dump is enough to test the
      prediction; keep the two aggregate scalars in the per-epoch history.
- [x] **Guard against the reduction masking divergence.** Consider also logging a
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

## 2d. System water-volume validation diagnostic (pred AND truth)

- **Why (see [DECISIONS.md](DECISIONS.md) → PROPOSED 08-09-2026).** Per-cell
  RMSE hides slow, systematic storage error. An integrated total-volume signal
  is the natural probe for the compounding `river_h` drift flagged in the
  `s2_stability` post-mortem. Total volume is **not** a conservation check (open
  system; local conservation is already guaranteed by the hard-MB layer) — it is
  a **pred-vs-truth storage-tracking** signal, so it MUST be computed for both
  the predicted and the ground-truth trajectory and always plotted/scored
  together.
- **Current code.** `rollout_mb_diagnostics(model, split, static)` in
  [src/rollout.jl](../src/rollout.jl) already returns per-node physical
  `(n_nodes × T)` matrices for both prediction and truth: `pred_h`, `true_h`,
  `inwater`, `net_flux`, `upstream_q`, `pred_q`, `true_q`. The
  `MassBalanceLayer` (`model.mass_balance`) exposes `postscale_q` (= drainage
  area `A`) and `postscale_h` (= `A/(w·l)`), so per-node `w_i·l_i =
  postscale_q_i / postscale_h_i` (equivalently `1 / ph_over_pq_i`).
- [x] **Compute storage series for BOTH pred and truth.**
      `V_pred(t) = Σ_i pred_h[i,t] · (w_i l_i)` and
      `V_true(t) = Σ_i true_h[i,t] · (w_i l_i)` (m³). Never compute one without
      the other — the diagnostic value is the comparison.
- [x] **Headline scalars → run metrics TOML.** `volume_pbias`
      (`100·Σ(V_pred−V_true)/Σ V_true`) and `volume_drift` (least-squares slope
      of `V_pred(t) − V_true(t)` vs `t`, units m³/step). Mirror the existing
      metrics-writing path (`evaluate_and_write` / `write_run_metrics_toml` in
      [src/run.jl](../src/run.jl)).
- [x] **Budget-closure series for BOTH pred and truth.** Cumulative lateral
      inflow `Σ inwater·Δt`, cumulative outlet outflow `Σ q_outlet·Δt`, and
      `ΔV(t) = V(t) − V(0)`, computed once with the predicted trajectory and once
      with the ground-truth trajectory. For the hard-MB model
      `ΔV_pred ≈ Δt·Σ_i net_flux_i`, so the pred budget residual should be ~0 —
      log it as a self-test. Do not assume the truth budget closes (daily
      discretisation).
- [x] **Outlet definition.** Using the existing single-outlet convention:
      `argmax(upstream_area)` (same as `plot_downstream_timeseries`). Multi-outlet
      sink-set aggregation can be added later if needed.
      Either the
      single `argmax(upstream_area)` node (reuse the
      `plot_downstream_timeseries` rule) or the full set of sink nodes (no
      downstream neighbour in the routing adjacency). Multi-outlet basins need
      the sink-set variant; document the choice.
- [x] **Plot.** New `plot_volume_budget(diags; path, timestamps)` alongside the
      existing `plot_mb_diagnostics` in [src/plot.jl](../src/plot.jl): row 1 =
      `V_pred(t)` vs `V_true(t)`; row 2 = budget closure (cumulative inflow /
      outflow / `ΔV`) for pred and truth. Write a companion CSV like the other
      plot writers.
- **Validation.** On a known-good full-curriculum `increment` run: confirm the
  pred budget residual is ~0 (self-test passes), and that `volume_drift` is small
  where 1-step q-skill is high. Cross-check `volume_pbias` sign against the
  existing per-node `bias`/`relbias` maps.
- Touch: `src/rollout.jl` (`volume_budget_diagnostics` helper),
  `src/plot.jl` (`plot_volume_budget` + CSV), `src/run.jl` (val evaluation wiring
  + metrics TOML scalars).

## 2e. Longitudinal upstream timeseries (percentile ladder + catchment inset)

- **Why (see [DECISIONS.md](DECISIONS.md) → PROPOSED 08-09-2026).** Localise
  *where* along the network routing error is injected vs merely advected, and
  expose the regime-dependence of the `river_h` failure (low-flow / small-area
  headwaters vs the outlet). Extends the outlet-only
  `plot_downstream_timeseries`.
- **Current code.** `plot_downstream_timeseries(pred_grids, true_grids, domain,
  grid, upstream_area; upstream_points=K, ...)` in [src/plot.jl](../src/plot.jl)
  now selects `K` percentile-spanning active nodes (downstream→upstream) and
  renders per-node timeseries outputs with inset maps.
- [x] **Percentile ladder over `upstream_area`.** Select `K` active nodes
      spanning the drainage-area distribution from outlet (max) to headwater
      (min active): e.g. the nodes nearest the `K` evenly-spaced percentiles of
      `upstream_area` over active (non-NaN) nodes. Order panels
      downstream → upstream.
- [x] **`K` configurable via TOML, default 5.** Thread a config key (e.g.
      `[eval].upstream_points = 5`) through to the plotting call; fall back to 5
      when absent.
- [x] **Per-panel catchment inset.** Add a small inset to each timeseries panel
      showing all active nodes in grid space (`grid.rows`/`grid.cols`) as faint
      points with the selected node highlighted, so the reader sees where in the
      catchment the series sits. Reuse `plot_timeseries` per node; extend it (or
      wrap it) to accept an optional inset spec rather than duplicating the
      panel-drawing code.
- [x] **CSV.** Keep the existing per-node CSV export from `plot_timeseries`; name
      outputs by node so the K series are distinguishable.
- **Validation.** On a full-curriculum `increment` run, confirm the `river_h`
  noise/negative-NSE concentrates in the low-`upstream_area` (headwater) panels
  relative to the outlet — reproducing the regime-dependence inferred in the
  post-mortem.
- Touch: `src/plot.jl` (`plot_downstream_timeseries` multi-node percentile
      ladder + per-panel inset; `plot_timeseries` inset support), `src/run.jl`
      (`[eval].upstream_points` parse + call site), `src/training.jl`
      (`TrainSettings.upstream_points` persistence/validation).

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