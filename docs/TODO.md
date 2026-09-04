Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

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