Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

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

**Scope note (2026-09-15, from E4/E5 downstream pattern).** Poorer downstream
skill has two causes: (a) **physical flux accumulation** — the MB layer routes in
physical m³/s ([src/gnn.jl](../src/gnn.jl#L281)), so the outlet's `net_flux` is a
small residual on a huge accumulated through-flow, and the per-step amplification
(`amp≈12`, `mb_gain≈22`) is δ- and loss-invariant (E5). (b) **heavy-tailed statics**
under plain z-score make the outlet a feature-space outlier. These items fix **(b)
only** (better-conditioned downstream nodes → tighter bounded/1-step fit + NSE p10
tail); they do **NOT** touch (a) — q/h stay linear-z-score so `amp` is unchanged
and the **outlet rollout explosion is out of scope here** (that's E6 / q≥0 floor).
Rescaling q/h to attack `amp` (per-node σ, or θ*=σ_h/(dt·σ_q), see
[notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md) §4)
is a separate, MB-invasive item — not below.

- [ ] Keep physics channels (`q`, `h`, `inwater`) on **linear z-score** (nonlinear
      transforms would need inverting inside the MB layer).
- [ ] Area-normalise `river_inwater` like `river_q` (fold into `σ/μ_inwater`).
- [ ] `log1p`+z-score the skewed statics (`river_slope`, `width`, `depth`,
      `length`); plain z-score for `river_manning_n`.
- [ ] Optional `log(upstream_area)` static feature.
- [ ] Fit all stats on the **training split only**; add a normalized-tail
      diagnostic dump.
- Touch: `src/preprocess.jl` (`VAR_SCALERS`), `src/gnn.jl`, `src/schema.jl`.

## 2b. Validation metrics — remaining items

Tier 1 (peak diagnostics: `peak_epoch_diagnostics`) and Tier 2
(`river_q_performance_metrics`: KGE, r/α/β, MAE, PBIAS, event peak + FHV, pooled)
are **done and wired in** (see [CHANGELOG.md](CHANGELOG.md) 2026-09-04).
Remaining:

- [ ] **Per-gauge (not pooled) reporting** — blocked: no "gauge" concept exists
      in the codebase yet; needs a design decision (which nodes count as gauges)
      before implementing. `river_q_performance_metrics` currently reports pooled
      (all-river-node) metrics only.
- [ ] **Tier 3 detection metrics (POD / FAR / CSI)** — `event_detection_metrics`
      exists as a unit-tested helper but is **not wired in**; deferred until a
      detection/exceedance head exists in the codebase.
- Touch: `src/postprocess.jl`, `src/run.jl` (when unblocked).

## 3. Pushforward / detached-rollout trick (toggle)

- **Definitions (both cut the BPTT tape to O(1) via stop-gradient — this is the
  `Flux.ignore_derivatives` primitive already used in `src/strategy.jl`, NOT a
  forward-mode/JVP linearisation of `f_θ`):**
  - **`:pushforward` (Brandstetter et al. 2022).** Unroll `k` steps with the
    *actual* nonlinear `f_θ` but under stop-gradient, then take **one**
    gradient-carrying step from that self-generated state and score **only that
    final step**: `x_{t+k-1} = detach(f_θ∘…∘f_θ(x_t))`, loss on
    `f_θ(x_{t+k-1})` vs `y_{t+k}`. The differentiated step sees a
    distribution-shifted (self-rolled) input → teaches robustness to the model's
    own rollout error. Supervises the endpoint only.
  - **`:detached` (truncation-length-1).** Detach the incoming state at **every**
    step, take one differentiable `f_θ` application, score **that** step, then
    detach its output before the next: `x̂_{t+1} = f_θ(detach(x_t))`, loss each
    step. Keeps per-step hydrograph supervision (matters for the dense daily q/h
    targets + peak-weighted loss) with O(1) tape depth. `k` is fixed at 1 — there
    is no configurable TBPTT window in this design.
- [x] `rollout_grad = :full_bptt | :pushforward | :detached`, default full-BPTT.
- [x] Detached (truncation-length-1) variant preferred: per-step loss, backprop
      one step, O(1) tape depth — should make gradient checkpointing unnecessary.
- [ ] Benchmark step time & peak memory vs. full-BPTT at `steps=10`.
- **Note:** fixes rollout-depth instability & training time, *not* the epoch-2
  single-step blow-up (that is items 1 & 2). Pair it with them.
- Touch: `src/strategy.jl`, `src/training.jl`, config.

## 3c. Hybrid pushforward + teacher-forced 1-step loss

**Motivation (E6).** Pure `pushforward` is the first lever to lower `amp`
(11.0→6.4) and give a finite fixed-horizon rollout, but it supervises **only the
detached endpoint**, so in deep curriculum phases the 1-step map and `river_h` are
never directly trained — E6 regressed `val_q_1step` 0.0144→0.165 and `river_h`
spatial NSE −7.9→−96. Decouple long-horizon self-correction from 1-step/h
supervision with a convex combination:
`L = α·L_pushforward + (1−α)·L_1step`.

- [x] Add the **teacher-forced** 1-step term: for `t in 1:k`, forward from the
      **ground-truth** state `batch[t].ndata.state` (NOT the rolled/detached
      state — that is the `:detached` signal, which E6 showed is a net negative,
      `amp`↑12.7) and score `step_loss` vs `targets[t]`. O(1) tape, no BPTT; reuse
      the existing `one_step_loss` machinery ([src/strategy.jl](../src/strategy.jl#L516)).
- [x] Only active when `rollout_grad = :pushforward`; keep `step_loss` shared so
      `loss_type` / `h_loss_weight` / peak params apply consistently to both terms.
- [x] Config knob `pushforward_tf_weight = (1−α)`, **default 0.0** (= today's pure
      pushforward exactly; no behaviour change for existing configs). Prefer a
      weight over a new enum value so `α=1` continuously recovers pure pushforward.
- [ ] Validate with an **α sweep** on the E6-pushforward base
      (`pushforward_tf_weight ∈ {0.0, 0.25, 0.5, 0.75}`). Win condition: recover
      `val_q_1step`/`river_h` toward full_bptt **while** `amp` stays ~6.4 and the
      fixed-horizon RMSE stays finite (i.e. break the E6 trade-off).
- **Prerequisite** for the deeper curriculum schedule (extending `steps` toward
  `eval_horizon`): decoupled 1-step/h supervision is what lets the deep phases be
  front-loaded without starving h.
- Touch: `src/strategy.jl` (`loss_function` pushforward branch), `src/training.jl`
  / `TrainingStrategy` (the weight field), config.

## 3d. Make peak-weighting applicable to any loss type

Currently the peak weight is entangled with Huber: `step_loss`
([src/strategy.jl](../src/strategy.jl#L423)) forks `:huber` → peak-weighted vs
`else` → **plain unweighted `Flux.mse`**, so peak-weighting can only be tested
*through* Huber (confounded by the δ-tail effect that E4/E5 showed dominates). But
the weight itself is already loss-agnostic — `_peak_weight_and_score`
([src/strategy.jl](../src/strategy.jl#L197)) depends only on `target, u, s, λ, γ,
w_max`; only `_huber_element` is the swappable kernel. This is a factoring, not a
rewrite.

- [ ] Factor the per-element kernel out of the weighting
      (`_element_loss(r, Val(loss_type), delta)` for `:mse` / `:huber`, extensible
      to `:mae`/`:logcosh` later); rename `peak_weighted_huber_loss` →
      `peak_weighted_loss` taking `loss_type` (+ `delta`, Huber-only).
- [ ] Collapse the `step_loss` `:huber`/`else` fork into one weighted path for
      both q and h, so `loss_type` and the peak params (`λ, γ, w_max`) become two
      **orthogonal** axes.
- [ ] **Fix the latent reduction-scale bug while here:** today `λ=0` reduces with
      `sum` but `λ>0` normalises weights to sum 1 (weighted **mean**) — so enabling
      peak-weighting silently rescales the loss by ~1/N. **Always normalise**
      (weighted mean, weights sum to 1) so `λ=0` reduces *exactly* to plain
      mean-MSE/mean-Huber (LR & `h_loss_weight` transfer unchanged) and `λ` is a
      pure shape knob. May also explain the E5 `c_peak`/`w_mean` split at δ=1.
- [ ] No breaking config change: `loss_type` stays; `peak_lambda/gamma/w_max`
      become applicable to any `loss_type`; document `peak_delta` as Huber-only.
      Existing `λ=0` / `mse` configs reproduce current behaviour after the
      normalise fix.
- **Caveat (evidence):** peak-weighting is a **weak** lever (E5 `w_mean≈1.0005`;
  loss-shape is orthogonal to the rollout instability). Value here is
  **cleanliness + unblocking a clean peak-weight test on MSE** (free of the Huber
  δ-tail confound) and three independent axes (`loss_type × peak-weight ×
  rollout_grad`), NOT a stability fix. Do not expect peak-weighted MSE to move
  stability.
- Touch: `src/strategy.jl` (`_huber_element` → `_element_loss`,
  `peak_weighted_huber_loss` → `peak_weighted_loss`, `step_loss`), any callers /
  tests referencing `peak_weighted_huber_loss`.

---

# Diagnostics & plotting

## 3b. Network-graph inset on the river_q / river_h timeseries plots

The `river_q` / `river_h` timeseries panels already carry a small inset, but it
currently draws only a faint **scatter of active-node positions** ("node map")
with the selected cell highlighted ([src/plot.jl](../src/plot.jl#L537)). Upgrade
it to show the actual **river network graph** so it is obvious where in the
catchment the plotted cell sits.

- [x] Draw the LDD/river-network connectivity (edges between each node and its
      downstream neighbour) in the inset, not just a point cloud — reuse the
      network-drawing logic from [scripts/plot_ldd.jl](../scripts/plot_ldd.jl).
- [x] Keep the current **marker** for the plotted cell (orangered), sized/z-ordered
      so it reads clearly on top of the network.
- [x] Thread the edge/connectivity info through to `plot_timeseries` via the
      existing `inset` named tuple (extend it with the edge list / downstream
      index) so `plot_downstream_timeseries` populates it from the graph.
- [x] Apply to both `river_q` and `river_h` panels (loops over `state_vars`, so
      one change covers both).
- Touch: `src/plot.jl` (`plot_timeseries` inset block, `plot_downstream_timeseries`
      `inset_spec`), optionally factor shared network-drawing out of
      `scripts/plot_ldd.jl`.

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