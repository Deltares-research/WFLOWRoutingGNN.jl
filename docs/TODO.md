Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

---

# Code restructuring — 4-stage pipeline + mass-balance separation

**DONE (shipped 2026-10-21).** The restructure landed as R1–R7 + the performance
acceptance gate: the 4-stage pipeline (`preprocess → train → infer → analyze`,
dispatched via a `Stage` hierarchy, CLI `scripts/pipeline.jl`), MB code split into
`src/sparse_ops.jl` + `src/massbalance.jl` with a `MassBalanceFormulation` dispatch
hierarchy, and the layer-free post-hoc `mb_residual_diagnostics`. Verified
bit-for-bit against the pre-refactor fused path on a golden run (`sava_small_v081`,
CPU+GPU) with no performance regression. See [DECISIONS.md](DECISIONS.md) →
IMPLEMENTED 19-10-2026 and [CHANGELOG.md](CHANGELOG.md) 2026-10-21. Only the R5b
lateral-inflow experimental follow-up (below) is still open.

## R5b. Lateral-inflow (inwater) scaling in `ThetaKinematicWave`

Behaviour-changing, so kept out of the restructuring steps; done as a separate,
A/B-tested change (default now uses the fixed physics).

- [x] Confirm the discrepancy: `inwater_phys = forcingÂ·Ïƒ_iw + Î¼_iw` lacked the
      `postscale["river_inwater"]` (upstream area `a`) factor, whereas
      preprocessing divides inwater by `a` before z-scoring. True-data check
      (`sava_small_v081`, val split, layer `h_raw(true q)` âˆ’ wflow `river_h`):
      legacy rms 3.92 m / median 0.29 m / max 252 m; fixed rms 0.0331 m /
      median 0.0016 m / max 2.7 m (equal to the wflow reference residual, so the
      remaining error is data-intrinsic, see last item).
- [x] Where the factor lives: `build_massbalance(...; unscale_inwater = true)`
      stores the per-node `postscale["river_inwater"]` in the new
      `ThetaKinematicWave.postscale_inwater` field; forward, `mb_decoder_features`
      and `mb_diagnostics` use `(forcingÂ·Ïƒ + Î¼)Â·a`. Toggle:
      `ModelSettings.mb_inwater_unscale` (config key `mb_inwater_unscale`,
      default `true`; `load_model_settings` defaults a missing key to `false`, so
      old checkpoints keep the legacy behaviour; `Flux.loadmodel!` fills the new
      field for legacy checkpoints). Templates updated.
- [x] Related bug fixed in `_rollout_mb_diagnostics` ([src/rollout.jl](../src/rollout.jl)):
      `mb_diagnostics` got the **predicted** state as the current state. It now
      gets the state before the step, with the new q from the prediction. This
      changes `h_raw`, floor fractions, `net_flux` and volume-budget numbers of
      **all** runs (old values in `docs/EXPERIMENTS.md` are not comparable).
- [x] Tests: `test/test_gnn.jl` ("lateral inflow un-scaling (R5b)": toggle,
      physical budget, legacy layer, Î¸ < 1, gpu/cpu/precompute, legacy checkpoint
      loading, settings round-trip); `test/test_pipeline.jl` residual agreement
      now compares the layer's own `pred_q/pred_h/inwater`. Full suite 4870/4870.
- [x] Effect on a learned model. (a) e13b weights (co-adapted to the legacy layer),
      inference only, val: legacy toggle reproduces e13b (pooled q KGE 0.877 vs.
      stored 0.887); the fixed layer with those weights degrades (0.379), as
      expected for a mismatched layer. (b) Compressed retrain from scratch, GPU,
      seed 1, 42 epochs (strategy steps [1,2,4,6,8,10] Ã— 7), same hyper-parameters
      as e13b (legacy | fixed): pooled q KGE 0.490 | 0.547, gauge q KGE
      0.281 | 0.370, gauge peak ratio 1.27 | 0.98, `frac_h_raw_neg_node_mean`
      0.587 | 0.120, layer residual rms 2.06 | 1.60 m; but depth worse: median
      `river_h` rmse 0.53 | 2.58 m, `overpred_freq` 0.24 | 0.90, volume_pbias
      458 | 593 %, val h_1step 8.5 | 71. Both variants are unstable in fixed-horizon
      rollouts at this length (rmse 2e16 | 3e6). Conclusion: q improves and the
      floor stops being hit by most nodes, but depth is not better yet;
      **inconclusive** because of the single seed and the 42-epoch schedule.
- [ ] Full-length A/B on HPC (300 epochs, several seeds, e13b recipe, with/without
      `mb_inwater_unscale`) before relying on the fixed layer for results; check
      the depth bias with the fixed layer (possibly needs re-tuned `h_floor`/prior).
- [ ] Explain the non-zero residual of the wflow reference on `sava_small_v081`
      (rms 0.033 m, max 2.7 m; `[mass_balance_residual.reference]`): dry/floored
      nodes, lakes/reservoirs, or a different inflow definition (e.g. `inwater`
      vs. `inwater + runoff`); inspect `metrics/mb_residual_by_node.csv`.

---

# Enforce non-negative predicted discharge (physics floor)

From E3: the free-rollout produced **unphysical negative discharge** (daterange
`river_q_pred` min ≈ −6 m³/s across cells) as the model collapsed toward zero.
River discharge cannot be negative.

- **Current code (partial floor already exists — do NOT re-implement it).** The
  MB layer already floors its *internal* physical discharge at zero
  (`q_phys_new = max.(0f0, …)`, [src/massbalance.jl](../src/massbalance.jl#L299),
  since commit `5cb4922`), and `rollout_mb_diagnostics` reports that floored copy.
  The gap is
  that this floored value does **not** propagate: the MB layer returns only `h`,
  so the **state/decoder q that carries to the next step is un-floored**, and
  `fixed_horizon_metrics` reconstructs its reported q directly from `q_norm`
  **without** a floor ([src/rollout.jl](../src/rollout.jl#L483)) — hence E3's
  negative anchor/daterange q.
- [x] Floor the **propagating** predicted `river_q` (the decoder/state q, not
      just the internal h-path copy) so negative flow cannot feed the next step —
      e.g. `softplus`/`relu`/`max(·,0)` on the reconstructed absolute flow, chosen
      so it does not break the MB water-balance derivation of `river_h` or its
      gradients.
- [x] Floor the reported q in `fixed_horizon_metrics`
      ([src/rollout.jl](../src/rollout.jl#L483)) consistently with the state floor.
- [ ] Verify it does not mask instability (a floored-but-still-collapsing model
      should still be detectable via PBIAS / peak_ratio, not hidden by clamping).
- Touch: `src/gnn.jl` (decoder / state q), `src/rollout.jl`
  (`fixed_horizon_metrics`), a test asserting `q ≥ 0`.

---

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
- Touch: `src/postprocess.jl` (metric helpers), `src/analysis.jl` (stage-4
  wiring) (when unblocked).

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

---

# Diagnostics & plotting

## 3e. Subcatchment error attribution (post-hoc; which subcatchments inject error)

**Why (see [DECISIONS.md](DECISIONS.md) → PROPOSED 19-10-2026 and
[notes/subcatchment_error_attribution.md](notes/subcatchment_error_attribution.md)).**
Tests the hypothesis that *some subcatchments contribute disproportionately to
downstream error* and that this correlates with WFLOW subcatchment attributes.
Raw per-outlet NSE conflates **locally injected** error with **advected** error
(discharge accumulates), so the key quantity is the **incremental** error each
subcatchment adds. Post-hoc like [src/analysis.jl](../src/analysis.jl)'s
`spatial_eval_run` (thin CLI [scripts/spatial_eval.jl](../scripts/spatial_eval.jl))
— no retraining, primary method needs no forward pass. Add a `subcatch_error_attribution`
analysis function in [src/analysis.jl](../src/analysis.jl) with a thin CLI,
reusing the now-promoted `resolve_staticmaps` / `load_regrid_nc` plus
`ldd_to_graph`, `river_q_performance_metrics`.

- [x] **Subcatchment definition + size filter.** *(done: both routes behind one outlet-set interface; `--gauges VAR` = A, default = B; `--min-area` sets the retained threshold, default 100 km2. `sava_small_v081` has one gauge so only B applies there.)* Route (A, preferred): use
      WFLOW-native `wflow_subcatch` (or a gauge-set variant) with outlets from
      `wflow_gauges` / `idx_out`. Route (B, fallback): delineate incremental
      catchments from the LDD tree, splitting at confluences where both branches
      exceed an upstream-area threshold. Keep only subcatchments whose incremental
      `wflow_uparea` (or median outlet flow) exceeds a **configurable threshold**
      ("sufficiently large" — also dodges the dry-headwater floor regime where NSE
      is degenerate).
- [x] **Subcatchment tree from LDD.** *(done: cells are labelled by their first downstream outlet, so the partition is a tessellation by construction; `tree.check` verifies it and the area vs `wflow_uparea`, and the run aborts if it fails. Cells with no outlet downstream stay `unassigned`.)* Build which subcatchment drains into which
      (immediate upstream sub-outlets per outlet) via `ldd_to_graph`. **Verify the
      partition is a proper tessellation** (every cell in exactly one subcatchment;
      all non-local inflow passes through tracked sub-outlets) — the incremental
      form below depends on it.
- [x] **Per-outlet skill (baseline).** *(done for `river_q` only; `h` is not covered.)* Extract pred/true `q` (and `h`) at each
      retained outlet; compute NSE / KGE(+r,α,β) / PBIAS / peak_ratio / RMSE via
      `river_q_performance_metrics`. This is total (local+advected) error, not yet
      an attribution.
- [x] **Incremental (contributed) error — the crux.** *(done, same-time difference with no routing-lag correction.)* For subcatchment `S` with
      outlet `o` and immediate upstream sub-outlets `{u_i}`,
      `e_local(S,t) = [Q_pred(o)−Q_true(o)] − Σ_i [Q_pred(u_i)−Q_true(u_i)]`.
      Report `rms(e_local)`, signed `mean(e_local)`, and a **specific injection**
      (`rms(e_local)` ÷ incremental Σ inwater or incremental area) so sizes are
      comparable. Also report the **advected-vs-local split** (`rms(Σ upstream
      error)` vs `rms(e_local)`) per outlet.
- [ ] **(Optional Tier-2 counterfactual — needs re-inference, still no
      retraining.)** Re-run the trained model with *true* upstream inflow
      substituted at `S`'s boundary; measure outlet error to isolate local skill
      exactly. Inference-time intervention on fixed weights.
- [x] **Attribute aggregation (per subcatchment).** *(done via the `SUBCATCH_ATTRIBUTES` table; variables absent from `staticmaps.nc` are skipped.)* Outlet-cell value
      (`wflow_uparea`, `wflow_streamorder`, `RiverSlope`); incremental mean/median
      (`KsatVer`, `SoilThickness`, `M`, `f`, `PathFrac`, `WaterFrac`, `Slope`,
      `N_River`, `RiverSlope`); presence (reservoir `ResTrapEff>0`, glacier
      `wflow_glacierfrac>0`); nonlinearity (fraction of time `Q > qbankfull`).
- [x] **Attribute ranking.** *(code done: permutation p, Fisher-z CI, Bonferroni, `low_power` flag, near-constant attributes skipped. **The `sava_v081` run is still open**: the only local full-Sava runs, `experiments/sava_full_v081` and `experiments/test_sava_v081`, have diverged predictions (|Q| ~ 1e38 from step 7), so rerun on a healthy full run, e.g. from the HPC.)* Spearman rank-correlate each attribute vs the
      local-error metric across subcatchments; scatter per attribute with ρ; rank
      by |ρ|. Report ρ CI/p; flag low power + multiple-comparison on small basins —
      run on **`sava_v081`** (many subcatchments) for a real test, not just
      `sava_small`.
- [x] **Outputs (mirror `spatial_eval.jl`).** `metrics/<split>_subcatch_error_attribution.csv`
      (row per subcatchment: id, outlet node, incremental area, n_cells, outlet
      skill, `e_local` stats, specific injection, advected RMS, aggregated
      attributes); plots: subcatchment map coloured by local injection,
      advected-vs-local stacked bars along the main stem, attribute scatter panel.
- **Note:** partly unblocks §2b (`wflow_gauges*`/`wflow_subcatch*` is a ready-made
  gauge/subcatchment definition for per-gauge reporting).
- Touch: new `subcatch_error_attribution` in `src/analysis.jl` + a thin
  `scripts/subcatch_error_attribution.jl` CLI (`resolve_staticmaps`/`load_regrid_nc`
  are already in `src/`); no source-model change for the primary method.

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
  `fieldtype(ThetaKinematicWave, :A_routing)` are `AbstractMatrix{Float32}`
  (abstract) though the stored value is concrete `SparseMatrixCSC`.
  `@code_warntype` shows `Body::ANY` with red
  `getproperty(l, :A)::ABSTRACTMATRIX{FLOAT32}` propagating `::ANY` through
  `neigh`/`net_flux`/`q_new`/`h_new`/`Δ`. One `SparseConv` forward measures
  **18 allocs / 331 KB / ~289 µs** (323-node CPU graph). Paid every layer,
  every step, in both rollout and training.
- [ ] Parameterise the adjacency field on `SparseConv`: add type params
      `SA<:AbstractMatrix{Float32}` (currently the struct is parametric only on
      `W_self`/`W_neigh`/`bias`/`σ`, not `A`) and `SB` (covers
      `Nothing`/concrete) so `A::SA; A_batched::SB`. Update the
      `Flux.gpu`/`Flux.cpu` overloads and `precompute_batched` to preserve the
      concrete types.
- [ ] Same treatment for `ThetaKinematicWave.A_routing` / `A_routing_batched`
      (currently `AbstractMatrix{Float32}` / `Union{Nothing,AbstractMatrix}`).
- [ ] Verify the fix: `@code_warntype` red→blue (no `::ANY` on the layer and
      top-level `WflowGNN` forwards) and BenchmarkTools allocs drop, via
      `scripts/typecheck_forward.jl` before/after.
- [ ] Re-run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` to quantify the wall-clock gain
      (the cuSPARSE/BLAS SpMM may still dominate; record the delta regardless).
- Touch: `src/gnn.jl` (`SparseConv` struct + `gpu`/`cpu`/`precompute_batched`),
  `src/massbalance.jl` (`ThetaKinematicWave` struct + `gpu`/`cpu`/
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
  `test_sava_small` fails to load (pre-restructure `WflowGNN` state predates the
  `augment_mb` field, `5b75540`; the struct changed again to `WflowGNN{M}` +
  `ThetaKinematicWave` in the restructure).
- [ ] Run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` on `sava_v081` (8235 nodes) on a
      ≥16 GB GPU to get the real surrogate-vs-Wflow inference number; persist to
      that run's `metrics/performance.toml` and add a STATUS entry to
      PERFORMANCE.md.
- [ ] Re-train / re-save a lightweight smoke model with the current
      `WflowGNN{M}` struct (e.g. under `test_data/test_model/`) so a
      current-struct smoke model exists for future benchmarks.
- Touch: `scripts/*`, `docs/PERFORMANCE.md` (no source change).

## 7. (Lower priority) Fuse ThetaKinematicWave broadcasts

- **Evidence (partly speculative):** GPU is launch-latency bound at ~323 nodes
  (single forward 1.71 ms → 0.25 ms/member under B=16). The MB layer issues many
  small separate-kernel broadcasts (`q_phys_new`, `upstream_q`, `net_flux`,
  `h_phys_new`, re-normalise).
- [ ] Fuse the elementwise broadcasts in the `ThetaKinematicWave` forward into
      fewer kernels; keep the θ=1 fast path bit-identical.
- [ ] Quantify launch-count/time change with `CUDA.@profile` before committing.
- Touch: `src/massbalance.jl` (`ThetaKinematicWave` forward).