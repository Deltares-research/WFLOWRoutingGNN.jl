This document serves as a changelog of architectural and design decisions.

Each entry records *what* was decided and *why*, with pointers to the research
note that holds the full reasoning. Newest at the top.

---


# PROPOSED 23-10-2026

## Diagnostic-metric convention for A/B across mass-balance formulations

**Status: PROPOSED (design only; see [TODO.md](TODO.md) §3h).** Several MB
diagnostics were written under the `ThetaKinematicWave` assumption that **`q` is the
fed-back state** and **`h` the stiff derived readout**. The hybrid
(`MuskingumContinuityHybrid`) **switches those roles** (`h`/storage is the carried
state, `q` a continuity readout out of the loop), so reported side-by-side in one
hpar CSV the affected columns are `NaN`/empty for the hybrid arm or silently mean
something different, making the A/B over `model.mb_formulation` misleading
(surfaced on the E15 search). Decision: **separate the axis of comparison (scientific
skill, must be identical across arms) from the mechanism diagnostic (stability, must
be meaningful per mechanism)** and treat each with a different strategy.

- **Scientific-skill metrics → one formulation-agnostic path, no switching.** All
  skill numbers are computed post-hoc from the physical `val_pred.nc`/`val_true.nc`
  fields and must never route through a layer-internal readout or be gated on the
  formulation. `q` is physical discharge in both arms (readout in the hybrid, state
  in theta), so `river_q_performance_metrics` (KGE, r/α/β, PBIAS, peak_ratio, FHV,
  per-node NSE), `river_h` RMSE/NSE/spatial medians, and the fixed-horizon RMSE on
  `q` are already directly comparable and stay as-is.
  - **Conservation is unified:** retire the theta-only *layer-internal* volume budget
    as the canonical conservation number and promote the **layer-free**
    `mb_residual_diagnostics` / `_analyze_mb_residual` (recomputed from the `q` & `h`
    fields, already "applies to any model with or without a MB layer") to the single
    conservation metric for both arms. The hybrid's exact continuity readout drives
    its field residual ≈ 0; theta exposes its structural mismatch — both land in the
    same `volume_pbias`/`drift` columns.
- **Stability metrics → one measured works-for-both headline + auto-switched analytic
  companions.** `amp` (q→h cross-amplification) and backbone contraction (h self-map
  spectral radius) measure different objects, but both proxy the per-step growth of
  the carried-state error. Define a single comparable **headline**:
  - `rollout_contraction` ρ̂ = `median_t ‖Δs_{t+1}‖/‖Δs_t‖` under teacher forcing,
    where `s` is **the formulation's own state row(s)** (dispatch only selects which
    row is the state: `q`-row for theta, `h`-row for the hybrid, via the existing
    `encoder_state`/state layout). Single orientation/target: **ρ̂ < 1 ⇒ contractive**,
    margin = 1 − ρ̂. This replaces `final_amp` as the sortable stability column.
  - **Companions (drill-down, namespaced, presence-based):** theta keeps
    `stability.analytic_gain = median(θ·dt·σ_q/σ_h)` and the realised
    `stability.q_to_h_amp`; the hybrid keeps `stability.backbone_rho = ρ(P)` (max and
    median backbone contraction, exact and horizon-independent). A companion column
    **exists only for the arm it means something for**.
- **Results-CSV schema.** The hpar scalar reader/aggregator
  (`_read_run_scalar_metrics`, `_aggregate_repetition_rows`) emits `rollout_contraction`
  for every run (the one stability column it averages) and the companions under their
  own keys, **skipping keys absent for an arm** (explicit not-applicable, never a NaN
  that silently averages to NaN — which is the current `final_amp` bug). Scientific
  columns are identical-definition across arms.
- **Loss weighting & labels (A/B fairness, not reported metrics).** Keep the
  *scientific* loss (physical `q` and `h` errors) identical across arms for a fair
  A/B; each formulation declares its internal weighting via the existing
  `mb_h_loss_weight` dispatch (hybrid returns `nothing` = plain strategy weight; theta
  keeps its q→h-stiffness compensation, the original rationale for
  `h_loss_scale="increment"`, which no longer applies when `h` is the primary state).
  Record the *effective* weight in `metrics.toml` so the comparison is auditable. Keep
  `loss_components` column names physical (`q_mse`,`h_mse`) but annotate which is the
  carried state vs the diagnostic readout per formulation.
- **Scope.** Reporting/labelling + the results-CSV schema, **not** the hybrid physics
  (which is correct). Goal: an A/B search over `model.mb_formulation` yields columns
  that are either directly comparable or clearly marked not-applicable per arm.

---


# IMPLEMENTED 21-10-2026

## Mass-balance redesign: reservoir backbone + continuity readout (`muskingum_continuity`)

**Status: IMPLEMENTED (A/B validation open).** A parallel `MassBalanceFormulation`
(`MuskingumContinuityHybrid`), selected by `mb_formulation = "muskingum_continuity"`;
the default `"theta_kinematic_wave"` and its checkpoints are untouched. Design:
[notes/mass_balance_formulations.md](notes/mass_balance_formulations.md) Â§4.2 / Â§6.

- **State and readout.** `h` (storage `S = hÂ·wÂ·l`) is the only carried state;
  `q = (Iâˆ’A)â»Â¹(I_lat âˆ’ Î”S/Î”t)` is a readout, never fed back. The encoder receives a
  zeroed `q` row (`encoder_state`), so `in_dim` and the checkpoint layout are unchanged
  but the dynamics cannot depend on `q`. *Open decision (a): pure readout (default).*
- **Backbone = implicit-Euler linear-reservoir cascade, not three-coefficient
  Muskingumâ€“Cunge.** `S_bb = P (S + Î”tÂ·I)`, `P = (I + Î”tÂ·D âˆ’ Î”tÂ·AÂ·D)â»Â¹`, `D = diag(1/K)`,
  `K_i = Î¼_h/Î¼_q` clamped to `[0.05, 100]Â·Î”t`. At daily steps the Courant number is â‰« 1, so
  the MC coefficient `Câ‚ƒ = (K(1âˆ’x) âˆ’ Î”t/2)/D` is negative for physical `K` (the build-time
  abort of the original plan would always fire). The reservoir form is the `x = 0` member
  of the Muskingum family with `P â‰¥ 0` (M-matrix) and spectrum `1/(1+Î”t/K_i) < 1` for any
  `K_i > 0`, with no Courant condition. `P` and `(Iâˆ’A)â»Â¹` are exact sparse inverses built
  in `O(nnz)` by a topological sweep (no dense inverse); the build aborts on a cyclic graph
  or a bad `mb_residual_bound`.
- **Residual is a bounded source, not an `h` increment.** `h_norm = h_bb + Ï_maxÂ·tanh(Î”)`
  (`mb_residual_bound = Ï_max`, default 0.5 Ïƒ_h), applied *after* the contracting backbone,
  so a raw increment (undamped integrator, constraint 4) is avoided. *Open decision (b).*
- **Nuance versus the design note.** The note says the residual "cannot move the
  spectrum". As implemented the guarantee is **bounded-input/bounded-state**: the carried
  map is a contraction plus a bounded term, so the state stays bounded for any weights; the
  learned part can still change the local Jacobian through the encoder's `h` input.
- **Conservation.** The readout satisfies `Î”S/Î”t = AÂ·Q + I âˆ’ Q` exactly for the produced
  storage. The only break is the output floor `Q â‰¥ 0` (outside the dynamics) and the
  non-negative `h` floor.
- **Not done / deferred.** The `K, x` bank and iterative variants; a rollout/volume-budget
  diagnostic for the hybrid (`layer_mb_analysis` only writes the contraction CSV);
  `amplification.png` shows the backbone contraction as the "gain" and no amplification.

---
# IMPLEMENTED 19-10-2026 (shipped 2026-10-21)

## Four-stage pipeline + mass-balance code separation

**Status: IMPLEMENTED.** Landed as TODO steps R1–R7 (plus the R5b inwater fix)
and verified bit-for-bit against the pre-refactor fused path on a golden run
(`sava_small_v081`, seed 42, CPU+GPU); the performance acceptance gate shows no
regression (see [CHANGELOG.md](CHANGELOG.md) 2026-10-21 and
[PERFORMANCE.md](PERFORMANCE.md), STATUS AS OF 21-10-2026). All four stages are
dispatched through a `Stage` type hierarchy in `src/run.jl`
(`run_stage(::PreprocessStage/TrainStage/InferStage/AnalyzeStage, ctx)`, folded by
`AllStages`), driven by `scripts/pipeline.jl <prep|train|infer|analyze|all>`; MB
code now lives in `src/sparse_ops.jl` + `src/massbalance.jl` with a
`MassBalanceFormulation` dispatch hierarchy (`ThetaKinematicWave` /
`NoMassBalance`), and post-hoc `mb_residual_diagnostics` makes MB analysis work on
any model. The original proposal is preserved below.

- **Restructure the monolithic `run_wflow_gnn` into four independently runnable
  stages** — (1) preprocess, (2) train, (3) infer, (4) analyze — that exchange
  **on-disk artifacts**, composed back-to-back to reproduce every artifact
  produced today. Full methodology in
  [notes/pipeline_and_massbalance_redesign.md](notes/pipeline_and_massbalance_redesign.md);
  implementation checklist in [TODO.md](TODO.md) → "Code restructuring".
- **Why.** Today preprocessing products live in memory only and inference +
  analysis are fused inside `evaluate_and_write`, so analysis cannot be re-run
  without re-inference, nor inference without the in-memory preprocessing state.
  Clean stage boundaries make each re-runnable from its predecessor's artifacts
  (the post-hoc pattern `spatial_eval.jl` already proves works).
- **Stage contract.** Preprocess owns the serialized bundle
  (`norm_stats`/`postscale`/`static_arr`/topology) **and the node↔cell mapping**
  (a pure function of staticmaps/LDD + node ordering). Inference produces **only**
  gridded `(lon,lat,time)` NetCDF matching wflow's `output.nc` as closely as
  possible. Analysis recovers node arrays by gathering river cells via the
  prep-owned mapping — no per-split node dump, no ordering drift.
- **Mass-balance separation.** Move all MB code out of `gnn.jl` into
  `src/massbalance.jl`; model formulations as a `MassBalanceFormulation` dispatch
  hierarchy (`NoMassBalance` null object replaces the `nothing` sentinel;
  `ThetaKinematicWave` wraps today's layer). The forward pass collapses to a
  three-line dispatch, removing every `isnothing(mass_balance)` branch across
  `gnn.jl`/`run.jl`/`rollout.jl`/`training.jl`. Adding a new formulation = one
  struct + a few dispatch methods + a factory + a settings enum value, with **no**
  edits to the generic GNN forward. Prefer multiple dispatch over `if/else`
  ladders throughout.
- **Layer-free MB analysis.** Add a post-hoc `mb_residual_diagnostics` that
  recomputes the kinematic-wave residual **from predicted q/h fields**, so
  mass-balance analysis works on **any** trained model (MB or not). The existing
  layer-internal diagnostics (floor fraction, stiffness) remain, dispatched on the
  formulation type and run only when a layer is present.
- **Scripts consolidation.** Promote reusable logic from `scripts/` into `src/`:
  the post-hoc helpers (`resolve_staticmaps`/`load_regrid_nc`), the layer-free MB
  routing/residual (prototyped by `check_mass_balance*.jl`), and the LR/batch
  tuning toolkit (`lr_range_test`/`autotune_train`). Thin drivers + benchmarks
  stay in `scripts/`; `smoke_test_sparse.jl` migrates into the test suite;
  `plot_ldd.jl`'s network drawing factors into `src/plot.jl`.
- **Performance invariants preserved (non-negotiable).** The `_topology_mul`
  rrule, `SparseConv` three-path dispatch + `precompute_batched`, CuSPARSE CSR
  conversions, zero-copy per-node reshape, and CPU-build-before-`gpu` ordering
  must be byte-for-byte preserved; `benchmark_message_passing`/`benchmark_rollout`
  are the before/after acceptance gate.
- **Migration** is a 7-step, individually-testable, reversible sequence;
  equivalence is checked by diffing the artifact set of a small golden run
  (`test_sava_small_v081`).

---

# IMPLEMENTED 19-10-2026 (shipped 2026-10-21)

## Subcatchment error-attribution diagnostic (post-hoc)

**Status: IMPLEMENTED** (TODO §3e, [src/attribution.jl](../src/attribution.jl),
[scripts/subcatch_error_attribution.jl](../scripts/subcatch_error_attribution.jl)).
Open: the Tier-2 counterfactual and a run on a non-diverged full `sava_v081` model.

- **Add a post-hoc analysis that attributes downstream error to the subcatchments
  that inject it**, testing the hypothesis that *some subcatchments contribute
  disproportionately to downstream error* and that this correlates with WFLOW
  subcatchment attributes (stream order, slope, soil, reservoir presence, …).
  Full methodology in
  [notes/subcatchment_error_attribution.md](notes/subcatchment_error_attribution.md);
  implementation checklist in [TODO.md](TODO.md) §3e.
- **Why incremental, not raw outlet NSE.** Discharge accumulates downstream, so a
  per-outlet metric conflates *locally injected* error with *advected* error. The
  decisive quantity is the **incremental** error a subcatchment adds on top of
  what it receives:
  `e_local(S,t) = [Q_pred(o)−Q_true(o)] − Σ_i [Q_pred(u_i)−Q_true(u_i)]`
  for outlet `o` and immediate upstream sub-outlets `{u_i}`, plus an
  advected-vs-local split and a size-normalised specific-injection metric.
- **Post-hoc, like every analysis in the repo.** Consumes only the artifacts
  `run_wflow_gnn` already writes (`output/<split>_{pred,true}.nc`) plus
  `staticmaps.nc` (`wflow_subcatch*`, `wflow_gauges*`, `wflow_ldd`,
  `wflow_streamorder`, `wflow_uparea`, channel/soil/reservoir attributes) — no
  retraining; the primary difference-of-errors method needs no forward pass
  (mirrors `scripts/spatial_eval.jl`). Only an optional Tier-2 boundary-forced
  counterfactual re-runs inference on the fixed weights.
- **Subcatchment definition.** Prefer WFLOW-native `wflow_subcatch` + `wflow_gauges`
  outlets (attributes map one-to-one); fall back to LDD-derived incremental
  catchments split at above-threshold confluences when the gauge set is too
  sparse. A configurable size threshold enforces "sufficiently large" and avoids
  the degenerate dry-headwater/floor regime.
- **Scope / open items.** Depends on a verified subcatchment **tessellation** for
  the incremental form; reservoirs/lakes break simple routing and are flagged;
  the test is under-powered on `sava_small` and should be run on `sava_v081`
  (many subcatchments). Partly unblocks the TODO §2b "no gauge concept" item.

---

# IMPLEMENTED 08-09-2026 (shipped 2026-09-11)

## System water-volume and longitudinal upstream validation diagnostics

**Status: IMPLEMENTED.** Both diagnostics landed on 2026-09-11 (see
[CHANGELOG.md](CHANGELOG.md)): `volume_budget_diagnostics` +
`plot_volume_budget` + `volume_pbias`/`volume_drift` metrics for the storage
diagnostic, and the `plot_downstream_timeseries` percentile-ladder + inset
generalisation with `[eval].upstream_points` (default 5) for the longitudinal
plots. The single-outlet `argmax(upstream_area)` convention was taken; the
multi-sink variant remains a documented future extension.

- **Add a system water-volume diagnostic on the validation set, computed for
  BOTH prediction and ground truth.** Total storage
  `V(t) = Σ_i h_i(t) · w_i · l_i` (m³) is reduced from the per-node physical
  series already produced by `rollout_mb_diagnostics` (`src/rollout.jl`), using
  `w_i · l_i = postscale_q_i / postscale_h_i` from the `MassBalanceLayer`. Both
  the `pred_h` and `true_h` matrices are summed, so `V_pred(t)` and
  `V_true(t)` are always plotted and scored together — the diagnostic is a
  **pred-vs-truth storage-tracking** signal, not a conservation check (the river
  is an open system; local conservation is already guaranteed by the hard-MB
  layer). Two framings are recorded:
  - **Storage series** `V_pred(t)` vs `V_true(t)` over the validation
    trajectory, plus headline scalars written to the run's metrics TOML:
    **volume PBIAS** (`V_pred` vs `V_true`) and **volume drift** (slope of
    `V_pred − V_true` over the horizon). Drift is the integrated signature of the
    compounding `river_h` error flagged in the `s2_stability` post-mortem, which
    per-cell RMSE hides.
  - **Budget closure** for both pred and truth: cumulative lateral inflow
    `Σ inwater·Δt`, cumulative outlet outflow `Σ q_outlet·Δt`, and
    `ΔV(t) = V(t) − V(0)`. For the hard-MB model
    `ΔV = Δt · Σ_i net_flux_i` holds by construction, so the pred budget closing
    to ~0 residual doubles as a self-test; the truth budget need not close
    through the daily discretisation, which is itself informative.
- **Rationale for computing both.** Comparing the model's storage trajectory to
  the ground-truth storage trajectory (rather than to a constant) is what makes
  volume drift/PBIAS interpretable; a truth-only or pred-only curve carries no
  accuracy information.
- **Add longitudinal upstream timeseries plots.** Generalise
  `plot_downstream_timeseries` (`src/plot.jl`, currently outlet-only via
  `argmax(upstream_area)`) to a **percentile ladder over `upstream_area`** —
  outlet (max) → headwater (min active) — with the number of points **K
  configurable via TOML, default 5**. Each node reuses `plot_timeseries`, with a
  **small per-panel inset** showing the node's location within the catchment
  (all active nodes in grid space, selected node highlighted). Purpose: localise
  *where* along the network routing error is injected vs merely advected, and
  expose the regime-dependence of the `river_h` failure (low-flow / small-area
  headwaters vs the outlet).
- **Cadence: end-of-run only.** Both diagnostics are derived from the
  date-range validation rollout already produced for the MB plots — no new
  rollout cost and no per-epoch overhead. A per-epoch volume-drift scalar is
  deferred.
- **Open question carried into TODO.** Outlet definition for the outflow term
  (single max-`upstream_area` node vs all sink nodes) is left to the engineering
  task; multi-outlet basins need the sink-set variant.

---

# STATUS AS OF 07-09-2026

## Peak-weighted Huber loss: implementation and device-safety

- **Peak-aware Huber is the supported alternative to MSE.** Training keeps MSE
  as default but supports `loss_type = :huber` with tunable
  `peak_delta`/`peak_lambda`/`peak_gamma`/`peak_w_max`. Weights are computed
  from untransformed targets and normalised by total weight, so peak emphasis is
  explicit without changing the base optimisation interface.
- **Per-node peak statistics are train-split derived and threaded explicitly.**
  `peak_node_stats` (98th percentile / IQR) is computed from training graphs and
  passed through training/loss evaluation so peak weighting is tied to local node
  regimes rather than a global pooled threshold.
- **GPU safety rule: fallback tensors must be allocated on the input device.**
  Loss-path fallback arrays now use device-matched `similar`/`fill!` helpers
  (`_same_device_fill`, `_same_device_falses`) instead of host `ones`/
  `Float32[...]` literals. This avoids CUDA kernel argument type violations in
  Zygote-broadcasted loss code and is now the required pattern for future loss
  fallback allocations.

---

## Rollout-path equivalence: fixed-horizon metric vs date-range trajectory

- **The two autoregressive rollout paths are equivalent from the same initial
  state.** The batched fixed-horizon anchor eval (`fixed_horizon_metrics` /
  `build_fixed_horizon_eval` in `src/rollout.jl`) and the single-trajectory
  date-range path (`evaluate_trajectory`) share the same state update and
  `forcing_next` convention, seed `states0` from `graphs[s].ndata.state`, and
  both run from the restored best-epoch checkpoint. A same-start regression in
  `test/test_training.jl` guards this to tolerance.
- **The earlier "discrepancy" was a diagnostic-interpretation issue, not a bug.**
  The fixed-horizon metric diverges from stiff high-flow-start anchors while the
  single benign date-range trajectory stays bounded — a real *conditional*
  instability of the MB rollout operator, surfaced (not caused) by the anchor
  eval. Model-selection metrics (`fixed_horizon.*`, `val_peak_ratio*`) are
  therefore trustworthy as a stability signal; the open work is the instability
  itself (`mb_theta`, noise, detached rollout), tracked in TODO.md.

---

# OLD STATUS AS OF 27-08-2026

*Everything below this header predates the introduction of these structured
status logs; it back-fills decisions made before 27-08-2026.*

---

## Model architecture: encode–process–decode GNN

- **GNN emulator of Wflow river routing.** The graph mirrors the river network
  (river mask + local drainage direction); nodes are river cells, edges follow
  drainage. Encode–process–decode with a physics decoder. Training data comes
  from a multi-decadal Wflow simulation.
- **Fixed-topology message passing via SpMM, not scatter-gather.** Neighbour
  aggregation `out[i] = Σ_{j→i} h[:,j]` is a sparse matrix–dense multiply
  `(A·hᵀ)ᵀ`. `SparseConv` uses cuSPARSE SpMM, fastest on GPU (~1.6× over
  `GraphConv` sparse; scatter slowest). See
  [notes/efficient_message_passing_with_edge_features.md](notes/efficient_message_passing_with_edge_features.md),
  [notes/message_passing_notes.md](notes/message_passing_notes.md).

## Mass balance: hard constraint

- **Hard constraint, not soft penalty (Option B).** The decoder predicts only
  `Δq` (`out_dim = 1`); `river_h` is derived analytically from the kinematic-wave
  water balance every step, so conservation is exact by construction. Soft
  penalty (A) and constrained-projection (C) rejected. See
  [notes/mass_balance_notes.md](notes/mass_balance_notes.md).
- **Fully-implicit discretisation (all fluxes at t+1).** Verified vs real Wflow
  output (median |residual| = 0, max 7e-4 m) against semi-implicit and lagged
  schemes which were wrong. Requires threading `forcing_next` (inwater at t+1).
- **Gradients flow through the mass balance, weighted.**
  `h_loss_weight = σ_h/(dt·σ_q)` equalises q and h gradient magnitudes
  (unweighted h-loss is ~dt× too strong, collapses q→0). Runtime-derived, not
  persisted to TOML.
- **Positivity floors in physical units, inside the layer.** A z-scored floor
  once forced ≥ mean discharge and collapsed h to 0 — a fixed bug.
- **θ-method implemented** (`mb_theta`, default `1.0` = fully implicit);
  `θ* = σ_h/(dt·σ_q)` cancels gradient stiffness. See
  [notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md).
- **Confirmed empirically:** the hard constraint is a *rollout governor* — MB-off
  diverges to Float32 overflow while MB-on stays finite over 1906 steps.
- **Manning rating is NOT used as a hard constraint.** Tested against Wflow over
  8.1M cell-steps, the daily-averaged rating holds only qualitatively (~15–23% of
  samples within ±25% even in the best wet/quasi-steady/in-bank regime; worse
  under unsteady / overbank flow). Daily-averaging a nonlinear law (Jensen +
  rising/falling-limb hysteresis) is the dominant error, not calibration. At most
  a one-sided soft cap on over-prediction; never a hard constraint. See
  [notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md).

## Normalisation & features

- **Per-variable z-score + physical postscale.** `river_q` area-normalised
  (specific discharge), `river_h` scaled by `w·l/A`; the MB layer stores inverse
  constants and evaluates physics in physical units. Stats fit on train split.
- **Nonlinear input scaling is encoder-input-only.** A selectable per-variable
  scheme (`log1p`/signed-log/`asinh` then re-z-score, auto scale = median non-zero
  `|x|`) is applied inside the model by a non-trainable first encoder layer, not in
  the data pipeline. Baking it into `ndata.state/forcing` would break the MB layer's
  affine physical-unit de-normalisation (routing additivity) and require inverses in
  `rollout.jl`/`analysis.jl`; the in-model design leaves data, `norm_stats`
  `(mean, std)`, `postscale`, MB, loss and de-normalisation untouched, so the MB
  budget is bit-unchanged and no inverse path exists to get wrong. Cost: one
  elementwise op per forward pass, and only when a scheme is configured. Static
  variables keep `STATIC_TRANSFORMS`.

## Training

- **Custom `_topology_mul` rrule** returns `NoTangent()` for the adjacency,
  avoiding a dense `∂A` (~17 GB at batched `(B·N)²`) — the OOM fix. Complementary
  to `@functor` field restriction (keeps `A` out of the optimiser). See
  [notes/training_tuning_notes.md](notes/training_tuning_notes.md).
- **Static features threaded separately** (removed from `GNNGraph.ndata`) for
  explicit data flow and a cleaner Zygote path; net GPU memory ≈ unchanged.
- **Curriculum learning:** rollout-length curriculum (`steps = [1,2,5,8,10]`)
  with a curriculum LR schedule and BPTT via `Zygote.checkpointed`; gradient
  clipping + adaptive backoff on non-finite/unstable epochs.
- **Ensemble rollout path** for GPU throughput (~2.5× per-member gain).
- **Batched training uses a precomputed block-diagonal SpMM**, ~2× faster than
  the reshape trick at batch size 16 on GPU (a single cuSPARSE call parallelises
  across all batch elements). `precompute_batched(model, B)` must be called on
  CPU **before** `Flux.gpu`, and re-synced whenever `B` changes. On CPU the
  reshape-trick fallback wins. See
  [notes/message_passing_notes.md](notes/message_passing_notes.md).
- **Curriculum LR schedule = SGDR-style warm restarts.** Monotone decay fails
  because each curriculum depth increase opens a *new, harder* objective; instead
  cosine decay within each phase, a linear warmup (`lr_warmup_epochs`) at each
  restart, and a geometrically shrinking peak (`lr_peak_decay`) per phase, with
  phase boundaries driven by `strategy.durations`. Batch↔LR follows **sqrt**
  scaling (Adam), not linear. Watch `grad_norm` at phase restarts as the
  overheating signal. See
  [notes/training_tuning_notes.md](notes/training_tuning_notes.md).
- **Autotune infrastructure** (`scripts/autotune_train.jl`): a horizon-1 LR
  range test (Leslie Smith) plus a batch-size estimator size the LR and batch
  before launch. The range test is teacher-forced / horizon-1 and blind to
  multi-step rollout collapse, so the autotuned LR is treated as an upper bound.