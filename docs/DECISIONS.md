This document serves as a changelog of architectural and design decisions.

Each entry records *what* was decided and *why*, with pointers to the research
note that holds the full reasoning. Newest at the top.

---

# PROPOSED 19-10-2026

## Four-stage pipeline + mass-balance code separation

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

# PROPOSED 19-10-2026

## Subcatchment error-attribution diagnostic (post-hoc)

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