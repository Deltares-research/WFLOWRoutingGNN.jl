This document serves as a changelog of architectural and design decisions.

Each entry records *what* was decided and *why*, with pointers to the research
note that holds the full reasoning. Newest at the top.

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