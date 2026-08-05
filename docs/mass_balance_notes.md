# Mass Balance Layer — Design, Gradients & Loss Weighting

A consolidated reference of the discussions on the `MassBalanceLayer` — how it
enforces the kinematic-wave water balance as a **hard constraint** in the forward
pass, how (and whether) it participates in gradient computation, and how the q/h
loss terms are weighted. Records the implemented design, the several approaches we
tried and reverted, and literature context.

Source: [src/gnn.jl](../src/gnn.jl) (`MassBalanceLayer`, `mb_diagnostics`),
[src/strategy.jl](../src/strategy.jl) (loss + weighting),
[src/run.jl](../src/run.jl) (`build_gnn_model`, weight derivation),
[scripts/check_mass_balance.jl](../scripts/check_mass_balance.jl) (data-side
verification of the discretisation scheme).

Companion docs: [docs/message_passing_notes.md](message_passing_notes.md) (the
sparse-adjacency SpMM the routing sum reuses) and
[docs/training_tuning_notes.md](training_tuning_notes.md).

---

## 1. The goal: hard mass conservation, not a soft penalty

Requirement: the emulator must **respect mass balance at every cell at every
timestep**, so that the model *has no choice* but to conserve water. For each
river cell $i$:

$$(h_i^{t+1} - h_i^{t})\, w_i l_i \;=\; \Delta t\left(\sum_{j\to i} Q_j + Q_{\text{in},i} - Q_i\right)$$

incoming = upstream discharge + lateral inflow (`river_inwater`), outgoing = the
cell's own discharge, balanced against the change in storage $h\,w\,l$.

Three ways to impose this were weighed:

| Option | Idea | Verdict |
|---|---|---|
| **A — soft** | add a physics-residual penalty $\lambda\lVert r\rVert^2$ to the loss | rejected: violations only penalised, not eliminated; extra hyperparameter |
| **B — hard post-correction** | decoder predicts only $Q$; derive $h$ analytically from the balance | **chosen** |
| **C — constrained output layer** | project a joint $(Q,\Delta h)$ prediction onto the constraint manifold | rejected: exotic, hard to implement |

Option B makes the constraint exact by construction: the decoder outputs only
$\Delta q$ (`out_dim = 1`), and $h^{t+1}$ is computed deterministically. This is
the encode–process–decode + physics-decoder pattern.

*References:* physics-informed / hard-constraint learning — Raissi, Perdikaris &
Karniadakis, *Physics-Informed Neural Networks*, J. Comp. Phys. 378 (2019),
arXiv:1711.10561; hard constraints in NN outputs — Beucler et al., *Enforcing
Analytic Constraints in Neural Networks Emulating Physical Systems*, PRL 126
(2021), arXiv:1909.00912; differentiable/hybrid physics — Shen et al.,
*Differentiable modeling*, and the kinematic-wave routing in wflow_sbm
(van Verseveld et al., *Wflow_sbm v0.7.3*, GMD 2024).

---

## 2. Getting the discretisation right (data-side verification)

Before trusting the constraint we verified **which discretisation wflow actually
uses** against real output, in
[scripts/check_mass_balance.jl](../scripts/check_mass_balance.jl). Three schemes
were compared on the Sava basin (8.1 M cell-steps):

| Scheme | Flux timing | median \|residual\| | max \|residual\| | verdict |
|---|---|---|---|---|
| (A) semi-implicit corrected | $\Sigma q[t{+}1] + iw[t] - q[t{+}1]$ | 5e-4 m | 213 m | ✗ wrong inwater timing |
| **(B) fully-implicit** | $\Sigma q[t{+}1] + iw[t{+}1] - q[t{+}1]$ | **0 m** | **7e-4 m** | ✓ machine-precision exact |
| (C) lagged upstream | $\Sigma q[t] + iw[t] - q[t{+}1]$ | 1.3e-2 m | 4689 m | ✗ wrong by design |

**Wflow is fully implicit** — all three flux terms are at $t{+}1$. That looked
problematic (each cell's $h^{t+1}$ needs its *own* and its *neighbours'* $q^{t+1}$
— an implicit coupling), but it resolves cleanly: the GNN predicts $q^{t+1}$ for
**all nodes in one forward pass**, so when node $i$ forms its upstream sum the
neighbours' $q_j^{t+1}$ are already available via message passing. Hence:

1. GNN predicts $q^{t+1}$ everywhere (message passing supplies $\Sigma_j q_j^{t+1}$),
2. $h^{t+1}$ is derived analytically — exact, no residual.

This required threading `forcing_next` (i.e. $iw[t{+}1]$) through every call
site (`WflowGNN` forward, `rollout`, `loss_function`, `one_step_loss`) — the
fully-implicit scheme uses lateral inflow at $t{+}1$, not $t$.

---

## 3. Interaction with scaling & normalisation

The physics must be evaluated in **physical units**, so the layer stores the
inverse-transform constants and undoes both the per-node *postscale* and the
global *z-score* before applying the balance, then re-applies them:

| Variable | stored (z-score space) | → physical |
|---|---|---|
| `river_q` | $(q/A - \mu_q)/\sigma_q$ | $\times\sigma_q+\mu_q$, then $\times A$ |
| `river_h` | $(h\,wl/A - \mu_h)/\sigma_h$ | $\times\sigma_h+\mu_h$, then $\times A/(wl)$ |
| `river_inwater` | z-scored forcing | undo z-score |

The precomputed `postscale_q` ($=A$) and `postscale_h` ($=A/(wl)$) vectors (built
in preprocessing via the `VAR_SCALERS` dict) are reused directly;
`ph_over_pq = postscale_h ./ postscale_q = 1/(wl)` is cached so the mass balance
is one broadcast:

```
h_phys_new = h_phys_curr + dt · ph_over_pq · (upstream_q + inwater_phys − q_phys_new)
```

The upstream sum `upstream_q` uses the **same sparse-adjacency SpMM** as
`SparseConv` (routing-only adjacency, no self-loops), with the identical
single-graph / precomputed-block-diagonal / reshape-fallback dispatch — see
[docs/message_passing_notes.md](message_passing_notes.md).

---

## 4. Should the constraint carry gradients? (the long saga)

This was the hardest part. The chain rule through the layer is exact:

$$\frac{\partial h_{\text{norm}}}{\partial q_{\text{norm}}} = -\,\frac{dt\,\sigma_q}{\sigma_h}$$

The `postscale` factors cancel; the *only* gradient path from $h$ back to the
decoder is through $q_{\text{phys,new}}$ (the upstream sum uses the input state,
so contributes nothing). With `dt = 86400 s` and $\sigma_q/\sigma_h = O(1)$, an
**unweighted** h-MSE produces a gradient ~$dt$ (tens of thousands ×) larger than
the q-MSE gradient. The optimizer then ignores q error and drives q toward
whatever constant minimises the average h constraint — i.e. $q\approx 0$ in
z-score space.

The sequence of fixes (each tried, most reverted):

1. **`Flux.ignore_derivatives` around h** — train q on q-loss only; h stays
   physically consistent at rollout but contributes no gradient. Worked, but…
2. **A z-scored floor `q_new = max(0, …)`** was added ("q can't be negative").
   **Bug:** z-scored 0 = *mean* discharge, not zero flow — this forced ≥ mean
   discharge everywhere, drove $h$ negative → floored to 0 → **h collapsed to 0
   for the whole rollout** and the loss got stuck ~1000× too high. Fix: **floor q
   in physical space inside the layer** (`q_phys_new = max(0, …)`), never in
   z-score space; the GNN may predict any z-scored value.
3. **Re-enable the h gradient but rescale it** — set
   `h_loss_weight = σ_h / (dt · σ_q)` so the two gradient magnitudes match. This
   removed the "stuck loss", but exposed a deeper issue…
4. **Direction conflict** — balancing the *magnitude* didn't fix the *direction*:
   raising q to match `target_q` also raises h (already too high), so the two
   signals partly cancel → near-zero $\Delta$ → constant q prediction. Weighting
   alone can't fix opposing gradients.
5. Options weighed for the conflict: (A) **phased curriculum** — train q-only
   (`h_loss_weight = 0`) then switch on h; (B) **soft residual** with independent
   heads; (C) **q-only training, mass balance at inference only**. Briefly used
   **C2** (ignore h + `h_loss_weight = 0`), but with a hard mass-balance layer in
   the training rollout, accumulated q error still poisons the fed-back h.

### Where it landed (current code)
The **weighted-gradient** approach (step 3) is what ships:
`h_loss_weight = σ_h / (dt · σ_q)`, computed in `build_gnn_model`
([src/run.jl](../src/run.jl)) from the layer's own stats so it tracks any change
in normalisation, and applied in `loss_function`:

```julia
loss += Flux.mse(pred[1:1,:], tgt[1:1,:]) +
        strategy.h_loss_weight * Flux.mse(pred[2:2,:], tgt[2:2,:])
```

- Gradients **do** flow through the mass balance (no `ignore_derivatives`).
- The weight equalises the q and h gradient magnitudes so neither dominates.
- Physical floors: `q_phys_new` and `h_phys_new` are clamped at 0 **in physical
  units**, inside the layer — never in z-score space.
- `h_loss_weight` is **runtime-derived, not persisted** to TOML (it defaults to
  `1f0` for non-river domains, which have no mass balance).

### Design consequences worth remembering
- The decoder emits only `Δq` for river; `river_h` is analytic. Existing
  two-output river checkpoints are therefore incompatible.
- `Flux.trainable(::MassBalanceLayer) = (;)` — the physics constants are never
  optimised. `Functors.@functor MassBalanceLayer (postscale_q, postscale_h,
  ph_over_pq)` restricts traversal; explicit `Flux.gpu`/`Flux.cpu` overloads
  convert `A_routing` / `A_routing_batched` between `SparseMatrixCSC` and
  `CuSparseMatrixCSR` (same rationale as `SparseConv`; see the message-passing
  doc for why `@functor` restriction + overloads are needed rather than
  `@layer trainable=`).
- When `h_loss_weight = 1` the reported loss is `mse_q + mse_h`, i.e. ~2× a
  full-matrix `mse` — keep this in mind comparing runs with/without mass balance.
- `mb_diagnostics` / `rollout_mb_diagnostics` expose every physical term
  (`upstream_q`, `inwater_phys`, `net_flux`, `h_phys_raw` before the floor, …)
  and can re-run the balance with **true** q to confirm the equation itself
  holds during validation.

---

## 5. Alternatives / unexplored paths

- **Soft physics-residual loss (Option A)** — never implemented. Would decouple
  the q/h heads (independent gradients, no cancellation) at the cost of only
  *approximate* conservation and an extra $\lambda$. Worth revisiting if the hard
  constraint's gradient conflict resurfaces at scale.
- **Teacher-forcing h during training (Option C1)** — carry `[q_pred, h_true]`
  through the training rollout, mass-balance h only at inference. Cleaner
  train-time stability; not adopted because of the train/inference input gap
  (mitigated in principle by input noise).
- **Phased curriculum on `h_loss_weight`** (0 → small value) — discussed as the
  lowest-risk way to sidestep the direction conflict; the LR/step curriculum
  exists ([docs/training_tuning_notes.md](training_tuning_notes.md)) but a
  dedicated h-weight schedule was not wired in.
- **Iterative implicit solver** — instead of the single-pass "predict all q then
  derive h", solve the fully-implicit system with a fixed-point/Newton iteration
  per step. Rejected: the single forward pass already satisfies the implicit
  coupling exactly, so an iterative solver adds cost with no accuracy gain.
- **Semi-implicit inwater timing (scheme A)** — leaving `river_inwater` at $t$
  instead of $t{+}1$. Rejected by the data check (§2): non-zero residual vs
  wflow's exact fully-implicit balance.
- **Multiple edge features in the routing sum** — incompatible with the single
  scalar-adjacency SpMM; would need per-feature matrices or a `propagate`
  fallback (noted in the message-passing doc).

---

## 6. Quick reference

| Question | Answer |
|---|---|
| Hard or soft constraint? | **Hard** — decoder predicts q, h is analytic |
| Which discretisation? | **Fully implicit** (all fluxes at t+1); verified exact vs wflow |
| Why thread `forcing_next`? | fully-implicit uses `river_inwater` at t+1 |
| Do gradients flow through h? | **Yes**, weighted by `σ_h/(dt·σ_q)` |
| Why the weight? | `∂h_norm/∂q_norm = −dt·σ_q/σ_h`; unweighted h-loss is ~dt× too strong |
| Where to floor q, h at 0? | **physical units, inside the layer** — never z-score space |
| Is `h_loss_weight` in TOML? | No — derived at runtime from the layer's stats |
| Upstream sum implementation | sparse-adjacency SpMM (shared with `SparseConv`) |

---

## References
- Raissi, Perdikaris & Karniadakis, *Physics-Informed Neural Networks*, J. Comp. Phys. 378 (2019); arXiv:1711.10561.
- Beucler, Pritchard, Rasp et al., *Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems*, Phys. Rev. Lett. 126 (2021); arXiv:1909.00912.
- Battaglia et al., *Relational Inductive Biases, Deep Learning, and Graph Networks*, arXiv:1806.01261 (2018) — encode-process-decode framing.
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, arXiv:1704.01212 (2017) — propagation as adjacency products.
- van Verseveld et al., *Wflow_sbm v0.7.3, a spatially distributed hydrological model*, Geosci. Model Dev. 17 (2024) — the kinematic-wave routing being emulated.
- Shen et al., *Differentiable modelling to unify machine learning and physical models*, Nat. Rev. Earth Environ. (2023) — hybrid differentiable physics context.
