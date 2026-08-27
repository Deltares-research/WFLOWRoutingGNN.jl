# Implementation TODO: peak-accuracy & training-stability work

Three independent workstreams distilled from the `slurm-246431` post-mortem and the
`peak_discharge_loss_functions_and_metrics.md` research notes. Each item is
self-contained and should land behind a toggle so it can be A/B-tested against the
current behaviour.

---

## 1. New feature scaling / normalization

Replace the uniform per-variable z-score with distribution- and physics-aware scaling.

**Key constraint:** `river_q`, `river_h`, and `river_inwater` are denormalized
**linearly** inside the hard mass-balance layer
(`state[1]*σ_q+μ_q`, `forcing[1]*σ_inwater+μ_inwater`, etc. in `src/gnn.jl`).
A nonlinear transform on these channels breaks the physics unless it is also
inverted inside the MB layer — so keep them linear.

- [ ] **Physics channels (`q`, `h`, `inwater`): keep linear z-score.**
      Do NOT apply log/asinh (would require inverting inside the MB layer).
      Their heavy tails are handled by the new loss (item 2), not the input transform.
- [ ] **Area-normalize `river_inwater`** like `river_q` (divide by upstream/drainage
      area → specific lateral inflow). This is a *linear* rescale, so fold it into
      `σ_inwater`/`μ_inwater` — physics-safe, and puts inflow in the same per-area
      space as discharge (they are summed in `net_flux`).
- [ ] **Skewed static geomorphic features → `log1p` then z-score:**
      `river_slope` (orders of magnitude, log-normal), `river_width`,
      `river_depth`, `river_length`. Encoder-only, so nonlinear transforms are safe.
- [ ] **Bounded static `river_manning_n` → plain z-score** (narrow range; log adds nothing).
- [ ] **Optional new static feature: `log(upstream_area)`** as an explicit
      "position in drainage network" signal (already loaded as the postscale multiplier).
- [ ] Fit every statistic (means, stds, area factors) on the **training split only**.
- [ ] Implement via the existing per-variable `VAR_SCALERS` mechanism in
      `src/preprocess.jl` so each transform is independently reversible.
- [ ] Add a diagnostic dump of per-node max / 99.9th-percentile of normalized targets
      (train vs val, both basins) to confirm tail reduction.

**Touch points:** `src/preprocess.jl` (`VAR_SCALERS`, `scale_river_*!`, z-score pass),
`src/gnn.jl` (only if `σ_inwater` fold-in needs a matching change), `src/schema.jl`
(if `upstream_area` is added as a static var).

---

## 2. Peak-weighted loss with toggle vs. current MSE

Implement the recommended **smooth, per-node, peak-weighted Huber** loss
(notes §3.3, §4), toggleable against the current MSE for clean comparison.

- [ ] Add a strategy-gated loss selector so the current MSE path
      (`src/strategy.jl` `loss_function`) remains the default and the new loss is
      opt-in (e.g. a `loss_kind` / `peak_weighting` field on `TrainingStrategy`,
      surfaced through the config TOML).
- [ ] **Base error → Huber** with tunable `δ` (bounds each residual's gradient;
      also a stability fix for the `1e6` gradient blow-up).
- [ ] **Per-node peak weights** `w_i,t = min(1 + λ·[max(0,(y-u_i)/s_i)]^γ, w_max)`,
      with per-node threshold `u_i` (e.g. training 98th percentile) and robust scale
      `s_i` (e.g. per-node IQR). "Basin" in the notes ⇒ "node" here.
- [ ] **Build weights from the untransformed (physical / area-normalized) target**,
      not a log-space target — otherwise peaks get de-emphasized.
- [ ] **Normalize by `Σw`** (divide weighted loss by sum of weights) so gradient
      scale does not change mechanically with `λ`.
- [ ] Apply the same peak weighting to the **`h` (depth) channel**, and revisit
      `h_loss_weight` (currently ~0.0462, ~22× below `q`) if flood height is a
      first-class output.
- [ ] Precompute per-node `u_i` / `s_i` in the preprocessing pass (training split only).
- [ ] **Diagnostics:** log `C_peak` (fraction of weighted loss from peak cells) and
      the fraction of gradient norm from peak cells, per epoch (notes §4.9, §15).
- [ ] Sensible defaults for first runs: `γ=1`, `λ ∈ {0.5,1,2}`, `δ ∈ {0.5,1,2}`
      (standardized units), `w_max` set by inspecting the weight distribution.
- [ ] (Deferred, only if a specific deficiency appears) rising-limb/derivative term
      `L_ΔQ` for timing; asymmetric penalty for systematic underprediction.

### 2a. Parameter estimation (avoid blind grid search)

Four of the seven parameters are **computed** from data/residuals, one is set by
**gradient balancing**, and only `λ` (optionally `γ`) needs an actual sweep — and
that sweep is guided by diagnostics, not raw validation loss. All target-derived
quantities are fit on the **training split only** (notes §1, §11).

- [ ] **`u_i` (per-node threshold) — compute directly.** Per-node empirical
      percentile of training-period discharge. Use the **operational warning
      threshold** where one exists (notes §4.2); otherwise start at the **98th
      percentile** and compare {95, 98, 99} as a small discrete set, not a
      continuous knob. Caveat: equal sample fraction ≠ equal hydrological impact.
- [ ] **`s_i` (per-node scale) — compute directly.** Robust dispersion of each
      node's training discharge; prefer **IQR** over SD (IQR is not inflated by the
      peaks being weighted) (notes §4.3). One-line addition to existing per-node stats.
- [ ] **`δ` (Huber knee) — estimate from baseline residuals.** Run unweighted
      Huber/MSE first; set `δ` near the ~80–90th percentile of `|residual|` in
      standardized target units (quadratic for typical errors, linear for the tail)
      (notes §3.3). Use the `{0.5,1,2}` grid only as a sanity range around that estimate.
- [ ] **`h_loss_weight` (q/h balance) — set by gradient-norm balancing.** Measure
      per-channel gradient contributions of the q and h loss terms; choose the weight
      that puts them on a comparable (or deliberately chosen) ratio (notes §7). Current
      ~0.0462 (~22× below q) is likely undertraining depth if flood height matters.
- [ ] **`λ` (peak-weight strength) — sweep, guided by `C_peak`.** Fix `γ=1`, sweep
      `λ ∈ {0.5,1,2,4,8}` (notes §4.8, §12 Stage B). Choose λ so peaks drive a
      *deliberate* share of the loss (e.g. target `C_peak ≈ 0.3–0.5`); stop when the
      peak-gradient-fraction diagnostic shows a tiny cell fraction producing almost
      all the gradient (§4.9). Select on a **peak-region validation metric under
      constraints** (false-alarm ratio, KGE degradation — §14), never on weighted
      validation loss (non-comparable across λ).
- [ ] **`γ` (relevance exponent) — leave at 1, tune last.** Only adjust if severity
      response is inadequate; do not co-tune with λ (partially confounded). γ<1 spreads
      weight to moderate exceedances, γ>1 concentrates on extremes (notes §4.8).
- [ ] **`w_max` (weight cap) — set from the weight distribution.** Stability guard,
      not a performance knob: compute `w_i,t` on training data and cap at a high
      percentile so no single cell dominates (notes §4.5). Tune by watching the
      gradient-fraction diagnostic, not validation loss.

**Estimation workflow (staged, keeps the search low-dimensional — notes §12):**

1. Precompute `u_i`, `s_i` per node from the training split (no tuning).
2. Baseline unweighted-Huber run → read `δ` from residuals; record per-channel
   gradient norms → set `h_loss_weight`.
3. Fix `γ=1`, sweep `λ`, selecting via `C_peak` + peak-gradient-fraction +
   peak-region validation metric.
4. Set `w_max` from the resulting weight distribution.
5. Only if needed: adjust `γ`, or add asymmetry/timing terms.

**Touch points:** `src/strategy.jl` (`loss_function`, `one_step_loss`, strategy struct),
`src/preprocess.jl` (per-node `u_i`/`s_i`, baseline-residual δ estimate),
`src/training.jl` (epoch diagnostics, per-channel gradient norms),
config parsing in `src/run.jl` / `src/training.jl`.

### 2b. Validation metrics needed for tuning (gap analysis)

Audit of the **currently implemented** validation metrics against the notes
catalog (§15–§16), scoped to what item 2 / §2a actually consume. Several tuning
"knobs" above reference diagnostics that **do not exist yet** — those must ship
*with* the weighted loss, not after.

**Already implemented (keep):**

- Per-epoch: `val_rollout`, `val_1step` (+ per-channel `val_q_1step`/`val_h_1step`),
  q→h amplification, `grad_norm`, `val_fixed_rmse` (fixed-horizon discharge RMSE,
  physical units), `val_peak_ratio` (`max|q_pred|/max|q_truth|`).
- Per-cell spatial maps (`src/postprocess.jl` `spatial_error_metrics`): `rmse`,
  `bias`, `relbias`, `overpred_freq`, `peak_err` (signed series-max diff),
  `peak_lag` (cross-correlation timing lag), `nbias`, `nse`; plus ramp-rate
  `corr(e,g)` overprediction diagnostic.
- Covers the whole-hydrograph "watch" (RMSE, NSE, bias, timing, peak ratio) well;
  `peak_err`/`peak_lag`/`val_peak_ratio` already give real peak-magnitude and
  peak-timing signal — a genuine strength, do not remove.

**Tier 1 — required to *tune the new loss* (land alongside item 2):**

- [ ] **`C_peak`** — fraction of the (weighted) loss coming from cells/timesteps
      above `u_i`. The stopping rule for the `λ` sweep (§2a, notes §4.9). Cheap:
      reuses the per-node residuals and weights the loss already forms.
- [ ] **Peak-gradient-fraction** — share of the total gradient norm attributable to
      peak cells. The `λ`/`w_max` guard ("tiny cell fraction producing almost all
      the gradient"). Needs a masked second backward or per-cell grad accumulation.
- [ ] **Per-channel (q, h) gradient norms** — to set `h_loss_weight` by balancing
      instead of the current arbitrary `~0.0462` (§2a, notes §7).
- [ ] **`RMSE_high` / `MAE_high`** — RMSE/MAE restricted to targets above per-node
      `u_i`. The peak-region **selection** metric for the `λ` sweep (§14). Distinct
      from `val_peak_ratio` (an amplitude ratio over the whole window, not an
      above-threshold error).
- [ ] **Weight summary stats** (`max`, 99th pct of `w_i,t`) once weights exist —
      to set `w_max` from the distribution (§2a, notes §4.5).

**Tier 2 — model selection (§14 constrained/Pareto, not weighted val loss):**

- [ ] **KGE + its r / α / β decomposition** — the single most valuable
      whole-hydrograph addition; NSE alone can select models that reproduce annual
      peaks poorly (notes §2.6, §16.1). A few lines from accumulators already kept
      (means, stds, Σ pred·truth). Required for the "KGE degradation" selection
      constraint in §2a's `λ` step.
- [ ] **MAE** and a **basin/gauge-level PBIAS** (not just per-cell `bias`/`relbias`)
      — trivial given existing sums.
- [ ] **Event peak-magnitude error** and **event volume bias (FHV)** at key gauges
      — the operational peak metrics (§16.2).
- [ ] **Report at key gauges, not just pooled** (notes §2.7): outlet + major
      confluences separately, so a few large reaches don't dominate. Reduce the
      existing per-cell maps at gauge nodes — cheap.

**Tier 3 — only if peak *detection* becomes a distinct objective (§16.3):**

- [ ] **POD / false-alarm ratio / CSI** from thresholded exceedance. Needed only if
      an exceedance head is added; the false-alarm-ratio selection constraint in
      §2a's `λ` step depends on this.

**Touch points:** `src/training.jl` (per-epoch `C_peak`, peak-gradient-fraction,
per-channel grad norms, `RMSE_high`/`MAE_high`, weight stats),
`src/postprocess.jl` (KGE + r/α/β, MAE, gauge-level PBIAS/FHV, gauge-node reduction),
`src/rollout.jl` (extend `fixed_horizon_metrics` if peak-region scalars are computed
in the rollout path), config/selection logic in `src/run.jl`.

---

## 3. Pushforward / detached-rollout trick behind a toggle

Replace full-BPTT curriculum loss with a stop-gradient unroll to cut training time
and stop the multiplicative gradient escalation (`1e6 → 1e16` across curriculum phases).

- [ ] Add a strategy toggle (e.g. `rollout_grad = :full_bptt | :pushforward | :detached`)
      defaulting to the current full-BPTT behaviour in `src/strategy.jl` `loss_function`.
- [ ] **Detached / truncation-length-1 variant (preferred):** at each unrolled step,
      `detach` (stop-gradient) the incoming state, compute that step's loss, and
      backprop only through the single step. Preserves per-step hydrograph supervision
      with O(1) tape depth and bounded gradients.
- [ ] **Pure pushforward variant (optional):** unroll all intermediate steps under
      stop-gradient and backprop only the final step (Brandstetter et al. 2022).
- [ ] Ensure compatibility with the existing `Zygote.checkpointed` path
      (the detached variant should make gradient checkpointing unnecessary — verify
      memory drops and remove/skip the checkpoint recompute when toggled on).
- [ ] Confirm the runtime stability machinery still behaves: with BPTT removed,
      expect the ~170 LR-backoff events and `phase_backoff_factor` firing to largely
      disappear.
- [ ] Benchmark step time and peak GPU memory in the deepest curriculum phase
      (`steps=10`) full-BPTT vs. detached vs. pushforward.

**Touch points:** `src/strategy.jl` (`loss_function` rollout loop, strategy struct),
`src/training.jl` (verify grad-norm / backoff interaction), config parsing.

**Note:** this fixes rollout *depth* instability and training time, but NOT the
epoch-2 single-step blow-up (that is the loss-conditioning problem addressed by
items 1 & 2). Pair it with those, do not treat it as a standalone fix.
