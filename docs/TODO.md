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

- [ ] Strategy-gated loss selector; MSE stays default.
- [ ] Huber base error with tunable `δ`; per-node peak weights
      `w_i,t = min(1 + λ·[max(0,(y-u_i)/s_i)]^γ, w_max)`; weights from the
      untransformed target; normalise by `Σw`; apply to the `h` channel too.
- [ ] Precompute per-node `u_i`/`s_i` (train split).
- **2a. Parameter estimation (avoid blind grid search):** compute `u_i` (≈98th
      pct), `s_i` (IQR); set `δ` from baseline residuals; set `h_loss_weight` by
      gradient-norm balancing; sweep only `λ` (guided by `C_peak`); leave `γ=1`;
      set `w_max` from the weight distribution.
- Touch: `src/strategy.jl`, `src/preprocess.jl`, `src/training.jl`, config in
  `src/run.jl`.

## 2b. Validation metrics needed for tuning

- **Tier 1 (with the loss):** `C_peak`, peak-gradient-fraction, per-channel
  (q,h) grad norms, `RMSE_high`/`MAE_high`, weight summary stats.
- **Tier 2 (model selection §14):** KGE + r/α/β, MAE, gauge-level PBIAS,
  event peak error + FHV, per-gauge (not pooled) reporting.
- **Tier 3 (only if a detection head is added):** POD / false-alarm ratio / CSI.
- Touch: `src/training.jl`, `src/postprocess.jl`, `src/rollout.jl`, `src/run.jl`.

## 3. Pushforward / detached-rollout trick (toggle)

- [ ] `rollout_grad = :full_bptt | :pushforward | :detached`, default full-BPTT.
- [ ] Detached (truncation-length-1) variant preferred: per-step loss, backprop
      one step, O(1) tape depth — should make gradient checkpointing unnecessary.
- [ ] Benchmark step time & peak memory vs. full-BPTT at `steps=10`.
- **Note:** fixes rollout-depth instability & training time, *not* the epoch-2
  single-step blow-up (that is items 1 & 2). Pair it with them.
- Touch: `src/strategy.jl`, `src/training.jl`, config.