This document serves as an archive of experiments, listing evaluation metrics and some basic diagnoses and hypotheses of each experiment.

Experiment configs live under `experiments/`. Newest at the top.

---

## Small-basin stability-lever sensitivity — `sava_small_v081_s2_stability`

**NAME:** `sava_small_v081_s2_stability` (8-run `box` hparsearch, small Sava
basin, MB on, `mb_theta = 1.0`, 8 layers / hidden 64 / mlp 2, 75,009 params,
250 epochs). Config:
[experiments/sava_small_v081_s2_stability/config.toml](../experiments/sava_small_v081_s2_stability/config.toml).

**⚠ Validity caveat — read first.** The intended 2×2×2 design (grad_clip ×
h_loss_scale × curriculum) did **not** execute as specified. All 8 runs trained
with `grad_clip = 1.0`; the `grad_clip = 0.1` arm was silently ignored (see
HYPOTHESES → *bug 1*). The search therefore reduced to a **2×2 factorial with 2
replicates** (h_loss_scale × curriculum), which is still informative but leaves
the grad_clip question **unanswered**. Run→lever mapping (from each run's saved
`model/train_settings.toml`):

| run | h_loss_scale | curriculum | grad_clip (actual) |
|---|---|---|---|
| hps1, hps3 | absolute  | full `[1,2,5,8,10]` | 1.0 (both) |
| hps2, hps4 | increment | full `[1,2,5,8,10]` | 1.0 (both) |
| hps5, hps7 | absolute  | short `[1,2,4]`     | 1.0 (both) |
| hps6, hps8 | increment | short `[1,2,4]`     | 1.0 (both) |

**SUMMARY.** Teacher-forced one-step **discharge** fit is excellent and nearly
identical across *all* 8 runs (`val_q_1step ≈ 0.013–0.016`). Everything else
splits sharply on two factors:

| run | h_scale · curric | val_q_1step | val_rollout | best_val_rollout | val_h_1step | fixed_horizon best_val_rmse | final_peak_ratio | max_grad_norm | n_backoffs | best_epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| hps1 | abs · full  | 0.0151 | 0.0673 | 0.0319 | 0.638 | 1.22e7 | 1.09e20 | 1.18e4 | 0 | 119 |
| hps3 | abs · full  | 0.0155 | 0.0626 | 0.0319 | 0.622 | 7.13e5 | 1.20e28 | 1.32e4 | 0 | 9 |
| hps2 | inc · full  | 0.0151 | **0.0262** | **0.0135** | 0.954 | 2.47e5 | 2.85e32 | 6.20e5 | 2 | 17 |
| hps4 | inc · full  | 0.0150 | **0.0243** | **0.0135** | 0.935 | 6.36e6 | 9.52e30 | 1.35e3 | 0 | 43 |
| hps5 | abs · short | 0.0136 | 31.68 | 3.237 | 148.9 | 4.29e6 | 2.83e6 | 2.49e9 | 2 | 221 |
| hps7 | abs · short | 0.0141 | 185.7 | 2.386 | 150.8 | 1.56e6 | 6.68e11 | 1.05e9 | 7 | 8 |
| hps6 | inc · short | 0.0134 | 130.4 | 0.203 | 143.0 | 5.77e5 | 1.61e9 | 4.09e7 | 8 | 21 |
| hps8 | inc · short | 0.0133 | 12.84 | 0.223 | 143.9 | 1.25e5 | 5.00e7 | 3.96e6 | 12 | 34 |

Three headline results: (1) **`increment` beats `absolute`** on the full
curriculum (~2.6× lower rollout loss); (2) the **short `[1,2,4]` curriculum
diverges catastrophically** regardless of h-scale (rollout loss 13–186,
grad-norms up to 2.5e9, 2–12 backoffs); (3) the **fixed-horizon free rollout
diverged in _every_ run** — `best_val_rmse ≥ 1.2e5` and `peak_ratio` up to
1e32 even for the otherwise-converged full-curriculum runs.

**IMPROVEMENTS (vs the `absolute` baseline / archived expectation).**
- **`h_loss_scale = increment` clearly wins on the full curriculum.** Rollout
  loss 0.024–0.026 vs 0.063–0.067 for `absolute` (~2.6×), and best-val-rollout
  0.0135 vs 0.0319 (~2.4×). *Evidence:* hps2/hps4 vs hps1/hps3. This is
  consistent with [DECISIONS.md](DECISIONS.md) — increment removes the stiff
  q→h gradient amplification of the hard MB decoder — and matches the shipped
  `mb_sweep` choice (`h_loss_scale = "increment"`).
- **Discharge one-step skill is robust to every lever tested** (`val_q_1step`
  0.013–0.016 across all 8; spatial `river_q` NSE ≈ 0.97 for the full-curriculum
  runs, e.g. hps1 0.966, hps2 0.975). The discharge head itself is not the
  fragile part.

**DEGRADED / FAILURE MODES.**
- **Short `[1,2,4]` curriculum → training divergence.** rollout loss 13–186,
  `val_h_1step` 143–151 (vs ~0.6–0.95 full), grad-norms 4e6–2.5e9, 2–12 LR
  backoffs. Crucially the **discharge stays fine** (`val_q_1step ≈ 0.013`) while
  the **MB-derived depth `h` explodes** — the failure is in the autoregressive
  h/free-rollout path, not the discharge prediction. *Evidence:* hps5–8.
- **Fixed-horizon free rollout is unstable in all 8 runs**, including the
  "good" full-curriculum increment runs (hps2 `best_val_rmse` 2.47e5,
  `peak_ratio` 2.85e32; hps4 6.36e6). No epoch produced a stable 30-step,
  32-anchor free rollout. This is **not** a contradiction with the stable
  222-step date-range trajectory (see HYPOTHESES → *conditional instability*):
  the metric samples 32 start states incl. stiff high-flow ones, the date-range
  rollout samples one benign winter state. It *retro-weakens* the archived
  `mb_sweep` "finite 1906-step rollout" claim (also a single benign trajectory)
  and is the most important signal from this batch.
- **Huge replicate spread in the unstable regime:** the two `absolute · short`
  replicates give rollout loss 31.7 vs 185.7 — an order of magnitude apart at
  identical settings, confirming these runs sit in a chaotic/divergent basin.

**HYPOTHESES.** *(evidence vs speculation flagged)*
- **Bug 1 (evidence): the grad_clip axis never ran.** All 8 saved
  `train_settings.toml` show `grad_clip = 1.0`. Root cause traced in code: the
  hparsearch inline `TrainSettings(...)` constructor in
  [src/hparsearch.jl](../src/hparsearch.jl) omits `grad_clip` (also
  `lr_warmup_epochs`, `lr_peak_decay`), so the `train.grad_clip` override is
  written into the config dict but never consumed — it defaults to `1.0`. The
  single-run path ([src/run.jl](../src/run.jl) line 120) *does* read it, so this
  is specific to the search path. **This is a code bug for the engineering
  agent, not a config error** (the config's `search_space` was correct).
- **Short-curriculum divergence is a real effect, not a schedule artifact
  (evidence + speculation).** I checked the loop:
  [src/strategy.jl](../src/strategy.jl) `update_steps!` holds `steps[end]` and
  [src/training.jl](../src/training.jl) `curriculum_lr` holds `lr_final` for
  epochs beyond `sum(durations)` — so the `durations` sum (50) ≠ `epochs` (250)
  mismatch is handled gracefully (epochs 51–250 run at steps=4, LR at floor),
  *not* an exploding undefined schedule. *Speculation:* the shallow curriculum
  never trains the model at horizons >4, so it never learns to keep the
  MB-derived `h` bounded under accumulated error; at eval the free rollout then
  diverges. The full curriculum (up to 10 steps) does learn this. **Caveat:** the
  two arms are still not like-for-like — the short arm compresses all learning
  into 50 epochs then idles 200 at `lr_final`, confounding "shallower horizon"
  with "different LR budget."
- **Fixed-horizon instability — resolved *within this run*: a
  conditionally-unstable rollout operator, not a broken metric or a regression
  (evidence).** The apparent contradiction ("fixed-horizon blows up but the
  rollout is stable") dissolves once horizon length is ruled out: the *blown-up*
  rollout is the **shorter** one (30 steps) while the stable date-range
  trajectory is **222** steps (hps4: bounded, pred q ∈ [4.0, 110] vs truth
  [3.7, 122], ends 40.8 vs 26.8). Longer-horizon error accumulation therefore
  *cannot* be the cause. The only difference left is the **start state**: the
  date-range rollout is a single benign winter-low-flow draw (Jan 2), whereas
  the fixed-horizon metric launches **32 anchors** spread across the val split —
  including stiff spring/summer rising limbs — and aggregates by RMSE / max-ratio,
  so a single diverging anchor dominates. The metric is effectively a *"does any
  start state diverge?"* detector.
  - **Mechanism (evidence):** the q→h amplification is logged at `val_amp ≈
    11–14` every epoch ([amplification.csv](../experiments/sava_small_v081_s2_stability/sava_small_v081_s2_stability_hps4/metrics/amplification.csv)),
    vs analytic `mb_gain ≈ 21.9`. A per-step gain ~12 gives geometric growth
    `peak_ratio ≈ amp^H`: `12^30 ≈ 1e32`, matching the observed 1e21–1e29
    (Float32-capped to `Inf`). This is **multiplicative**, the signature of an
    unstable rollout *operator*, not additive error drift.
  - **State-dependence & the fragile channel (evidence):** `val_amp > 1` is
    benign in low-flow states (winter trajectory + the ≤10-step curriculum
    rollout stay bounded) but geometrically unstable from high-flow starts. It is
    the **h channel** that is chronically ill-behaved: even in the *stable*
    date-range run `river_h_pred` flickers 0 ↔ spurious (0.0, 0.92, 0.0, 0.13)
    and spatial `river_h` NSE is negative everywhere (e.g. hps2 −8.98). Whether h
    drags q to infinity depends entirely on the start state.
  - **Consequence for prior claims (evidence + inference):** this
    *retro-weakens* the archived `mb_sweep` "stable over 1906 steps" result —
    that too was a single benign trajectory and never demonstrated a globally
    stable operator; the same latent instability was almost certainly present but
    unsampled. It also **closes the earlier metric-stringency-vs-regression
    question without needing an `mb_sweep` re-run**: it is initial-condition
    coverage, confirmed within this run. The metric is not broken — it exposes
    what the old single-trajectory evaluation could not see.

**RECOMMENDATIONS.**
1. **Fix the grad_clip search bug first** (engineering): make
   `src/hparsearch.jl` thread `grad_clip`, `lr_warmup_epochs`, `lr_peak_decay`
   into `TrainSettings` (or have it reuse the same parser as `run.jl` to avoid
   drift). Until then, *no* hparsearch can vary those three fields — a latent
   trap for all future searches.
2. **Add a per-anchor fixed-horizon diagnostic** — highest priority; converts
   the instability story from strong inference to direct evidence. Log the
   fixed-horizon RMSE *per anchor* (not just the aggregate) alongside the flow
   percentile of each anchor's start state, for one full-curriculum run.
   Prediction: divergence concentrates in high-flow-start anchors while
   low-flow anchors stay bounded. This pinpoints the regime where the MB rollout
   operator loses stability — the exact target for the S3 stability / peak-loss
   work. (An `mb_sweep` re-run is *no longer needed* to disambiguate — the cause
   is established as initial-condition coverage, see HYPOTHESES.)
3. **Treat the conditional rollout instability as the primary open problem.**
   The q→h amplification (~12×/step, `val_amp`) is stable-in-mean but
   geometrically unstable from stiff high-flow states; damping it is what the
   `mb_theta` lever and the peak-weighted-loss workstream should target. Report
   `val_amp` and a high-flow-anchor rollout metric as first-class signals going
   forward, not the single benign date-range trajectory (which is
   under-powered and gave false confidence in `mb_sweep`).
4. **Re-run the intended grad_clip A/B** once bug 1 is fixed, on the full
   curriculum only (the short curriculum is a dead arm), `{1.0, 0.5, 0.1}` — to
   answer the original de-risking question for the full-basin remedy.
5. **Drop the short `[1,2,4]` curriculum** from further small-basin work; if
   curriculum depth is to be studied, hold total-epoch and LR budget constant
   across arms (e.g. `[1,2,4]` with `durations = [80,80,90]`) so it is a
   like-for-like comparison.
6. **Confirm `increment` as default** — this batch independently reproduces its
   ~2.5× rollout advantage; fold into the S3 peak-weighted-Huber baseline when
   that engineering lands.

---

# OLD STATUS AS OF 27-08-2026

*Everything below this header predates the introduction of these structured
status logs; it back-fills experiments run before 27-08-2026.*

---

## Mass-balance on/off A/B sweep — `sava_small_v081_mb_sweep`

Identical setup (8 layers, hidden 64, mlp 2, batch 8, 250 epochs, curriculum
`steps = [1,2,5,8,10]`, `lr_start ≈ 1.35e-4`, `grad_clip = 1.0`), varying **only**
`enforce_mass_balance`.

| Metric | MB on | MB off |
|---|---|---|
| final train / val loss | 0.0285 / 0.0242 | 0.0441 / 0.0438 |
| date-range rollout Q RMSE | **27.5** | ≈ 7.5×10⁵ |
| date-range rollout H RMSE | **6.55** | ≈ 1.1×10⁴ |
| full val rollout (1906 steps) | finite | Float32 overflow → Inf from step ≈1571 |

**Diagnosis:** teacher-forced training only ~1.8× worse without MB, but the
free-running rollout diverges exponentially. Without an algebraic anchor a small
consistent bias compounds multiplicatively. **The hard MB stabilises at
inference time.** Basis for the hard-constraint decision.

## θ-method sweep — `sava_small_v081_theta_sweep`

Explores the implicit/explicit/trapezoidal (θ) discretisation; `θ* = σ_h/(dt·σ_q)`
cancels the q↔h gradient stiffness. Default shipped: `mb_theta = 1.0` (fully
implicit). See [notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md).

## h_loss_scale variants — `sava_small_v081_increment`, `*_mb_prior_increment`, `*_mb_no_prior_absolute`, `*_mb_exp_prior`

Compares how the h loss term is scaled/weighted (increment vs absolute, with/without
Q rating prior). Informed the `h_loss_weight = σ_h/(dt·σ_q)` choice.

## Hyperparameter search — `sava_small_v081_hpar_search_hps09`, `sava_v081_autotune`

LR-range test / autotune and hidden-dim / layer-count / mlp-depth search on the
small basin.

## Full-basin scale-up — `sava_full_v081` (slurm-246431)

Large run: 8 SparseConv layers, hidden 64, mlp 2, `enforce_mass_balance=true`,
`mb_theta=1.0`, 75,137 params; graph 8235 nodes / 11322 timesteps; 7918 train /
2262 val / 1132 test windows.

**Outcome — training divergence.** Single-step loss blew up around epoch 2
(~170 LR-backoff events fired). **Root cause:** heavy-tailed normalised targets
(extreme cells at hundreds of σ) amplified by the squared loss; larger basins
have heavier tails. *Not* the σ-ratio (identical across models), *not* LR step
size (peak LR only ~2e-7), *not* N-scaling (loss is a mean over nodes).
**Remedies** → variance-stabilising input transform + peak-weighted Huber loss;
tracked in [TODO.md](TODO.md) and [notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md).

## Smoke / regression runs — `test_run`, `test_run_full`, `test_sava_small(_v081)`, `test_sava_v081`

Small fixtures used for quick end-to-end validation of preprocessing, training,
rollout and postprocessing.