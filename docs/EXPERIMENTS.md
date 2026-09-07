This document serves as an archive of experiments, listing evaluation metrics and some basic diagnoses and hypotheses of each experiment.

Experiment configs live under `experiments/`. Newest at the top.

---

# PROPOSED — not yet run

*Post-S2 peak-accuracy / stability plan, unblocked by the 2026-09-04 code
changes (shared config parser + verified `grad_clip` propagation; peak-weighted
Huber; Tier-1 peak diagnostics; Tier-2 `river_q` KGE/PBIAS/FHV; per-anchor
fixed-horizon diagnostics). E1 and E2 are **done** (see COMPLETED); E3 depends on
E1. Full rationale in [TODO.md](TODO.md) and
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md).*

> **⚠ Cross-cutting finding from E1 + E2 (read before running E3).** No
> configuration tested so far produces a stable 30-step free rollout: the
> fixed-horizon metric diverges from **every** anchor regardless of start-flow
> percentile (E1) and regardless of `grad_clip` (E2). The instability is a
> property of the learned MB **rollout operator**, not a training-gradient or
> initial-condition effect.
>
> **Update 2026-09-07 — rollout-path discrepancy RESOLVED (not a code bug).**
> Engineering added a same-start regression ([TODO.md](TODO.md)): the batched
> fixed-horizon anchor path and the single date-range trajectory path **agree to
> tolerance when seeded from the same initial state**, and both run from the
> restored best-epoch checkpoint. So the fixed-horizon numbers are a *faithful*
> measure of model behaviour — the earlier "anchor 1 ≈ date-range start"
> equivalence (E1) was the wrong premise: the bounded date-range and the
> divergent anchors are simply *different* start states/forcings, not a code
> mismatch. **Consequence for E3:** fixed-horizon metrics are trustworthy, but
> because the free rollout is unconditionally unstable across every tested
> config, they will likely be Inf/astronomical for all λ cells and so cannot
> *discriminate* between them — still select E3 on teacher-forced peak
> diagnostics (`c_peak`, `rmse_high`, KGE).

## E3 — peak-weighted Huber λ-sweep — `sava_small_v081_e3_huber_lambda` (PROPOSED)

**Status:** config staged at
[experiments/sava_small_v081_e3_huber_lambda/config.toml](../experiments/sava_small_v081_e3_huber_lambda/config.toml);
not yet run. **Depends on E1** (for `peak_delta` + `h_loss_weight`). `box`
hparsearch, 6 runs, `train.strategy.peak_lambda ∈ {0, 0.5, 1, 2, 4, 8}`
(λ=0 = unweighted-Huber baseline). Per-node `u_i` (98th pct) / `s_i` (IQR)
computed automatically (train split) when `loss_type = "huber"`.

**Purpose.** The core peak-accuracy intervention
([notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md) §2/§2a): does
per-node peak-weighting improve peak-region skill without regressing the whole
hydrograph, on the stable small basin, before any full-basin spend.

**Autotune folded in (Step 0, once).** Loss changes MSE → Huber ⇒ gradient scale
shifts, so re-tune. Peak weighting is Σw-normalised (≈λ-invariant) and Huber
bounds residual gradients, so a **single** tune on the unweighted-Huber base
config transfers across the λ grid:
`autotune_train.jl <config> --dry-run --range-horizon 10` (deep-horizon peak,
since S2 located the instability at the deep-horizon / high-flow end), then run
the sweep. **`peak_delta`: E1 harvested `≈0.5`** (heavy-tailed standardised
`river_q` residuals: median 0.098, mean 1.16, p90 2.0) — update the config's
placeholder `1.0` to `0.5` before launching; sanity-check against `{0.5,1,2}`.

**Selection (never weighted val loss; fixed-horizon metrics are now trusted as a
faithful instability signal — resolved 2026-09-07 — but will diverge for every λ
cell and so can't discriminate between them).** Target `peak_loss.final_c_peak ≈
0.3–0.5`; minimise `peak_loss.final_rmse_high`; constraint
`river_q_performance.pooled.kge` must not degrade materially vs λ=0 (**and treat
KGE gaps < ~0.15 as seed noise, per E2**); guard
`peak_loss.final_peak_grad_frac` (no tiny cell fraction dominating the
gradient → sets `peak_w_max`).

**Results:** _pending._

---

# COMPLETED

## E1 — diagnostic baseline — `sava_small_v081_e1_diag`

**NAME:** `sava_small_v081_e1_diag` (single MSE run; S2 winner: increment
h-loss, full `[1,2,5,8,10]` curriculum, `grad_clip = 1.0`, 8×64 / mlp 2, 75,009
params, 250 epochs; LR reused from autotuned `mb_sweep`). Config:
[experiments/sava_small_v081_e1_diag/config.toml](../experiments/sava_small_v081_e1_diag/config.toml).

**SUMMARY.** Purpose was to harvest the new diagnostics and test the S2
conditional-instability hypothesis. Teacher-forced discharge fit is strong
(`val_q_1step` 0.0144, spatial `river_q` NSE 0.973). But the 30-step free
rollout diverges from **all 32 anchors** (`fixed_rmse = Inf`), and Tier-2
selection metrics reveal only mediocre skill even on the bounded path
(pooled KGE 0.577, outlet KGE **−0.043**, outlet PBIAS +46%, date-range peak
overshoot 1.9×). `river_h` remains broken (spatial NSE −9.9).

**IMPROVEMENTS.**
- **New Tier-2 `river_q` metrics now available** — pooled vs outlet-gauge
  KGE/r/α/β/MAE/PBIAS/peak_error/FHV, plus per-anchor fixed-horizon CSV and
  `peak_ratio_frac_gt2` / `rmse_highflow`. First run to expose gauge-level skill.
- **`peak_delta ≈ 0.5` harvested for E3** (evidence). Per-cell standardised
  `river_q` RMSE is heavy-tailed: median 0.098, mean 1.16, p90 2.0, std 3.56.
  A Huber knee ~0.5 keeps the bulk quadratic while linearising the extreme-cell
  tail. **Confirms the heavy-tail hypothesis on the small basin** (previously
  only asserted for the full basin).

**DEGRADED / FAILURE MODES.**
- **S2 conditional-instability hypothesis REFUTED (evidence).** The per-anchor
  CSV shows **all 32 anchors → Inf** with *no* dependence on start-flow
  percentile: the 2.2-percentile low-flow anchor (idx 19) blows up exactly like
  the 98.5-percentile high-flow anchor (idx 12). Divergence is **unconditional
  across initial states**, not high-flow-triggered as I inferred from the single
  S2 trajectory. Only 5 / 250 epochs produced a finite (<1e6) fixed-horizon RMSE.
- **Poor outlet skill (evidence).** Outlet-gauge KGE −0.043 (worse than the mean
  baseline), PBIAS +46%, FHV +44% — systematic over-prediction concentrated at
  the outlet even where the rollout stays bounded.

**HYPOTHESES.**
- **~~Two rollout paths disagree from the same start~~ — RESOLVED 2026-09-07
  (not a code discrepancy).** I originally hypothesised a code mismatch because
  the 222-step date-range rollout stays bounded (pred q ∈ [3.9, 228.6]) while
  the 30-step anchor rollout → Inf, assuming anchor 1 ≈ the date-range start.
  Engineering's same-start regression ([TODO.md](TODO.md)) shows the two paths
  **agree to tolerance from an identical seed state** and both use the restored
  best-epoch checkpoint. **The inferred "anchor 1 ≈ date-range start" premise was
  wrong** — the bounded trajectory and the divergent anchors are genuinely
  different initial states/forcings. Net: the fixed-horizon divergence is a real
  model-instability signal, not a measurement artifact, and it is now safe to
  trust `fixed_horizon.*` as a faithful (if uniformly damning) instability metric.
- **`h_loss_weight` not directly measurable here (evidence).** The MSE run does
  not emit the huber-only per-channel grad-norm diagnostic; proxy from loss
  magnitudes `val_q_1step` 0.0144 vs `val_h_1step` 0.848 (~59×). The Step-0
  unweighted-Huber autotune run will give the real per-channel grad norms.

**RECOMMENDATIONS.**
1. **~~Reconcile the two rollout paths~~ — DONE 2026-09-07.** Engineering
   confirmed the paths agree from the same seed state and use the restored
   best-epoch checkpoint; the fixed-horizon divergence is genuine model
   instability, not a code mismatch ([TODO.md](TODO.md)). Fixed-horizon metrics
   are now trusted as an instability signal.
2. **Prioritise an inference-stability lever over peak accuracy (now the top
   open item)** — no config yet yields a stable free rollout; `mb_theta < 1`
   (damps the confirmed `val_amp ≈ 11.5` q→h amplification), noise injection, or
   the detached-rollout trick (TODO item 3).
3. **E3 still worth running** for the peak signal, but expect its fixed-horizon
   numbers to diverge for every λ cell (so they can't discriminate); select on
   teacher-forced peak diagnostics and update its `peak_delta` from `1.0` to
   `≈0.5`.

## E2 — grad_clip A/B — `sava_small_v081_e2_gradclip`

**NAME:** `sava_small_v081_e2_gradclip` (3-run `box` hparsearch,
`train.grad_clip ∈ {1.0, 0.5, 0.1}` = hps1/hps2/hps3; else E1/S2-winner config,
MSE, full curriculum). Config:
[experiments/sava_small_v081_e2_gradclip/config.toml](../experiments/sava_small_v081_e2_gradclip/config.toml).
Run before the E1-derived revised recommendations could be applied.

**⚠ Measurement note.** An earlier "hps3 = 32/32 finite anchors" reading was a
**false positive** from filtering anchors by the literal string `Inf`: hps3's
anchor RMSEs are finite-but-astronomical (1e10–1e12), i.e. diverged *below* the
Float32 overflow threshold, not stable. **All three cells diverge.** Filter
fixed-horizon anchors by *magnitude* (e.g. `< 10×` truth peak), never by `Inf`.

**SUMMARY.**

| clip | pooled KGE | outlet KGE | outlet PBIAS | pooled peak_ratio | final peak_ratio | anchor RMSE scale | best_val_rmse | n_backoffs |
|---|---|---|---|---|---|---|---|---|
| 1.0 (hps1) | **0.736** | **0.406** | 24.9% | 1.65 | 1.5e19 | Inf (overflow) | 3.80e6 | 0 |
| 0.5 (hps2) | 0.701 | 0.228 | 31.7% | 1.55 | 4.4e27 | Inf (overflow) | 4.73e5 | 0 |
| 0.1 (hps3) | 0.684 | 0.088 | 23.0% | 3.10 | 4.4e11 | 1e11 (finite, diverged) | 1769 | 0 |

**IMPROVEMENTS.** None actionable — no clip value fixes the divergence, and the
default (1.0) is already the best on skill.

**DEGRADED / FAILURE MODES.**
- **grad_clip does NOT fix the free-rollout divergence (evidence).** All three
  cells diverge 11–27 orders of magnitude over 30 steps. Tighter clipping lowers
  the blow-up *magnitude* (1e19 → 1e11) but never removes it. Confirms the
  divergence is an **inference-time rollout-operator** property, not a
  training-gradient effect a clip can cure.
- **Tighter clipping actively hurts skill (evidence).** Monotonic pooled-KGE
  degradation 0.736 → 0.701 → 0.684 as clip tightens; outlet KGE collapses
  0.406 → 0.088 (approaching no-skill). **Keep `grad_clip = 1.0`.**

**HYPOTHESES.**
- **Lever-resistant structural instability (evidence, accumulating).** Curriculum
  (S2) and now grad_clip (E2) both fail to stabilise the free rollout — it is
  intrinsic to the learned MB rollout operator, reinforcing the inference-stability
  priority.
- **Large seed variance (evidence).** E2-hps1 and E1 are the *identical* config
  yet differ pooled KGE 0.736 vs 0.577 (best_epoch 56 vs 37, peak_ratio 1.65 vs
  2.29) — a ~0.15 KGE spread from RNG alone. Echoes the S2 replicate spread.
  **Implication:** treat single-run KGE differences below ~0.15 as noise; compare
  E3 λ cells with this band in mind (consider replicates). *Caveat:* assumes only
  RNG differs between E1 and E2-hps1 (no seed pinning / config drift) — worth
  confirming.

**RECOMMENDATIONS.**
1. **grad_clip is settled — keep 1.0**; no further grad_clip experiments.
2. **Escalate the inference-stability workstream** (mb_theta / noise / detached
   rollout) above peak accuracy.
3. **Carry the seed-variance band into E3 selection** — don't over-read small
   KGE gaps.

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