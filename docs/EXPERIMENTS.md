This document serves as an archive of experiments, listing evaluation metrics and some basic diagnoses and hypotheses of each experiment.

Experiment configs live under `experiments/`. Newest at the top.

---

# PROPOSED

> **E11 — distribution/physics-aware feature scaling (TODO §1) on the best-known
> base, 5-seed replication. CONFIGURED, not yet run.** The first item off the
> "still owed" list below: the new normalization (log1p-scaled skewed statics
> `slope`/`width`/`depth`/`length`, area-normalised `river_inwater`, train-split-only
> stats — all now default — plus the optional `log1p(upstream_area)` static context
> feature, `include_log_upstream_area = true`) applied to the established pure-
> pushforward base and run **5× over seeds 1–5** so the effect reads as mean ± std,
> not a single seed-dominated point. Only intended deltas vs the E9/E10 hps1 base:
> the new normalization, plain MSE (drops the E10 `huber + peak_delta=1000`
> workaround since peak-weighting is closed), and the 5-seed replication. Everything
> else held at the working base (`rollout_grad = pushforward`, `tf = 0`, `noise = 0`,
> `peak_lambda = 0`, `mb_theta = 1`, `lr = 1.3457e-4`, `h_loss_scale = increment`,
> `[1,2,5,8,10]×50 = 250` ep, 8×64 / mlp 2, q≥0 floor). **Key read-out:** does
> `spatial_median.river_h.nse` (the binding failure, stuck ~−120 across E4→E10 and
> hypothesised to be a scaling problem) move, and do the downstream/outlet rollout
> tail (per-node RMSE, NSE p10, outlet peak ratio) tighten — judged on the deployed
> free-rollout timeseries (E9 lesson), read as mean ± std. `amp` expected ~unchanged
> (q/h stay linear z-score). Config:
> [experiments/sava_small_v081_e11_normalization/config.toml](../experiments/sava_small_v081_e11_normalization/config.toml).
> Cost ~2 h/seed on the pushforward base ⇒ ~10 h for the 5 seeds (sequential in one
> job; the box has a single combo).
>
> **The planned post-E6 pushforward follow-up series (E7–E10) is COMPLETE.** No
> single-knob sweep is queued. What the series settled: the boundedness win was the
> always-on **q≥0 floor** (E7; noise was a weak/chaotic lever); deepening the
> curriculum on pure pushforward was a **net negative** (E8, but confounded with
> epoch budget); the hybrid loss recovered only the **teacher-forced** 1-step metric
> and did NOT help the free rollout at acceptable cost (E9, tf=0.25 NOT adopted); and
> per-node **peak-weighting is a closed lever** — even when the weight genuinely bit
> (E10, λ=2 `w_mean` 1.37), it gave no reliable peak or rollout benefit and mildly
> **worsened** the deployed rollout. **Working base stays pure pushforward
> (`rollout_grad = pushforward`, `tf_weight = 0`, MSE, q≥0 floor, `mb_theta = 1`).**
>
> **Still OWED (not yet configured) — the real open work:**
> 1. **Distribution/scaling fixes (TODO §1)** for downstream conditioning — the
>    prime suspect for the persistent `river_h` failure (spatial NSE stuck ~−120)
>    and the outlet-dominated rollout error. **→ now being tested as E11 (5-seed,
>    configured above).**
> 2. **Clean budget-controlled schedule study** (E8 follow-up): isolate fragmentation
>    (horizon 10 in many short phases) vs depth (add deep phases holding ~50 epochs
>    at steps=1), maybe a longer total budget.
> 3. **Multi-seed confirmation** (≥3 seeds) of the levers currently within the seed
>    band before relying on any of them — fixed-horizon RMSE for the pure-pf base is
>    now replicated 4× at {9.3, 30.7, 95, 1558}, i.e. hopelessly seed-dominated.
>    **→ E11 establishes the 5-seed pure-pf-base reference distribution.**
> 4. **Engineering:** gate `peak_stats` on `loss_type == :huber || peak_lambda > 0`
>    (drops the E10 `peak_delta = 1000` workaround) if peak-weighting is ever revisited.

---

# COMPLETED

> **⚠ Cross-cutting finding from E1 + E2 + E3 + E4 (read before any
> peak-accuracy run).** No configuration tested so far produces a *faithful*
> stable 30-step free rollout. The rollout **attractor** depends on the LR, and
> the **loss function (MSE vs Huber) is not the lever**:
> - **E1/E2 (MSE, lr ≈ 1.3e-4):** rollout **over-predicts and diverges to +Inf**
>   from every anchor, regardless of start-flow percentile (E1) and `grad_clip`
>   (E2).
> - **E4 (Huber δ0.5 λ2, lr = 1.3e-4):** stiff fixed-horizon anchors **diverge
>   upward like E1/E2 MSE** (`frac_gt2 = 1.0` from **epoch 2**, peak_ratio ~1e13),
>   **but the benign daterange is far worse than MSE** — outlet 286× (35,012 vs
>   122) vs MSE's 1.9×. So holding LR fixed, MSE→Huber keeps the stiff-anchor
>   failure but **additionally destabilises the benign rollout** (δ-tail effect).
> - **E3 (Huber δ0.5, autotuned lr = 0.012, ~90×):** rollout **under-predicts and
>   collapses to ~0** (even negative flows) — "bounded" anchors, but degenerate.
>
> **Disentangling conclusion (E4).** E3's collapse-to-zero was **LR-driven** (the
> ~90× jump), **not Huber-driven**: at the stable LR, Huber's *stiff fixed-horizon
> anchors* diverge upward like MSE. **But the loss is not irrelevant** — on the
> benign daterange trajectory, Huber (δ0.5) is ~150× worse than MSE at the same LR
> (outlet 286× vs 1.9×), most likely from Huber's **δ linear tail** under-
> penalising large errors (NOT the peak-weight λ, which is per-node-relative and
> nearly dormant).
>
> **δ-sweep conclusion (E5) — loss-retuning workstream CLOSED.** Sweeping the
> Huber knee `δ ∈ {0.25…5}` confirms **both** halves: the benign-path
> over-prediction shrinks **monotonically** as δ→5 and asymptotes to MSE (δ-tail
> mechanism real), **but** the rollout instability is untouched — all 32
> fixed-horizon anchors diverge at every δ, and the amplification is **δ-invariant**
> (`amp ≈ 11.2–12.0`, `mb_gain = 21.9` in all cells, = E1). The loss moves *where*
> the benign trajectory sits, not *whether* the rollout is stable. **The loss knob
> is exhausted; use MSE and attack the rollout operator directly.** So: **LR sets
> the attractor (+∞ ↔ 0); loss shapes the benign-path magnitude only; prefer
> MSE.** Neither loss yields a faithful rollout. **Always cross-check a "bounded"
> fixed-horizon RMSE against the daterange pred-vs-true magnitude and pooled
> PBIAS/peak_ratio** — both a collapse-to-zero and an explosive over-prediction
> can look "finished" (and divergent magnitudes are chaotic: E4 & E5 same config
> gave outlet 286× vs 23,557× — trust trends, not single ratios). The rollout-path
> discrepancy is *resolved* (2026-09-07, not a code bug): the fixed-horizon metric
> is faithful; it's the model that's unstable. The standing priority is a
> **training-time error-correction mechanism** (rollout-gradient mode —
> detached/pushforward, **E6**; **noise injection**; longer curriculum) — **NOT**
> `mb_theta < 1` (already swept: θ=1 is the most stable, lower θ diverges) and
> **not** loss retuning (exhausted by E5).
>
> **E6 — the mechanism IS attackable (the positive result).** The rollout-gradient
> mode is the first lever to move the learnable q→h amplification: `pushforward`
> lowers `amp` 11.03→**6.37** (invariant to loss/δ/θ before this), is the **only**
> mode with a **finite** fixed-horizon RMSE (61,092 vs full_bptt 9.4e18), and wins
> every benign-rollout metric (pooled KGE 0.876, per-node NSE p10 0.85) at 2.4×
> lower cost. Confirms the instability is **autoregressive exposure bias**, fixable
> in *what state the model trains on*, not the objective. Caveats: it only **damps**
> (frac_gt2 still 1.0), and it **regresses the 1-step fit + `river_h`** (endpoint-
> only supervision). `detached` is a net negative; drop it. Next: fix the
> pushforward h/1-step regression (hybrid loss) and stack **pushforward × noise ×
> q≥0 floor**.
>
> **E7 — the q≥0 floor is the real boundedness win; noise is a weak, chaotic
> lever (partial negative).** Two surprises from the noise sweep on the
> pushforward base. **(1) The floor, not noise, did the heavy lifting.** `hps1`
> (noise=0) is the E6-pushforward config unchanged *except* for the now-always-on
> `_floor_q_norm_nonnegative` (src/gnn.jl); it moved `best_val_rmse` **1196 → 5.60**
> and final fixed RMSE **61,092 → 95** (finite) — ~200–600×, well beyond E5's ~80×
> chaotic band, so the floor is the driver (seed-confounded but too large for seed
> alone). **(2) The hypothesised mechanism — noise lowers `amp` — is REJECTED:**
> `amp` is flat at 6.8–7.5 across all five cells, uncorrelated with noise. Noise
> *did* pull `frac_gt2` off 1.0, but only at 0.01 (0.844) and 0.1 (0.875), while
> 0.03 **diverged** (final RMSE 1e13) and 0.3 stayed at 1.0 — a non-monotone
> zig-zag = weak signal under seed chaos, not a dose-response. Noise mildly helps
> benign skill (KGE 0.782→0.908, pbias 11.7→4.7 with dose) and the 1-step/h fit is
> best at 0.1 (`val_q_1step` 0.109, `val_h_1step` 6.24), but `river_h` NSE stays
> ~−93 — **noise does NOT fix the pushforward h/1-step regression** (that is E9's
> job, as predicted). Standing read: adopt a **mild** noise (~0.05–0.1) as a cheap
> regulariser, credit the **q≥0 floor** as the boundedness win to carry forward,
> and treat `frac_gt2` gains as fragile until confirmed across seeds. Priority
> stays **E9 (hybrid loss)** for the unresolved h/1-step regression.
>
> **E8 — deepening the curriculum on *pure* pushforward is a NET NEGATIVE.**
> Pushing the deepest curriculum phase from horizon 10 → 20 → 30 (to close the
> train/eval-horizon gap) **backfired**: `frac_gt2` got *worse* with depth
> (0.75 → 0.875 → 1.0), final fixed RMSE degraded (9.3 → 15.7 → **3.6e6**), and the
> cell trained *all the way to* horizon 30 still diverges at 30 — so the
> hypothesis (train to eval horizon ⇒ close the gap) is **refuted**. Worse, depth
> **destabilises pushforward training**: the horizon-30 cell logged **137
> non-finite skips / 66 back-offs / max grad 1.2e16**, and the final models diverge
> (val_rollout 416 @20, **3.9e13** @30). And it **starves the 1-step/`river_h`
> map** even harder (val_h_1step 8.6 → 122 → 161). **⚠ CONFOUND — this sweep does
> NOT cleanly isolate depth.** With a fixed 250-epoch budget, deeper = more phases
> = fewer epochs each; the `steps=1` teacher-forced budget was **halved** (50 → 25)
> to make room, and **all three best checkpoints land in the `steps=1` phase**
> (epochs 18/10/22 — before any deep rollout training). So (a) the 1-step/`river_h`
> starvation is at least as much a **shallow-budget** effect as a depth effect, and
> (b) the apparent "skill improves with depth" is **misattributed** — those
> checkpoints never saw deep training, so it is shallow-phase/seed variance, not a
> depth benefit. What IS cleanly depth-driven: the horizon-30 **gradient
> explosions** (137 skips / 66 back-offs / max grad 1.2e16) and diverging final
> models. **Conclusion: on pure pushforward, deepening as-designed is a net
> negative — but a clean depth test (holding the shallow budget fixed) is still
> owed; run E9 first, then a proper schedule study.** (Seed note: this control got
> `frac_gt2 = 0.75` vs E7's identical-config 1.0 — reconfirms `frac_gt2` is
> seed-chaotic.) Priority stays **E9**.
>
> **E9 — the hybrid loss fixes the *teacher-forced* 1-step metric, but that does
> NOT carry to the free rollout (the deflating keystone result).** Sweeping
> `pushforward_tf_weight = (1−α) ∈ {0,.25,.5,.75}` (adds a teacher-forced 1-step
> term to the endpoint-only pushforward loss) recovers the **teacher-forced**
> `val_q_1step` (0.152→0.017 at tf=0.25, ≈ full_bptt) and `val_h_1step` (9.34→1.55)
> monotonically and strongly. **But those are one-step, ground-truth-fed errors —
> not deployed skill.** In the actual free-rollout validation timeseries (the thing
> that gets plotted) the tf=0.25 gain nearly evaporates: outlet `river_q` RMSE
> 13.30→12.27 (−8%, plausibly within the seed band), MAE −27% and peak ratio
> 0.71→0.86 are the only clear wins and they are **outlet-only**; upstream gauges
> are a wash or slightly worse (node152 +39%), and rollout `river_h` is **mixed to
> worse** (outlet RMSE 3.91→6.40, +64%). So the E6 1-step/`river_h` regression is
> **not** meaningfully fixed where it matters, and the 3.3× wall-clock cost buys
> little. The genuine, seed-robust findings survive: (1) the low-amp pushforward
> solution is **lost the instant any tf is added** — `amp` jumps 7.7→**13.5**;
> (2) `tf ≥ 0.5` is **catastrophic** — over-prediction (outlet 0.71→1.26, pbias
> 6.8→20.4) → fixed-horizon RMSE **1e6→1e31**; (3) `amp` **decouples from rollout
> stability** under hybrid training, undercutting E6's "amp is the mechanism" story.
> **Revised verdict: tf=0.25 is NOT worth adopting on this single-seed evidence** —
> the improvement is a teacher-forced-metric artifact plus a modest outlet-only q
> gain at 3.3× cost and worse outlet h. **Pure pushforward stays the base.** If
> pursued, tf=0.25 needs a multi-seed run proving the outlet peak/MAE gain is real
> and not seed noise. Lesson: judge this loss on **free-rollout** timeseries, never
> on `val_*_1step`.
>
> **E10 — peak-weighting is a CLOSED lever (the series-ending null).** Sweeping the
> per-node `peak_lambda ∈ {0,1,2,4}` on the pushforward base finally made the weight
> **bite** (λ=2: `w_mean` 1.37, `c_peak` 0.996 — vs E5's inert 1.0005), so this is a
> real "weighting does not help" result, not a dormant-lever artifact. `amp` is dead
> flat (7.0–7.3, rollout-neutral) and pooled KGE flat (0.82–0.85), but on the
> **deployed free-rollout** timeseries every weighted cell is *worse* at the two
> largest nodes (outlet q RMSE +24–37%, node74 +60–80%) because the weight drives
> **over-shoot** (node74 peak ratio 0.87→1.4). The pooled `peak_ratio` drifting to
> 1.0 is over-shoot cancelling under-shoot, not skill; the only clean positive (mild
> FHV) is offset by the RMSE cost. `river_h` and 1-step are untouched. **Verdict:
> stop tuning the loss (E4→E10 all confirm loss shape is orthogonal to the rollout
> operator); the binding failures — `river_h` (NSE ~−120 everywhere) and outlet-
> dominated rollout error — need the distribution/scaling work (TODO §1), which is
> now the priority.** (Process note: hps1 λ=0 is a 4th pure-pf replicate; fixed-
> horizon final RMSE now spans {9.3, 30.7, 95, 1559} — hopelessly seed-dominated.)

## E10 — per-node peak-weight (`peak_lambda`) sweep — `sava_small_v081_e10_peakweight_sweep`

**NAME:** box search, 4 cells, one knob
`train.strategy.peak_lambda ∈ {0.0, 1.0, 2.0, 4.0}` (`peak_gamma = 1`,
`peak_w_max = 4` held), on the pure-pushforward base (`rollout_grad = pushforward`,
`tf_weight = 0`, `lr = 1.3457e-4`, `mb_theta = 1.0`, full `[1,2,5,8,10]` curriculum,
8×64 / mlp 2, 250 epochs, always-on q≥0 floor). **⚠ Uses `loss_type = "huber"` with
`peak_delta = 1000`** — numerically identical to MSE across the O(1) normalized
range, but needed because the per-node peak stats (`uᵢ`/`sᵢ`) are only computed
under the `== :huber` gate ([src/run.jl](../src/run.jl) ~L674); `hps1` (λ=0) is
therefore an MSE-equivalent pure-pushforward control. Per-node weight
`w_i = min(1 + λ·peak_scoreᵢ, w_max)`, `peak_scoreᵢ = max(0, (target − uᵢ)/sᵢ)`.
Config:
[experiments/sava_small_v081_e10_peakweight_sweep/config.toml](../experiments/sava_small_v081_e10_peakweight_sweep/config.toml).

**SUMMARY — peak-weighting is a CLOSED lever; this test is stronger than E5 because
the weight actually bit and still gave no reliable benefit (mild net harm in
rollout).** Unlike E5 (inert, `w_mean ≈ 1.0005`), at λ=2 the weight genuinely
activated (`w_mean` 1.37, `c_peak` 0.996). Yet across the sweep: `amp` is **dead
flat** (7.02–7.32, rollout-neutral as hoped), pooled KGE is **flat within seed
noise** (0.82–0.85), and 1-step / `river_h` are **untouched** (`val_q_1step`
0.18–0.20; spatial `river_h` NSE −118 → −154, no help). Re-scored on the **deployed
free-rollout** timeseries (the E9 lesson), every weighted cell is **worse** in
`river_q` RMSE at the two largest nodes (outlet +24–37%, node74 +60–80%) because
the weight drives **over-shoot** (node74 peak ratio 0.87 → 1.16–1.41). The only
consistent positive is a mild pooled-FHV improvement (0.111 → 0.088/0.101/0.098),
offset by the RMSE degradation. The weight-activation is also **non-monotonic**
(λ=2 bites harder than λ=4 — `w_mean` 1.00017 / 1.37 / 1.05), i.e. seed/trajectory-
driven, not a clean dose. **Conclusion: close the peak-weight lever; the persistent
failures (`river_h`, outlet-dominated rollout error) need the distribution/scaling
work, not loss re-weighting.**

**The four cells (best-epoch; outlet = node1_r1, true peak 122.4):**

| cell | λ | `w_mean` / `c_peak` | **final_amp** | pooled KGE | pooled peak_ratio | pooled FHV | pbias | val_q_1step | river_h NSE | final fixed RMSE | frac_gt2 | best_ep | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hps1 (ctrl) | 0.0 | 1.000 / 0.00 | 7.22 | 0.836 | 0.963 | 0.111 | 8.5 | 0.183 | −118 | 1559 | 1.0 | 26 | 2.22 h |
| hps2 | 1.0 | 1.00017 / 0.02 | 7.20 | **0.852** | 1.197 | **0.088** | 7.1 | **0.140** | −122 | 1.99e5 | 1.0 | 11 | 2.25 h |
| hps3 | 2.0 | **1.367 / 0.996** | 7.32 | 0.843 | **1.000** | 0.101 | 8.8 | 0.191 | −153 | **8.87** | **0.656** | 28 | 2.28 h |
| hps4 | 4.0 | 1.050 / 0.651 | 7.02 | 0.824 | 0.983 | 0.098 | 7.7 | 0.202 | −128 | 25.3 | 1.0 | 20 | 2.27 h |

**Deployed free-rollout `river_q` RMSE per gauge (control λ=0 → λ=4):**

| gauge | λ=0 | λ=1 | λ=2 | λ=4 |
|---|---|---|---|---|
| outlet (node1) | **12.28** | 16.84 | 15.20 | 16.38 |
| node74 | **0.504** | 0.901 | 0.812 | 0.834 |
| node152 | **0.183** | 0.210 | 0.226 | 0.210 |
| node183 | 0.112 | 0.106 | 0.106 | 0.119 |
| node48 | 0.100 | 0.101 | 0.100 | 0.095 |

**IMPROVEMENTS.**
- **Pooled FHV mildly better with weighting (weak evidence).** 0.111 → 0.088 / 0.101
  / 0.098 — the up-weighting does nudge high-flow volume error down. But it is offset
  by the rollout-RMSE degradation below and does not survive as a net win.
- **The weight genuinely bit at λ=2 (evidence — closes E5's open question).**
  `w_mean` 1.37, `c_peak` 0.996, `w_max` 4.0 — so the null result is NOT because the
  lever was dormant (E5's caveat); it is a real "weighting does not help" result.

**DEGRADED / FAILURE MODES.**
- **Deployed rollout is WORSE with any weighting (evidence — the decisive read).**
  `river_q` RMSE rises at the two largest nodes: outlet 12.28 → 15.2–16.8 (+24–37%),
  node74 0.504 → 0.81–0.90 (+60–80%). Cause = peak-driven **over-shoot** (node74
  peak ratio 0.87 → 1.16 / 1.34 / 1.41; node48 0.91 → 1.10–1.15). The pooled
  `peak_ratio` moving toward 1.0 (0.963 → 1.000 at λ=2) is over-shoot at some nodes
  cancelling under-shoot at others, **not** a genuine accuracy gain.
- **1-step and `river_h` untouched / worse (evidence).** `val_q_1step` 0.18–0.20 (no
  trend), spatial `river_h` NSE −118 → −122 → −153 → −128 (worse at λ=2). Peak
  weighting does not address the binding `river_h`/1-step constraint.
- **Weight activation is non-monotonic / seed-driven (evidence).** `w_mean` 1.00017
  (λ=1) → 1.37 (λ=2) → 1.05 (λ=4): higher λ bit *less*. Activation depends on the
  training trajectory, so even the "dose" is not clean — reinforcing that the effect
  is dominated by seed/trajectory noise.
- **Fixed-horizon is hopelessly seed-dominated (evidence — process note).** hps1
  (λ=0) is a 4th pure-pushforward replicate; final fixed RMSE now spans
  **{9.3, 30.7, 95, 1559}** across identical-intent configs. hps3's low frac_gt2
  (0.656) and RMSE 8.87 are NOT attributable to λ. Ignore single-cell fixed-horizon.

**HYPOTHESES.**
- *(evidence)* Peak-weighting is **orthogonal to the rollout operator** (amp flat,
  KGE flat) but **not free** in free rollout: forcing sharper peaks in the training
  objective biases the model toward over-shoot, which spatial accumulation amplifies
  at the large nodes → higher rollout RMSE. So the lever can only trade peak-bias for
  volume-error, it cannot add skill.
- *(speculation)* The `river_h` failure (NSE ~−120 everywhere, invariant to loss/
  weighting/curriculum across E5–E10) is a **scaling/normalisation** problem, not a
  loss-shape problem — consistent with it being the one axis untouched by every loss
  lever tried. Prioritise TODO §1.

**RECOMMENDATIONS.**
1. **Close the peak-weight lever.** Keep `peak_lambda = 0` (plain MSE) on the
   pushforward base; do not sweep λ/γ/w_max further. E5 (inert) + E10 (bit but no
   benefit) together close it.
2. **Redirect to distribution/scaling (TODO §1).** `river_h` and outlet-dominated
   rollout error are the binding failures and are invariant to every loss lever
   tried (E4→E10) — they are the next real target.
3. **Apply the engineering fix** (gate `peak_stats` on `loss_type == :huber ||
   peak_lambda > 0`) only if peak-weighting is ever revisited, to drop the
   `peak_delta = 1000` workaround.
4. **Never trust single-cell fixed-horizon RMSE** (now replicated 4× at 9.3–1559);
   judge on deployed free-rollout per-node RMSE/peak and multi-seed replicates.

## E9 — hybrid-loss (`pushforward_tf_weight`) sweep — `sava_small_v081_e9_hybrid_sweep`

**NAME:** `sava_small_v081_e9_hybrid_sweep` (box search, 4 cells, one knob:
`train.strategy.pushforward_tf_weight ∈ {0.0, 0.25, 0.5, 0.75}` = `(1−α)`;
everything else at the E6-pushforward base — `rollout_grad = pushforward`, MSE,
`lr_start = 1.3457e-4`, `mb_theta = 1.0`, full `[1,2,5,8,10]` curriculum, 8×64 /
mlp 2, 250 epochs, always-on q≥0 floor). **The keystone experiment** — designed to
break the E6 trade-off (pushforward stabilises the rollout but wrecks the 1-step /
`river_h` fit) by adding a teacher-forced 1-step term to the endpoint-only loss:
`L = α·L_pushforward(endpoint) + (1−α)·L_1step(ground-truth)`. Config:
[experiments/sava_small_v081_e9_hybrid_sweep/config.toml](../experiments/sava_small_v081_e9_hybrid_sweep/config.toml).

**SUMMARY — the hybrid loss fixes the *teacher-forced* 1-step metric, but that gain
does NOT carry to the free rollout, so tf is not worth it on this evidence.** The
`val_*_1step` metrics recover dramatically (`val_q_1step` 0.152→0.017 at tf=0.25,
≈ full_bptt; `val_h_1step` 9.34→1.55) — but these are **one-step, ground-truth-fed**
errors, not deployed skill. Re-scoring the **free-rollout** validation timeseries
(what actually gets plotted) shows the tf=0.25 benefit nearly evaporates and is
**outlet-localised**: outlet `river_q` RMSE 13.30→12.27 (−8%, plausibly within the
seed band), MAE −27%, peak ratio 0.71→0.86 are the only clear wins; upstream gauges
are a wash-or-worse (node152 q_RMSE +39%), and rollout `river_h` is **mixed to
worse** (outlet RMSE 3.91→6.40, +64%). So the E6 1-step/`river_h` regression is
**not** meaningfully fixed *in rollout*, and the **3.3× wall-clock cost** buys
little. What IS seed-robust: the low-amp pushforward solution is lost with any tf
(`amp` 7.7→13.5), `tf ≥ 0.5` is catastrophic (over-prediction → fixed RMSE 1e6→1e31),
and `amp` decouples from rollout stability. **Revised verdict: keep pure pushforward
as the base;** tf=0.25 needs a multi-seed run proving the outlet peak/MAE gain is
real before adoption. (Seed context: pure pushforward replicated 3× gives fixed RMSE
{95, 9.3, 30.7} — the horizon metric is seed-dominated.)

**The four cells (best-epoch; outlet = node1_r1 daterange, true max 122.4):**

| cell | tf_weight | **final_amp** | best_val_rmse | final fixed RMSE | frac_gt2 | pooled KGE | outlet ratio | pbias | q_nse med / p10 | **val_q_1step** | **val_h_1step** | river_h NSE | best_ep | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hps1 | 0.0 | **7.70** | 5.90 | 30.7 | 1.0 | **0.875** | 0.71 | 6.8 | 0.977 / **0.839** | 0.152 | 9.34 | −124 | 30 | 2.23 h |
| **hps2** | **0.25** | 13.54 | 5.95 | **12.4** | **0.906** | 0.853 | 0.86 | 8.6 | 0.976 / 0.837 | **0.017** | 1.55 | −86 | 14 | 7.36 h |
| hps3 | 0.5 | 13.41 | 5.80 | **1.19e6** | 1.0 | 0.735 | 1.03 | 15.5 | 0.971 / 0.809 | 0.014 | 0.92 | −63 | 16 | 7.39 h |
| hps4 | 0.75 | 12.02 | 5.78 | **2.9e31** | 1.0 | 0.687 | 1.26 | 20.4 | 0.977 / 0.776 | **0.012** | **0.69** | −41 | 19 | 7.29 h |
| *E6 full_bptt* | *1.0* | *11.03* | *120903* | *Inf* | *1.0* | *0.681* | *1.29* | *—* | *0.976 / 0.77* | *0.0144* | *—* | *−7.9* | *13* | *4.98 h* |

**Free-rollout validation timeseries — the deployed skill (hps1 tf=0 → hps2 tf=0.25),
RMSE per gauge on the daterange window.** This is the corrective read: the
`val_*_1step` columns above are teacher-forced and do NOT represent this.

| gauge | q RMSE tf=0 | q RMSE tf=0.25 | Δq | h RMSE tf=0 | h RMSE tf=0.25 | Δh |
|---|---|---|---|---|---|---|
| outlet (node1) | 13.30 | 12.27 | **−8%** | 3.91 | 6.40 | **+64%** |
| node74 | 0.569 | 0.538 | −5% | 1.29 | 0.74 | −43% |
| node152 | 0.175 | 0.244 | **+39%** | 1.88 | 1.95 | +4% |
| node183 | 0.117 | 0.111 | −5% | 0.76 | 0.70 | −8% |
| node48 | 0.105 | 0.109 | +4% | 15.70 | 14.33 | −9% |

(Outlet q MAE 10.12→7.38 −27% and peak ratio 0.71→0.86 are the only clear tf=0.25
wins; everything else is a wash or mixed. Single seed.)

**IMPROVEMENTS.**
- **Teacher-forced 1-step metric recovers strongly (evidence — but see the caveat).**
  `val_q_1step` 0.152 → 0.017 → 0.014 → 0.012 (matches full_bptt 0.0144 at tf=0.25);
  `val_h_1step` 9.34 → 1.55 → 0.92 → 0.69; spatial `river_h` NSE −124 → −41. **CAVEAT:
  these are one-step, ground-truth-fed errors and do NOT translate to free rollout**
  (see the rollout table) — so E6 caveat #2 is *not* meaningfully fixed where it
  matters. Treat this as a diagnostic of the 1-step map, not a skill gain.
- **Modest outlet-only `river_q` rollout gain at tf=0.25 (weak evidence, single
  seed).** Outlet RMSE −8% (likely within seed band), MAE −27%, peak ratio 0.71→0.86.
  Confined to the outlet; upstream gauges wash-or-worse, rollout `river_h` mixed to
  worse (outlet +64%). Not worth the 3.3× cost without multi-seed proof.

**DEGRADED / FAILURE MODES.**
- **Low-amp pushforward solution lost with any tf > 0 (evidence).** `amp`
  7.70 → 13.5 / 13.4 / 12.0 — jumps to ≥ full_bptt's 11 and stays. Low-amp and
  sharp-1-step are **opposed**; this knob trades along that axis, it does not escape
  it. The literal win condition (`amp ≈ 6.4`) is **not met**.
- **`tf ≥ 0.5` catastrophic (evidence).** Over-prediction (outlet 0.71 → 1.03 → 1.26,
  pbias 6.8 → 20.4, pooled peak_ratio 0.91 → 1.16) compounds in free rollout → fixed
  RMSE 1.19e6 (tf=0.5), **2.9e31** (tf=0.75). Benign KGE decays 0.875 → 0.687,
  q_nse p10 0.839 → 0.776.
- **`amp` decouples from rollout stability under hybrid training (evidence —
  undercuts E6 mechanism).** hps2 has the **highest** amp (13.5) yet the **best**
  (bounded) rollout; hps3/hps4 have similar amp but diverge. So amp no longer orders
  fixed-horizon outcomes — the tf-driven **over-prediction bias**, not amp, is the
  operative failure mode at high tf. E6's "amp is the mechanism" needs multi-seed
  re-examination.
- **`river_h` improved but NOT solved (evidence).** Best NSE −41 (tf=0.75) is still
  badly negative; the hybrid loss lifts it but does not make it skilful.
- **`tf > 0` is ~3.3× slower for little rollout gain (evidence).** 26.5 ks vs 8.0 ks
  (extra teacher-forced 1-step forward/backward per step). Combined with the
  near-flat free-rollout timeseries, the cost/benefit is poor. `best_epoch` is early
  everywhere (14–30) and `best_val_rmse` is flat (~5.8–5.95) — the selection metric
  does not discriminate.
- **The `val_*_1step` metric is misleading for this loss (evidence — process
  lesson).** It improved 9× yet the deployed free-rollout timeseries barely moved
  (and worsened on outlet h). Judging the hybrid loss on `val_*_1step` overstated
  its value; always score it on **free-rollout** timeseries.

**HYPOTHESES.**
- *(evidence)* The system has a **fundamental low-amp ↔ sharp-1-step tension**: the
  contractive/smoothed map pushforward learns (low amp, stable rollout) is the same
  one that fits the sharp 1-step transition poorly, and vice versa. Loss weighting
  moves along this axis but cannot escape it — a faithful rollout AND a sharp 1-step
  map may be **unreachable via loss weighting alone**.
- *(speculation)* The amp/rollout decoupling suggests the fixed-horizon divergence
  at high tf is a **bias** phenomenon (systematic over-prediction from matching
  sharp peaks) rather than a **gain** (amp) phenomenon — so the remaining lever may
  be an explicit anti-bias / volume constraint, not further gain reduction.

**RECOMMENDATIONS.**
1. **Keep pure pushforward (`tf_weight = 0`) as the working base.** On this
   single-seed evidence the free-rollout timeseries barely improves at tf=0.25 (and
   worsens on outlet h) for 3.3× the wall-clock — not a justified base change. **Do
   NOT exceed 0.25** regardless (over-prediction blow-up at tf≥0.5).
2. **Do not adopt tf=0.25 without a multi-seed {0, 0.25} run (≥3 seeds each)**
   proving the outlet `river_q` peak/MAE gain is real and not seed noise (the 8%
   outlet RMSE gap is plausibly within the pure-pushforward seed band). Same run
   settles the amp/rollout decoupling.
3. **Score any future hybrid-loss variant on free-rollout timeseries, never on
   `val_*_1step`** — E9 showed a 9× teacher-forced-metric gain that did not survive
   rollout.
4. **Treat `river_h` as unsolved** (rollout outlet h got *worse* at tf=0.25); it
   likely needs the separate scaling/normalisation work (TODO §1), not tf.
5. **Re-examine the E6 "amp is the mechanism" claim** in light of the decoupling;
   the operative high-tf failure is over-prediction bias, not gain.

## E8 — curriculum-schedule (rollout-depth) sweep — `sava_small_v081_e8_schedule_sweep`

**NAME:** `sava_small_v081_e8_schedule_sweep` (box search, 3 cells; whole
`[train.strategy]` table replaced per cell so steps + durations stay consistent,
each dict summing to 250 epochs, `pushforward` + MSE + `tf_weight = 0` pinned).
Deepest curriculum horizon **10 → 20 → 30** (`nhz` 11 / 21 / 31); everything else
at the E6-pushforward base (`lr_start = 1.3457e-4`, `mb_theta = 1.0`, 8×64 /
mlp 2, 250 epochs, always-on q≥0 floor). Config:
[experiments/sava_small_v081_e8_schedule_sweep/config.toml](../experiments/sava_small_v081_e8_schedule_sweep/config.toml).

**SUMMARY — the hypothesis is refuted, but the design confounds depth with epoch
budget; a clean schedule study is still owed.** The premise was that training to
`eval_horizon = 30` would close the 3× train/eval-horizon extrapolation and cut
the horizon-30 over-shoot. Instead, going deeper made the **fixed-horizon rollout
worse** (`frac_gt2` 0.75 → 0.875 → 1.0; final RMSE 9.3 → 15.7 → **3.6e6**), and the
horizon-30 cell — trained exactly to the eval horizon — still diverges at
horizon 30. **But the sweep varies three things at once** (max horizon,
phase-count, and — under the fixed 250-epoch budget — epochs-per-phase): going
hps1→hps3 **halved** the `steps=1` teacher-forced budget (50 → 25 epochs). So the
two failure modes have *different* causes: the **gradient explosions** at depth 30
(137 non-finite skips, 66 back-offs, max grad 1.2e16; diverging final models,
val_rollout 416 @20 / 3.9e13 @30) are genuinely **depth-driven** (longer detached
prefix ⇒ endpoint further off-trajectory ⇒ larger gradient), whereas the **1-step
/ `river_h` starvation** (val_h_1step 8.6 → 122 → 161) is largely a **shallow-
budget** effect (less time at `steps=1`, and training ends far from it).
**Critically, all three best checkpoints sit in the `steps=1` phase** (epochs
18/10/22), so the deep training never once beat the shallow checkpoint
(`best_val_rmse` flat 5.68→6.10→6.01) and the apparent best-epoch skill gain is
**not** a depth benefit. Net: **as-designed deepening is a net negative on pure
pushforward, but this cannot isolate depth from budget — a proper schedule study
(shallow budget held fixed) is still needed. Run E9 first.**

**The three cells (best-epoch; outlet = node1_r1 daterange, true max 122.4):**

| cell | deepest | **final_amp** | best_val_rmse | final fixed RMSE | **frac_gt2** | final peak_ratio | pooled KGE | pbias | fhv | outlet ratio | q_nse med / p10 | val_q_1step | val_h_1step | best_ep | skips | backoffs | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hps1 | 10 | 7.40 | **5.68** | **9.3** | 0.75 | 2.92 | 0.885 | 6.4 | 0.072 | 0.81 | 0.979 / 0.843 | **0.144** | **8.59** | 18 | 0 | 0 | 2.27 h |
| hps2 | 20 | **8.97** | 6.10 | 15.7 | 0.875 | 20.98 | 0.938 | 3.0 | 0.036 | 0.80 | 0.979 / **0.866** | 0.156 | 122.5 | 10 | 0 | 0 | 3.06 h |
| hps3 | 30 | 6.73 | 6.01 | **3.6e6** | 1.0 | 5.8e6 | **0.948** | **2.6** | **0.028** | 0.76 | 0.979 / 0.856 | 0.292 | 161.3 | 22 | **137** | **66** | 3.82 h |
| *E6 pushforward* | *10* | *6.37* | *1196* | *61,092* | *1.0* | *1.67e5* | *0.876* | *—* | *—* | *0.93* | *0.980 / 0.85* | *0.165* | *—* | *200* | *0* | *0* | *2.03 h* |

**IMPROVEMENTS.**
- **Best-epoch benign skill trends up across cells — but NOT attributable to depth
  (evidence + confound).** pooled KGE 0.885 → 0.938 → 0.948, pbias 6.4 → 3.0 → 2.6,
  fhv 0.072 → 0.036 → 0.028. The monotone *direction* is real, but **all three
  best checkpoints are in the `steps=1` phase** (best_epoch 18/10/22), so none saw
  deep rollout training — this cannot be a depth benefit. It reflects shallow-phase
  training + differing `steps=1` epoch counts + seed variance. Do **not** read it
  as "deeper ⇒ better skill."
- **q_nse p10 mildly up** (0.843 → 0.866 → 0.856) — marginal, within seed band.

**DEGRADED / FAILURE MODES.**
- **Fixed-horizon over-shoot got WORSE, not better (evidence — hypothesis
  refuted).** `frac_gt2` 0.75 → 0.875 → **1.0**; final RMSE 9.3 → 15.7 → **3.6e6**.
  Training to the eval horizon did **not** close the gap; the horizon-30 cell
  diverges at horizon 30.
- **Depth destabilises pushforward training (evidence — depth-driven).** hps3:
  **137 non-finite skips, 66 back-offs, max grad 1.2e16**; final models diverge
  (val_rollout 416 @20, 3.9e13 @30); `best_epoch` collapses (18 → 10). The
  unanchored deep-endpoint loss explodes — this one IS intrinsic to horizon length.
- **1-step / `river_h` starved harder (evidence — mostly budget-driven).**
  val_h_1step 8.6 → 122 → 161; val_q_1step 0.144 → 0.292. The `steps=1` budget
  shrank 50 → 25 and all best checkpoints are shallow, so this is largely the
  epoch-redistribution confound, not pure depth — the cost E9 (per-phase 1-step
  term) is designed to offset regardless of budget.
- **`amp` non-monotonic (evidence).** 7.40 → 8.97 → 6.73 — no clean depth effect;
  the mid cell is *worse* than the control.
- **1.35–1.68× slower** (2.27 → 3.06 → 3.82 h) with no usable payoff.

**HYPOTHESES.**
- *(evidence)* The two failures have **different causes** (the design confounds
  them): gradient explosions are **depth-driven** (longer detached prefix ⇒ larger
  endpoint residual/gradient — intrinsic to horizon length, budget-independent),
  while 1-step/`river_h` decay is **budget-driven** (the `steps=1` phase shrank
  50 → 25 and training ends far from it — would occur even if the added phases were
  shallow). The sweep cannot separate these on its own.
- *(evidence)* Depth delivered **no realised benefit**: best checkpoints are all
  shallow, `best_val_rmse` is flat, and the final deep models diverge. Whatever
  useful signal deep supervision carries was never converted into a selected model.
- *(speculation)* With `pushforward_tf_weight > 0` (E9) re-injecting the 1-step
  term at *every* phase, the shallow map is protected regardless of budget,
  neutralising the budget failure and letting depth be tested on its own merits —
  the **E8×E9 stack**. Separately, a **budget-controlled schedule study** (below)
  is needed to isolate depth from fragmentation.

**RECOMMENDATIONS.**
1. **Do NOT deepen the curriculum on pure pushforward** (`tf_weight = 0`). It is a
   net negative: worse fixed-horizon rollout, unstable training, starved h/1-step,
   slower.
2. **Run E9 first, then revisit depth as an E8×E9 stack** — deepen only with a
   non-zero `pushforward_tf_weight` (and consider tighter `grad_clip` / more
   `phase_backoff` for the horizon-30 phase).
3. **Keep the deepest phase at ~10 for now** (hps1 is the best fixed-horizon cell
   and the only clean-training one).
4. **A clean schedule study is STILL OWED — this sweep confounds depth with epoch
   budget.** Design the follow-up with proper controls: (a) **isolate
   fragmentation** — max horizon 10 but chopped into ~9 short phases (hps3's
   phase-count without the depth); (b) **isolate depth** — add deep phases while
   **holding ~50 epochs at `steps=1`** (steal from mid phases or raise total
   epochs); (c) consider a **longer total budget** so deep phases get real epochs
   *and* the shallow map is preserved. Only then can "is deeper better?" be
   answered. Until then, treat E8's depth verdict as *as-designed*, not intrinsic.
5. **Report `frac_gt2` with a seed caveat** — this control gave 0.75 vs E7's
   identical-config 1.0; the metric is seed-chaotic (do not rank cells by it).

## E7 — training-noise (input-perturbation) sweep — `sava_small_v081_e7_noise_sweep`

**NAME:** `sava_small_v081_e7_noise_sweep` (box search, 5 cells, one knob:
`train.strategy.noise_scale ∈ {0.0, 0.01, 0.03, 0.1, 0.3}`, normalized/z-scored
state units; everything else at the **E6-pushforward** base — `rollout_grad =
pushforward`, MSE, `lr_start = 1.3457e-4`, `mb_theta = 1.0`, full `[1,2,5,8,10]`
curriculum, 8×64 / mlp 2, 250 epochs). `hps1` (noise=0) is the E6-pushforward
control — but note the code now carries the always-on q≥0 floor
(`_floor_q_norm_nonnegative`, src/gnn.jl) added *after* E6, so `hps1` doubles as a
(seed-confounded) A/B of that floor vs E6. Config:
[experiments/sava_small_v081_e7_noise_sweep/config.toml](../experiments/sava_small_v081_e7_noise_sweep/config.toml).

**SUMMARY — noise is a weak/chaotic lever; the q≥0 floor is the boundedness win.**
Two findings. **(1)** The hypothesis that *noise lowers `amp`* is **rejected**:
`amp` sits at 6.8–7.5 in every cell with no trend vs noise. **(2)** Every E7 cell
is far more bounded than E6, including the noise=0 control — `hps1` improved
`best_val_rmse` **1196 → 5.60** and final fixed RMSE **61,092 → 95** vs the E6
pushforward cell, whose only code delta is the always-on q≥0 floor. That ~200–600×
jump exceeds E5's ~80× chaotic band, so the **floor** is the credible driver (not
seed). Noise *did* move `frac_gt2` off 1.0 — but only at 0.01 (0.844) and 0.1
(0.875); 0.03 **diverged** (final RMSE 1.07e13) and 0.3 held at 1.0. The
non-monotone zig-zag = a weak signal swamped by seed chaos, not a dose-response.
Noise mildly improves benign skill with dose (pooled KGE 0.782→0.908, pbias
11.7→4.7) and the 1-step/`river_h` fit peaks at 0.1, but `river_h` NSE stays
catastrophic (~−93) — **noise does not fix the pushforward h/1-step regression.**
No cell diverges on the benign daterange (outlet ratio 0.68–1.0). Best balance:
**hps4 (noise=0.1)**.

**The five cells (best-epoch; outlet = node1_r1 daterange, true max 122.4):**

| cell | noise | **final_amp** | best_val_rmse | final fixed RMSE | **frac_gt2** | final peak_ratio | pooled KGE | pooled peak_ratio | pbias | outlet ratio | q_nse med / p10 | val_q_1step | val_h_1step | best_epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hps1 | 0.0 | 7.09 | **5.60** | 95 (finite) | 1.0 | 69.8 | 0.782 | 1.045 | 11.7 | 0.97 | 0.978 / 0.812 | 0.150 | 8.05 | 26 |
| hps2 | 0.01 | 7.52 | 7.28 | **11.4** | **0.844** | 3.91 | 0.880 | 0.897 | 6.3 | 0.68 | 0.981 / 0.856 | 0.164 | 9.53 | 18 |
| hps3 | 0.03 | 7.10 | 6.26 | **1.07e13** *(diverges)* | 1.0 | 3.34e13 | 0.849 | 1.006 | 7.8 | 1.00 | 0.981 / 0.854 | 0.149 | 8.32 | 150 |
| **hps4** | **0.1** | 7.15 | 6.38 | **15.9** | **0.875** | 5.74 | 0.872 | 0.966 | 6.1 | 0.81 | 0.982 / 0.859 | **0.109** | **6.24** | 22 |
| hps5 | 0.3 | **6.76** | 6.32 | 19.3 | 1.0 | 18.6 | **0.908** | 0.924 | **4.7** | 0.81 | 0.983 / **0.877** | 0.136 | 6.91 | 44 |
| *E6 pushforward* | *0* | *6.37* | *1196* | *61,092* | *1.0* | *1.67e5* | *0.876* | *1.086* | *—* | *0.93* | *0.980 / 0.85* | *0.165* | *—* | *200* |

**IMPROVEMENTS.**
- **q≥0 floor → step-change in boundedness (evidence, seed-confounded).** hps1
  vs E6-pushforward (identical config, only the added always-on floor differs):
  `best_val_rmse` 1196→5.60, final fixed RMSE 61,092→95. Magnitude ~200–600× is
  beyond the E5 chaotic band (~80×), so credited to the floor. Carry it forward.
- **Mild noise nudges `frac_gt2` below 1.0 (weak evidence).** 0.01→0.844 and
  0.1→0.875 are the first sub-1.0 `frac_gt2` values in the series — but see the
  non-monotonicity caveat; treat as fragile until multi-seed confirmed.
- **Benign skill improves with noise dose (evidence).** pooled KGE 0.782→0.908,
  pbias 11.7→4.7, q_nse p10 0.812→0.877 monotone in noise. Cheap regularisation.
- **1-step/h best at noise=0.1 (evidence, partial).** `val_q_1step` 0.109 and
  `val_h_1step` 6.24 are the sweep-best — mild help, not a cure.

**DEGRADED / FAILURE MODES.**
- **`amp` does NOT respond to noise (evidence — hypothesis rejected).** Flat
  6.8–7.5, no trend. The `frac_gt2` gains are NOT via reduced per-step gain, so
  the "noise → contractive fixed point" story is unsupported here.
- **Non-monotone / chaotic dose-response (evidence).** 0.03 diverged (final RMSE
  1.07e13) between two stable neighbours (0.01, 0.1); 0.3 held `frac_gt2`=1.0.
  Single-seed noise cannot be trusted as a clean lever — the signal is weak
  relative to seed chaos (per E5).
- **`river_h` regression persists (evidence).** river_h NSE ~−82…−109 in every
  cell; noise does not fix the endpoint-only-supervision cost. → E9's job.

**HYPOTHESES.**
- *(evidence)* The q≥0 floor removes a negative-discharge feedback path that let
  the free rollout blow up; it bounds the tail far more than any noise level. The
  cleanest confirmation is a **fixed-seed A/B toggling the floor** (code toggle,
  since it is now always-on).
- *(speculation)* Noise's real value is mild off-trajectory regularisation
  (benign KGE/pbias, slight 1-step/h help), not stability. Its `frac_gt2` effect
  may be a seed artefact; a 3-seed repeat at {0, 0.05, 0.1} would settle it.

**RECOMMENDATIONS.**
1. **Adopt the q≥0 floor as permanent** (already always-on) and treat it as the
   headline boundedness lever, not noise.
2. **The cell RANKING is within seed noise — do NOT read "0.1 is best" as a tuned
   value.** The whole-sweep pooled-KGE spread is 0.782→0.908 = **0.13**, at/below
   the single-config reseed band (~0.15, E1 vs E2-hps1). Only the **monotone
   direction** survives that noise floor: benign skill improves smoothly with dose
   (q_nse p10 0.812→0.877, pbias 11.7→4.7, KGE ↑ across all 5 points — a 5-point
   monotone trend is far harder to get by chance than a single "winner"). The
   per-cell winners on the **chaotic** axes (`frac_gt2` zig-zag 1.0/0.844/1.0/
   0.875/1.0; 0.03 diverging to RMSE 1e13 between two stable neighbours;
   `val_q_1step` un-ordered) are **coincidence**, not a ranking. `frac_gt2` is also
   coarse (32 anchors → ~0.03 granularity), so its sub-1.0 dips are not reliable.
3. **If used, carry noise for the REGULARISATION direction, not a tuned value:**
   pick a **small** ~0.05 for the smooth benign-skill benefit — NOT because 0.1 won
   a horse race — and only keep it if a **≥3-seed repeat at {0, 0.05, 0.1}**
   confirms the effect clears seed noise. Given `amp` did not move and the effect
   is within the reseed band, noise is a **low-priority** knob.
4. **Do not raise noise ≥0.3** (no benefit; `frac_gt2` back to 1.0).
5. **Priority stays E9 (hybrid loss)** — the persistent `river_h`/1-step
   regression is untouched by noise and is the next real target.

## E6 — rollout-gradient strategy comparison — `sava_small_v081_e6_rollout_grad`

**NAME:** `sava_small_v081_e6_rollout_grad` (box search, 3 cells, one knob:
`train.rollout_grad ∈ {full_bptt, detached, pushforward}`; everything else held
at the stable E1 **MSE** baseline — `lr_start = 1.3457e-4`, `mb_theta = 1.0`,
`noise_scale = 0`, full `[1,2,5,8,10]` curriculum, 8×64 / mlp 2, 75,009 params,
250 epochs). First head-to-head of the training-time error-correction lever that
E4/E5 identified as the real priority (loss/θ are NOT stability levers). Config:
[experiments/sava_small_v081_e6_rollout_grad/config.toml](../experiments/sava_small_v081_e6_rollout_grad/config.toml).

**SUMMARY — the first lever that moves the mechanism.** `pushforward` is the
**first intervention in the entire E-series to lower `amp`** (11.03 → **6.37**,
−42%) — a quantity that was invariant to loss (E5), δ (E5), and, by design, θ=1.
It is also the **only** cell with a **finite** final fixed-horizon RMSE (61,092 vs
full_bptt 9.4e18, detached 1.26e21) and the best fixed-horizon selection score
(`best_val_rmse` **1,196** — ~100× better than full_bptt's 120,903, ~1000× better
than detached's 1.18e6). On the benign daterange rollout it is the best by every
pooled and per-node measure (pooled KGE **0.876**, peak_ratio **1.086**, per-node
`river_q` NSE median **0.980** / p10 **0.85** / mean **0.95**, outlet 0.93×), and
it trains **2.4× faster** (2.03 h vs 4.98 h) thanks to O(1) tape depth. **This
confirms the E4/E5→E6 thesis: the rollout instability is autoregressive exposure
bias, attackable via the rollout gradient / state exposure — NOT via the
objective.** Caveats: (1) it only **damps** the stiff-anchor divergence (~13
orders: final peak_ratio 1.7e5 vs 9.4e18), it does **not cure** it —
`frac_gt2 = 1.0` still holds for all three modes; (2) it badly **regresses the
teacher-forced 1-step fit and `river_h`** (classic pushforward trade-off, below).
`detached` is a **net negative**; `full_bptt` validates as the E1 control.

**The three modes (best-epoch, daterange outlet = node1_r1, true max 122.4):**

| mode | q_1step | outlet ratio | pooled KGE | pooled peak_ratio | per-node NSE med / p10 / mean | **final_amp** | best_val_rmse | final fixed RMSE | final peak_ratio | train |
|---|---|---|---|---|---|---|---|---|---|---|
| full_bptt *(=E1 control)* | 0.0144 | 1.29× | 0.681 | 1.43 | 0.976 / 0.77 / 0.91 | 11.03 | 120,903 | **Inf** (9.4e18) | 9.4e18 | 4.98 h |
| detached | 0.0182 | 1.32× | −0.054 | 2.84 | 0.968 / 0.64 / 0.60 | **12.71** | 1.18e6 | **Inf** (1.26e21) | 1.26e21 | 4.67 h |
| **pushforward** | **0.165** | 0.93× | **0.876** | **1.086** | **0.980 / 0.85 / 0.95** | **6.37** | **1,196** | **61,092** | **1.67e5** | **2.03 h** |
| *E1 (MSE ref)* | *0.0144* | *1.9×* | *0.577* | *2.29* | *0.973 / 0.772 / 0.884* | *11.5* | *—* | *Inf (1e23)* | *—* | *—* |

**IMPROVEMENTS.**
- **Rollout gradient IS a lever on `amp` (evidence — the key result).**
  Pushforward `final_amp` 6.37 vs full_bptt 11.03 and the E1–E5 invariant ~11–12.
  `mb_gain` is still pinned at 21.946 (that term is geometry, not learnable), but
  the learnable q→h amplification finally responds — to *state exposure*, exactly
  where E5 predicted the mechanism lives.
- **Pushforward: first finite fixed-horizon rollout (evidence).** Final
  `val_rmse` 61,092 (finite) and final peak_ratio 1.7e5 vs full_bptt's Inf /
  9.4e18 — the stiff anchors are damped ~13 orders of magnitude. `best_val_rmse`
  1,196 makes it the first model whose selection metric is O(10³), not O(10⁵–10⁶).
- **Best benign-rollout skill and cheapest (evidence).** Pooled KGE 0.876 /
  peak_ratio 1.086, per-node NSE p10 0.85 (all best), and 2.4× faster wall-clock
  (no gradient checkpointing — `use_ckpt` is full_bptt-only). A rare
  accuracy-and-speed win.
- **Control validates the harness (evidence).** full_bptt reproduces E1: `amp`
  11.03 ≈ 11.5, `val_q_1step` 0.0144 = 0.0144, pooled KGE 0.681 (E1 0.577),
  outlet 1.29× (E1 1.9×, within the chaotic band per E5). Harness unchanged.

**DEGRADED / FAILURE MODES.**
- **Pushforward wrecks the teacher-forced 1-step fit and `river_h` (the
  trade-off).** `val_q_1step` 0.165 (11× worse than full_bptt's 0.0144),
  `val_h_1step` 8.58 (10× worse), spatial `river_h` NSE median **−96.3** (vs
  full_bptt −7.9). It supervises only the endpoint after a detached k-1 roll, so
  in deep curriculum phases it never directly constrains the 1-step map — q-rollout
  robustness is bought at the cost of the 1-step operator and h everywhere.
  **Note the 1-step q metric is a *poor proxy*:** pushforward has the worst
  `val_q_1step` yet the *best* q rollout — select on rollout/spatial, not 1-step.
- **Still not truly stable.** `frac_gt2 = 1.0` for all three — every mode still has
  anchors that exceed 2× at some point. Pushforward damps but does not eliminate
  the stiff-anchor divergence.
- **`detached` is a net negative (evidence).** Worse 1-step (0.0182), worse pooled
  KGE (−0.054), worst spatial mean (0.60), and `amp` actually *rises* to 12.71 —
  the myopic truncation-length-1 gradient exposes the model to its own drifted
  state but gives no multi-step credit assignment, so it never learns to correct
  the compounding amplification. Barely faster than full_bptt. Drop it.
- **`river_h` remains broken in all modes** (spatial NSE −7.9 / −38.6 / −96.3);
  the q-side gains do not transfer to h, and pushforward makes h worse.

**HYPOTHESES.**
- **The instability is autoregressive exposure bias — CONFIRMED as attackable
  (evidence).** E5 closed the loss (amp δ-invariant); E6 opens the rollout gradient
  (amp mode-dependent, pushforward −42%). The lever that moves the mechanism is
  *what state the model is trained on*, not *how errors are penalised*. This is the
  central positive result of the E-series.
- **Endpoint-only supervision under-constrains the 1-step map and h (evidence +
  interpretation).** Pushforward's finite-rollout gain and its 1-step/h regression
  are two faces of the same mechanism (supervise the drifted endpoint, not the
  step). Suggests a **hybrid**: pushforward endpoint term + a teacher-forced 1-step
  term to recover the 1-step operator and h. Speculation until tested.
- **Damped-not-cured ⇒ pushforward is necessary but not sufficient (interpretation).**
  `frac_gt2` still 1.0 implies a residual instability that a single lever won't
  close; pair pushforward with noise injection and the q≥0 floor.

**RECOMMENDATIONS.**
1. **Adopt `pushforward` as the base rollout-gradient mode.** First lever to move
   `amp` and the first finite fixed-horizon rollout; best benign-rollout skill and
   2.4× cheaper. Make it the new baseline for the stability workstream.
2. **Fix the pushforward 1-step/`river_h` regression** — highest-priority follow-up.
   Try a **hybrid loss** (pushforward endpoint term + teacher-forced 1-step term),
   or restrict pushforward to the deeper curriculum phases only (keep steps=1–2
   teacher-forced). Goal: keep `amp`↓ and finite rollout while recovering
   `val_q_1step` and h.
3. **Drop `detached`** — a dead end (myopic gradient, `amp` rises, worse across the
   board).
4. **Stack the remaining levers on the pushforward base** now that it works:
   (a) rollout **noise injection** (`noise_scale > 0`), (b) **q ≥ 0** propagating-
   state floor, (c) the distribution/scaling conditioning fixes (TODO §1) for the
   downstream nodes. Natural next experiment: **pushforward × noise_scale**.
5. **Selection-metric caveat:** `val_q_1step` is a poor proxy under pushforward
   (worst 1-step, best rollout) — select and report on rollout/spatial/fixed-horizon
   metrics, not the teacher-forced 1-step.

## E5 — peak-Huber δ (knee) sweep — `sava_small_v081_e5_delta_sweep`

**NAME:** `sava_small_v081_e5_delta_sweep` (box search, 5 cells, one knob:
`train.strategy.peak_delta ∈ {0.25, 0.5, 1.0, 2.0, 5.0}`; everything else held at
the stable E1/E4 baseline — `loss_type = "huber"`, `peak_lambda = 2`,
`peak_w_max = 4`, hand-set `lr_start = 1.3457e-4`, `mb_theta = 1.0`,
`noise_scale = 0`, full `[1,2,5,8,10]` curriculum, 8×64 / mlp 2, 75,009 params,
250 epochs). Tests the **E4 δ-tail hypothesis**: with residuals in normalized
σ≈1 space, `δ = 0.5` puts the extreme-cell errors in Huber's *linear* regime,
giving a weaker restoring gradient than MSE's quadratic; raising δ should push
Huber back toward MSE. Config:
[experiments/sava_small_v081_e5_delta_sweep/config.toml](../experiments/sava_small_v081_e5_delta_sweep/config.toml).

**SUMMARY.** **Both halves of the hypothesis confirmed, and the workstream is now
closed.** (1) On the benign daterange rollout the outlet over-prediction shrinks
**monotonically as δ grows**, converging toward MSE at `δ = 5` — the δ-tail
mechanism is real, so E4's `δ = 0.5` really was too soft on the tail. (2) But δ
**does not fix the instability**: every cell's 32 fixed-horizon anchors still
diverge (`frac_gt2 = 1.0`, final RMSE 1e12–1e25), and the intrinsic amplification
is **δ-invariant** (`final_amp ≈ 11.2–12.0`, `final_mb_gain = 21.946` to 4 s.f.
in *all* five cells, identical to E1). The best the loss knob can do (δ=5 ≈ MSE)
merely reproduces E1, which was already unstable. **This decisively closes the
loss-retuning workstream** (the "if even large δ still diverges" branch of the
E5 plan): the loss shapes the benign-path over-prediction but is orthogonal to
the rollout-instability mechanism. Teacher-forced 1-step is strong and
δ-invariant throughout (`val_q_1step` 0.0136–0.0150, spatial `river_q` NSE
median 0.93–0.97).

**The δ ladder (daterange outlet = node1_r1, true max 122.4):**

| δ | outlet pred/true | pooled peak_ratio | pooled PBIAS | pooled KGE | per-node NSE median / p10 / mean | final_amp | fixed-horizon |
|---|---|---|---|---|---|---|---|
| 0.25 | **32.9×** | 25.3 | +116% | −5.06 | 0.962 / −1.19 / −5.5 | 11.52 | all diverge (1e16) |
| 0.5 (=E4) | **23,557×** | 23,693 | +39,189% | −4,751 | 0.965 / −2,755 / −3.0e6 | 11.17 | all diverge (1e13) |
| 1.0 | **862×** | 11,417 | +9,763% | −1,634 | 0.971 / −2.17 / −3.6e5 | 11.71 | all diverge (1e16) |
| 2.0 | **92.7×** | 160 | +243% | −16.0 | 0.929 / −3.16 / −44.8 | 11.97 | all diverge (1e25) |
| 5.0 | **2.72×** | 3.48 | +93% | **−0.90** | 0.964 / **−0.62** / **+0.04** | 11.76 | all diverge (1e16) |
| *E1 (MSE ref)* | *1.9×* | *2.29* | *+46%* | *0.577* | *0.973 / 0.772 / 0.884* | *11.5* | *all diverge (1e23)* |

From `δ = 0.5` upward the trend is cleanly monotone (23,557 → 862 → 92.7 → 2.72),
asymptoting to — but never quite reaching — the E1 MSE quality (δ=5 outlet 2.72×
& KGE −0.90 vs MSE 1.9× & KGE 0.577). `δ = 5` also has the best per-node tail
(p10 −0.62, mean +0.04, closest to E1), and its `amp`/`mb_gain` (11.76 / 21.9)
match E1 exactly, confirming `δ = 5 ≈ MSE`.

**IMPROVEMENTS.**
- **δ-tail mechanism CONFIRMED (evidence).** The monotone δ=0.5→5 collapse of the
  outlet over-prediction (23,557× → 2.72×) and pooled peak_ratio (23,693 → 3.48)
  is exactly the E4 prediction: larger δ ⇒ more of the residual range is
  quadratic ⇒ stronger restoring gradient on large errors ⇒ less benign-path
  over-shoot. The `δ ≥ 1` sub-trend (862 → 92.7 → 2.72) is the cleanest evidence
  because peak-weighting is inert there (see caveat), isolating pure δ.
- **1-step fit preserved and δ-invariant** — `val_q_1step` 0.0136–0.0150 across
  the whole ladder; the loss reshaping never harms next-step skill.

**DEGRADED / FAILURE MODES.**
- **δ does NOT stabilise the rollout (the decisive negative result).** All 5
  cells: `frac_gt2 = 1.0` (every anchor diverges every epoch), final fixed-horizon
  RMSE 1e12–1e25, final peak_ratio 1e13–1e25. No δ keeps the stiff anchors
  bounded. Even the MSE-like `δ = 5` only *matches* E1, which is itself unstable.
- **Amplification is completely δ-invariant** — `final_amp` 11.2–12.0 and
  `final_mb_gain = 21.946` (identical to 4 s.f.) in all cells. The actual q→h
  amplification that drives the blow-up does not respond to the loss knob at all.
  This is the single most decisive datum: **the loss cannot touch the mechanism.**
- **δ = 0.5 produces negative outlet flow** (daterange `river_q_pred` min −1,061
  m³/s) — the same unphysical-flow pathology as E3, reinforcing the un-floored
  propagating-state gap (TODO physics-floor item).

**HYPOTHESES.**
- **Loss-retuning is orthogonal to rollout stability — WORKSTREAM CLOSED
  (evidence).** Two independent signals: (a) the δ-invariant `amp`/`mb_gain`, and
  (b) uniform anchor divergence across the whole ladder. The loss moves *where the
  benign trajectory sits* (over-prediction magnitude) but not *whether the rollout
  is stable*. Consistent with DECISIONS.md (stability lives in the rollout
  operator, not the objective) and with E4's disentangling conclusion.
- **Divergent-magnitude is chaotic / not reproducible (evidence, caveat).** E4 and
  E5-hps2 are the **same config** (Huber δ0.5, same LR) yet the outlet ratio is
  286× (E4) vs 23,557× (E5) — an ~80× spread between identical configs. The
  absolute magnitude at the unstable end is seed/chaos-sensitive; **only the
  qualitative δ trend is trustworthy**, not the precise ratios. (This also
  retro-cautions the E4 "286×" number.)
- **Peak-weighting went inert for δ ≥ 1 (anomaly, needs an engineering check —
  speculation).** `final_c_peak`/`final_w_mean`/`final_w_max` = 0.57/1.03/4.0 for
  δ∈{0.25,0.5} but **0/1/1** for δ∈{1,2,5}, splitting exactly at δ=1 even though
  `peak_lambda`/`peak_w_max` are fixed across all cells and the peak weight is
  δ-independent by construction ([src/preprocess.jl](../src/preprocess.jl#L468),
  [src/strategy.jl](../src/strategy.jl#L199)). This is a *diagnostic-reporting*
  split at worst (final-batch had no peak content) or a δ/peak-weight coupling at
  worst; it does not change the conclusion (the δ≥1 sub-trend is monotone with
  peak-weighting uniformly off), but it warrants a look before any future
  peak-weighted run. Flag for engineering.

**RECOMMENDATIONS.**
1. **Close the loss-retuning workstream.** δ (and, from E4, λ) cannot stabilise
   the rollout; `amp`/`mb_gain` are loss-invariant. Do not run a full λ/γ/w_max
   peak sweep until the rollout is stable. If a loss must be picked now, **use MSE**
   (δ=5 only asymptotes toward it and stays slightly worse).
2. **The priority is training-time error correction against exposure bias** — this
   is exactly what **E6** (`rollout_grad ∈ {full_bptt, detached, pushforward}`,
   already prepped) tests. E5 is the negative control that makes E6 the clear next
   step: the mechanism (`amp`) is untouched by the loss, so it must be attacked in
   the rollout gradient / state exposure, not the objective.
3. **Enforce `q ≥ 0` on the propagating state** — δ=0.5's −1,061 m³/s outlet flow
   re-confirms the un-floored decoder/state-q gap (TODO physics-floor item).
4. **Engineering: check the δ≥1 peak-weight-inert anomaly** (c_peak→0, w_max→1 at
   δ=1) before the next peak-weighted run.
5. **Report metric caveat:** at the unstable end, absolute over-prediction ratios
   are chaotic (E4 286× vs E5 23,557× same config). Trust trends and the
   δ-invariant `amp`/`mb_gain`, not single divergent magnitudes.

## E4 — single Huber baseline at a sane LR — `sava_small_v081_e4_huber_baseline`

**NAME:** `sava_small_v081_e4_huber_baseline` (single train run, no autotune;
peak-weighted Huber `loss_type = "huber"`, `peak_delta = 0.5`, `peak_lambda = 2`,
`peak_w_max = 4`, full `[1,2,5,8,10]` curriculum, 8×64 / mlp 2, 75,009 params,
250 epochs). **LR hand-set to the known-stable E1/E2 value `lr_start = 1.3457e-4`**
(NOT autotuned) so the *only* change vs E1 is MSE→Huber. Config:
[experiments/sava_small_v081_e4_huber_baseline/config.toml](../experiments/sava_small_v081_e4_huber_baseline/config.toml).

**SUMMARY.** Teacher-forced 1-step is strong and matches E1 (`val_q_1step`
0.0140, spatial `river_q` NSE **0.973**, rmse 0.10). But the 30-step free rollout
is **unstable from epoch 2** (`val_peak_ratio = 1e24`, `Inf` RMSE) and
`val_peak_ratio_frac_gt2 = 1.0` at **every** epoch — all 32 anchors diverge at all
epochs. The final-epoch model is fully blown up (fixed `val_rmse = 6.8e12`,
`peak_ratio = 1.6e13`, every anchor 1e11–1e13 regardless of start-flow
percentile). `best_epoch = 104` is merely the least-bad diverging epoch
(`val_rmse = 14,899`, peak_ratio still **37,281**).

**⚠ Direction: explosive over-prediction (opposite of E3).** The best-epoch
daterange outlet rollout **over-predicts by ~290×**: `river_q_pred` max **35,012**
/ mean **4,023** vs true max **122** / mean **21.7**. This is the E1/E2 blow-up,
**not** the E3 collapse. Pooled `river_q_performance`: KGE **−76.4**, PBIAS
**+2092%**, peak_ratio **163**, r 0.52. Yet median per-cell `river_q` NSE is
**0.973** — most cells track well; a subset of (high-flow-start) nodes explode and
dominate the pooled sums. `river_h` broken as before (spatial NSE **−36.9**,
peak_err 9.6 m). Volume budget diverges (pbias +35,557%, drift 9.2e5 m³/step).

**⚠ The instability is spatially localized to the downstream / high-drainage
nodes.** The divergence is **not** network-wide. Along the upstream-area ladder
(daterange peak pred/true ratio, outlet→headwater): r1 outlet **286×** (35,012 vs
122), r2 **0.9**, r3 **1.3**, r4 **1.0**, r5 headwater **0.8** — only the outlet
explodes; every other rung tracks truth. The per-node `river_q` distribution
confirms it: **NSE median 0.973 but p10 −74.6, mean −827**; rmse median 0.10 vs
mean 87.8 (p90 28). So ~the worst 10% of nodes (the highest-drainage/most
downstream) carry the entire blow-up; the pooled KGE −76 / peak_ratio 163 are
**outlet-dominated artifacts**, and the "all 32 anchors diverge to 1e11–1e13" is
consistent with a *single* exploding node (peak_ratio maxes over nodes, so one
poisoned outlet contaminates every anchor's score). **Mechanism (evidence +
interpretation):** discharge is spatially cumulative — the routing operator sums
upstream q into each node ([src/gnn.jl](../src/gnn.jl#L310)), so a per-node
fractional over-prediction invisible in a headwater accumulates additively
downstream, and the outlet (largest drainage area, largest `postscale_q`,
`upstream_q`-dominated `net_flux`) is where the geometric per-step amplification
(`amp ≈ 11.9`, `mb_gain ≈ 21.9`) manifests first and hardest. Spatial
accumulation × temporal amplification, both maximal at the outlet.

**IMPROVEMENTS.**
- **Disentangling achieved (the point of E4).** The E3 vs E4 **direction** flip
  (collapse→0 vs explode→+∞) is **LR-driven**: same loss, only the LR differs
  (0.012 vs 1.3e-4). And the **stiff fixed-horizon anchors diverge under both MSE
  (E1) and Huber (E4)** — that intrinsic instability is loss-independent
  (`frac_gt2 = 1.0` from epoch 2, all-anchor 1e11–1e13 in E4; 1e23 in E1).
- **1-step fit fully preserved under Huber** — `val_q_1step` 0.0140 ≈ E1 0.0144;
  spatial `river_q` NSE median 0.973 = E1. The loss swap does not harm next-step
  skill, and `amp`/`mb_gain` are identical to E1 (11.9/21.9 vs 11.5/21.9) — the
  learned 1-step operator is the same; only the multi-step behaviour differs.
- **Clean run, no instability counters tripped** — `n_nonfinite_skips = 0`,
  `n_backoffs = 0`, `stopped_early = false`; the divergence is a *model* property,
  not an optimiser failure.

**⚠ CORRECTION to the initial read: Huber IS worse than MSE on the benign
daterange (same LR).** The fixed-horizon anchors diverge under both losses, but
the **daterange free trajectory does NOT** — and here Huber is dramatically worse:

| | E1 (MSE) | E4 (Huber δ0.5 λ2) |
|---|---|---|
| daterange outlet pred/true (max) | 228 / 122 = **1.9×** | 35,012 / 122 = **286×** |
| per-node `river_q` NSE median / p10 / mean | 0.973 / **0.772** / **0.884** | 0.973 / **−74.6** / **−827** |
| pooled KGE / peak_ratio | 0.577 / 2.29 | −76.4 / 163 |

At the **same LR**, MSE's daterange stays bounded (1.9×, every node fine) while
Huber's blows up ~150× more at the outlet with a catastrophic downstream tail.
So the claim "loss is not a stability lever" holds **only for the stiff-anchor
instability**; for the benign long rollout, **Huber actively hurts**.

**DEGRADED / FAILURE MODES.**
- **Huber daterange rollout ~150× worse than MSE at the same LR** (286× vs 1.9×
  outlet; per-node NSE p10 −74.6 vs +0.77). The loss switch destabilises the
  benign free rollout that MSE keeps bounded.
- **Rollout unstable from epoch 2, never recovers** — not a late-training
  degradation; the MB rollout operator is unstable across the whole run
  (`final_amp = 11.9`, `final_mb_gain = 21.9` — q→h error amplification ~12–22×).
- **Explosive over-prediction at every anchor** — unlike E3 (bounded/collapsed),
  E4 diverges from **all** start-flow percentiles (0.02–0.98), so at this LR the
  fixed-horizon instability is **not** conditional on high-flow starts.
- **`river_h` still decoupled** (spatial NSE −36.9) — the q→h coupling amplifies
  rather than damps, consistent with fully-implicit `mb_theta = 1`.

**HYPOTHESES.**
- **E3 collapse was LR-driven, not Huber-driven (evidence, high confidence).**
  The only difference between E4 (diverge → +∞) and E3 (collapse → 0) at the same
  loss is the LR (1.3e-4 vs 0.012, ~90×). Therefore the ~90× LR pushed E3 into the
  zero attractor; Huber itself does not cause collapse. This retires the
  "Huber-driven collapse" branch of the E3 read-out.
- **Loss is not a lever for the STIFF-anchor instability, but IS for the benign
  daterange (evidence).** Fixed-horizon anchors diverge under both MSE and Huber
  (loss-independent intrinsic instability). But MSE keeps the benign daterange
  bounded (1.9×) where Huber explodes (286×) — so the loss shape does matter for
  the trained multi-step dynamics even though the 1-step operator is identical.
- **Likely culprit is the Huber δ linear tail, NOT the peak-weight λ
  (interpretation, testable).** `δ = 0.5` in normalized space (σ≈1) puts all
  high-flow residuals in Huber's **linear** regime → Huber ≈ MAE there → a much
  **weaker restoring gradient on large errors** than MSE's quadratic tail. In a
  free rollout the outlet over-prediction is under-corrected and drifts; with
  spatial accumulation (routing sums upstream q) the under-correction is worst
  exactly where errors are largest (the outlet) → 286×. The peak-weight λ is
  **not** the driver: thresholds are **per-node relative** (each node's own
  98th-pct/IQR, [src/preprocess.jl](../src/preprocess.jl#L489)) in normalized
  (area-scaled) space, so downstream is not systematically up-weighted, and the
  weighting is nearly dormant (`w_mean = 1.0005`, `w_max = 2.72`,
  `c_peak = 0.038`). This rebuts the "downstream is higher → up-weighted"
  mechanism specifically, while confirming the broader "Huber params matter".
- **Instability is intrinsic to the rollout operator (evidence + speculation).**
  `amp ≈ 12`, `mb_gain ≈ 22`, fully-implicit `mb_theta = 1`, divergence from
  epoch 2 → the MB kinematic-wave update magnifies q→h errors geometrically over
  the horizon (the amplification metrics are evidence; the exact route is
  speculation).
- **The single benign daterange trajectory hides it (evidence).** Median per-cell
  NSE 0.973 vs pooled KGE −76 / peak_ratio 163 — exactly the "don't trust one
  benign trajectory" caution: most nodes track, a few (high-drainage) explode and
  dominate.

**RECOMMENDATIONS.**
1. **Prefer MSE over Huber until the rollout is stable** — at the same LR, MSE
   keeps the benign daterange bounded (1.9×) while Huber explodes (286×). Do not
   pursue peak-weighted Huber for peak accuracy until stability is solved; if
   Huber is revisited, it is the **δ**, not λ, that needs attention.
2. **(Optional, cheap) δ-vs-λ disentangling sweep** — if the Huber daterange
   regression matters, a 4-cell run at lr = 1.3e-4 isolates the cause:
   `λ=0` (peak-weight OFF) × `δ ∈ {0.5, 2, 5}`, plus one `λ=2, δ=0.5`.
   Prediction (mine): `λ=0, δ=0.5` still worse than MSE and large δ → recovers
   toward MSE (confirms **δ linear tail**); if instead `λ=2` ≫ `λ=0`, the
   **peak-weight** mechanism is vindicated. Read-out = daterange outlet ratio +
   per-node NSE p10, not pooled scalars.
3. **Do NOT lower `mb_theta` — already swept, θ=1 is the most stable.** The
   archived `sava_small_v081_theta_sweep` (θ ∈ {1, 0.95, 0.75, 0.5, 0.25, 0.05,
   0}) shows outlet daterange ratio degrades monotonically as θ drops: θ=1 **1.69**
   (best), θ=0.5 6.9, θ=0.25 **6.5e11** (blowup), θ=0.05 5.1e4, θ=0 67. This
   matches [notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md)
   §4 — the fully-implicit term *is* the rollout governor; lowering θ reintroduces
   a one-step feedback delay (CFL-like growth). So θ=1 is correct; the amplification
   `amp ≈ 12` is not fixable by θ.
4. **Prioritise a training-time error-correction mechanism (top lever)** — the
   remaining, un-swept inference-stability levers target autoregressive **exposure
   bias** (the model never sees its own compounding errors during teacher-forced
   1-step training): (a) **rollout noise injection** (`strategy.noise_scale > 0`,
   currently 0) to teach self-correction; (b) **detached / pushforward rollout
   gradient** (TODO §3); (c) **longer curriculum horizons / more rollout epochs**.
   Read-out: does the outlet/high-area ladder ratio stay O(1) over the full
   daterange, not just the pooled scalar?
5. **Pair with the `q ≥ 0` floor on the propagating state** ([TODO.md](TODO.md)) —
   prevents the E3-style negative-flow degenerate branch under other LRs.
6. **Report a downstream/area-weighted stability metric, not pooled sums** — E4's
   pooled KGE −76 / peak_ratio 163 are dominated by the single outlet while median
   node NSE is 0.973. Track the per-node distribution (median + p10) and an
   outlet/high-area-node rollout score; a high median NSE alone must never be read
   as rollout success.

---

## E3 — peak-weighted Huber λ-sweep — `sava_small_v081_e3_huber_lambda`

**NAME:** `sava_small_v081_e3_huber_lambda` (6-run `box` hparsearch,
`train.strategy.peak_lambda ∈ {0, 0.5, 1, 2, 4, 8}` = hps1…hps6; `loss_type =
"huber"`, `peak_delta = 0.5`, `peak_w_max = 4.0`, full `[1,2,5,8,10]` curriculum,
8×64 / mlp 2, 250 epochs). **LR autotuned (Step 0, 20 inits, seeded):
`lr_start = 0.012`, `grad_clip = 1.0`, `batch_size = 32`.** Config:
[experiments/sava_small_v081_e3_huber_lambda/config.toml](../experiments/sava_small_v081_e3_huber_lambda/config.toml).

**SUMMARY.** All 6 cells trained and all diagnostics wrote (the first fully clean
run of the peak-accuracy tooling after the autotune/parser fixes). Teacher-forced
1-step is accurate (`val_q_1step ≈ 0.0133`, as good as E1). For the first time
ever the **free rollout is "bounded"** — all 32 fixed-horizon anchors finite
(median RMSE ~0.9) across every cell, vs all-Inf in E1/E2. **But this is a
degenerate collapse, not learned stability:** the model drove discharge to ~zero.

**⚠ Read-the-magnitude lesson.** "Bounded" ≠ stable. The daterange rollout shows
`river_q_pred` max ≈ 5 / mean ≈ 1.8 vs **true max 122 / mean 21.7** (a ~12×
under-prediction even for the best cell), with **unphysical negative flows**
(≈ −6). It's bounded only because predicting ~0 is trivially bounded; spatial
`river_q` NSE is negative (−5.3, worse than climatology). Always cross-check
fixed-horizon RMSE against pred-vs-true magnitude + pooled PBIAS/peak_ratio.

**λ table (teacher-forced + rollout eval):**

| hps | λ | pooled KGE | outlet KGE | pooled PBIAS | pooled r | c_peak | rmse_high | best_ep |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | −0.583 | −0.728 | −108% | −0.16 | 0.0 | — | 69 |
| 2 | 0.5 | −0.186 | −0.443 | −74% | 0.10 | 0.788 | 4.00 | 138 |
| 3 | 1 | −0.030 | −0.361 | −61% | 0.21 | 0.005 | 52.7 | 72 |
| **4** | **2** | **+0.356** | −0.230 | **−30%** | **0.49** | **0.964** | 3.43 | 124 |
| 5 | 4 | −0.009 | −0.351 | −60% | 0.23 | 0.911 | 0.93 | 225 |
| 6 | 8 | −0.152 | −0.424 | −71% | 0.12 | 0.386 | 0.92 | 185 |

**IMPROVEMENTS.**
- **Tooling works end-to-end.** Autotune (20 seeded inits, robust median →
  `lr_start = 0.012`), the parser fix (grad_clip propagates), and all Tier-1/2
  diagnostics wrote for every cell. The prior blockers are cleared.
- **λ=2 is a clear non-monotonic optimum** on every whole-hydrograph metric
  (pooled KGE +0.356 vs ≈ −0.02 neighbours — beyond the ~0.15 seed-noise band;
  PBIAS −30% vs −60–108%; r 0.49). Higher λ nudges the mean prediction up
  (0.49 → 1.84), partially counteracting the collapse — which is *why* KGE peaks
  at λ=2. So the λ signal is real but only measures "how much the peak weight
  offsets the collapse", not genuine peak skill.

**DEGRADED / FAILURE MODES.**
- **Rollout collapse-to-zero** (the headline) — degenerate under-prediction,
  negative flows, spatial NSE negative. Mirror image of E1/E2's over-predict →
  +Inf. Peak-weighting-on-1-step cannot fix an autoregressive collapse.
- **`river_h` catastrophically broken** — spatial NSE −5.7e6, predicted head
  exploding while q collapses; q and h have **decoupled** despite the MB coupling
  (worse than E1's −9.9).
- **Degenerate outlet** — outlet-gauge `r` is identical (−0.171) across all six
  cells: a near-flat outlet prediction.

**HYPOTHESES.**
- **Triple confound (evidence).** E3 changed loss (MSE→Huber), LR (1.3e-4→0.012,
  ~90×), **and** the LR was tuned on MSE, not Huber — autotune's range test drops
  `loss_type`/`peak_delta` and silently tunes MSE
  ([TODO.md](TODO.md); [scripts/lr_range_test.jl](../scripts/lr_range_test.jl)
  `make_model_loader`). So even "we re-tuned for Huber" is false. The ~90× LR
  jump + MSE-scaling is the prime suspect for the collapse. **E4 isolates the
  loss** by holding lr at the E1/E2 value.
- **Peak-weighting question is unanswerable while collapsed** — λ=2 only "wins" a
  bad lot; do not read the λ sweep as evidence for/against peak-weighting until
  the model is in a non-degenerate regime.

**RECOMMENDATIONS.**
1. **Do NOT promote any λ setting to the full basin** — λ=2 wins a collapsed
   regime; the result is an artifact.
2. **Run E4 (loss-vs-LR disentangling)** — single Huber run at lr ≈ 1.3e-4.
   Decides whether the collapse is LR- or Huber-driven. **→ DONE (see E4):
   LR-driven.** At lr 1.3e-4 Huber diverges upward like E1/E2; the E3 collapse
   was the ~90× autotuned LR, not the loss.
3. **Fix autotune loss-blindness** (forward `loss_type`/`peak_delta` to the range
   test) and **enforce `q ≥ 0`** at the decoder (negative flow is unphysical) —
   both in [TODO.md](TODO.md).
4. **Inference-stability remains the top lever** — loss/LR moved the attractor
   (+∞ → 0) but neither gives a faithful rollout; prioritise noise / detached
   rollout. *(Correction 2026-09-12: `mb_theta < 1` is NOT a lever — the archived
   `theta_sweep` shows θ=1 is the most stable, lower θ diverges. See E4 rec #3.)*

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
   the detached-rollout trick (TODO item 3). *(Correction 2026-09-12: the
   `mb_theta < 1` suggestion is superseded — the archived `theta_sweep` shows θ=1
   is the MOST stable and lowering θ worsens the rollout, θ=0.25 catastrophically.
   Pursue noise / detached rollout, not θ. See E4 rec #3.)*
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

**RESULT (backfilled 2026-09-12) — θ=1 is the most stable; lower θ diverges.**
7-cell box sweep `model.mb_theta ∈ {1.0, 0.95, 0.75, 0.5, 0.25, 0.05, 0.0}`
(hps1…hps7), MSE, lr 1.3e-4, otherwise the standard 8×64 curriculum. Outlet
daterange over-prediction ratio (pred max / true max 122.4) vs θ:

| θ | 1.0 | 0.95 | 0.75 | 0.5 | 0.25 | 0.05 | 0.0 |
|---|---|---|---|---|---|---|---|
| outlet ratio | **1.69** | 2.45 | 2.04 | 6.92 | **6.5e11** | 5.1e4 | 67 |

Monotone-ish degradation as θ drops (θ=0.25 blows up catastrophically). This
**confirms the design-note prediction**
([notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md)
§4): the fully-implicit term *is* the rollout governor; lowering θ reintroduces a
one-step feedback delay → CFL-like growth. **Conclusion: keep `mb_theta = 1`;
`mb_theta < 1` is NOT an inference-stability lever** (it is the opposite). The
`amp ≈ 12` q→h amplification is intrinsic and must be addressed by training-time
error correction (noise injection / detached rollout), not θ.

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