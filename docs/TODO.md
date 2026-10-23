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

# Mass-balance redesign — B/C hybrid (Muskingum backbone + continuity readout)

**Why (the root-cause fix for the `amp≈12` blow-up).** The current
`ThetaKinematicWave` (Option A) carries the *fragile* discharge `q` as the fed-back
state with the normalisation stiffness `κ(Aᵀ−I) = −21.95` in the same loop, so the
rollout Jacobian has `ρ(J) > 1` from stiff high-flow anchors — the demonstrated
geometric amplification (`peak_ratio ≈ amp^H`). The design thread concluded the
**B/C hybrid** is the recommended target architecture: it is the only closure that
satisfies all four design constraints simultaneously, including the decisive
train-short (≤10) / roll-long (1000s) gap. Full derivation, Jacobian analysis and
side-by-side comparison in
[notes/mass_balance_formulations.md](notes/mass_balance_formulations.md) §4.2 / §6;
see also [DECISIONS.md](DECISIONS.md) and the `amp` lever in
[PROJECT.md](PROJECT.md) Known limitations.

**The construction (§4.2).** Carry `h` (≡ storage `S`) as the state; advance it
with a **Muskingum–Cunge backbone `P_MC`** (Option B) giving a *structural*
contraction `ρ(∂_h F) = C₃ < 1` (horizon-independent); add the decoder's freedom as
a **bounded residual in the source** (cannot move the spectrum); and **diagnose `q`
by the exact continuity readout** `q = (I−Aᵀ)⁻¹(I_lat − dS/dt)` (Option C) so the
fragile discharge is *never a state and never fed back*:

    S^{t+1} = P_MC·S^t + P_MC·source(z^t, r)         # contracting backbone + bounded residual
    q^{t+1} = (I−Aᵀ)⁻¹·(I_lat − dS/dt)                # exact readout, out of the loop

Both operators are static SpMVs → precompute once (`A` is static, constraint 3).

- [x] **New formulation struct** implementing the `MassBalanceFormulation`
      interface in [src/massbalance.jl](../src/massbalance.jl) (e.g.
      `MuskingumContinuityHybrid <: MassBalanceFormulation`, alongside
      `ThetaKinematicWave`/`NoMassBalance`). Must provide every interface method the
      abstract type lists: `decoder_outputs`, `extra_decoder_features` /
      `extra_decoder_dim`, `floor_state`, `apply_formulation`, `has_mass_balance`,
      `mb_h_loss_weight`, `precompute_batched`, `Flux.gpu`/`Flux.cpu`. The decoder
      now predicts a **bounded source residual** (not `Δq`), so `decoder_outputs`
      and the decoder-feature set change accordingly.
- [x] **Muskingum backbone `P_MC`.** Build the static three-coefficient operator
      from reference `(K, x)` so the coefficients are non-negative (M-matrix ⇒
      `C₃ < 1`, the contraction guarantee). Validate `C₂ ≥ 0` (no negative-
      coefficient oscillation) at build time and **abort** if violated, mirroring
      how `build_massbalance` sets up the routing operator today. Precompute as a
      sparse operator in the same class as the current `A_routing`.
- [x] **Continuity readout `(I−Aᵀ)⁻¹`.** Precompute the accumulation inverse once
      (same sparsity/precompute path as the current adjoint in
      [src/sparse_ops.jl](../src/sparse_ops.jl)); apply as one SpMV each step to
      diagnose `q` from `I_lat − dS/dt`. Keep a positivity transform (softplus) on
      `h` outside the solve. `q` is diagnostic-only → it must **not** be fed back
      into the decoder by default.
- [x] **Open design decisions (resolve before/within implementation, from §5).**
      (a) Is `q` a **pure readout** or a **defensively-regularised input feature**
      (`C+q-input`, §3.3 — small/stop-gradient weights so `∂_q F ≈ 0`)? Default to
      pure readout. (b) Does "increment beats absolute" (measured under A) still
      hold with `h` as the free state? §3.3/§6 warn **increment-`h` breaks
      constraint 4** (undamped integrator drift) — so predict the source
      conservatively via `P_MC`, not a raw `h` increment. Record the choice in
      [DECISIONS.md](DECISIONS.md).
- [x] **Config toggle (reuses the existing mechanism).** The formulation is
      already selected by the `mb_formulation` string key
      ([src/gnn.jl](../src/gnn.jl) `ModelSettings.mb_formulation`,
      `mb_formulation_kind`, dispatched via `_build_mb_formulation(::Val{…})` in
      [src/run.jl](../src/run.jl)). Add the new name to the `MB_FORMULATIONS` tuple,
      add the `Val{:muskingum_continuity}` (or chosen symbol) dispatch method, and
      keep the default `"theta_kinematic_wave"` so existing configs are unchanged.
      Update `MB_FORMULATIONS` error text + templates.
- [x] **Wire all formulation-dispatched sites.** Add methods for the new type to:
      `_loss_components` / `_mb_amplification` ([src/strategy.jl](../src/strategy.jl)),
      `_rollout_mb_diagnostics` ([src/rollout.jl](../src/rollout.jl)),
      `layer_mb_analysis` ([src/analysis.jl](../src/analysis.jl)), and the
      `mb_h_loss_weight` path. The `amplification.png`/volume-budget diagnostics
      should report the new backbone's `C₃` contraction instead of A's −21.95
      stiffness.
- [x] **Expose to the hpar search.** `model.mb_formulation` is a plain string key,
      so the box search can sweep it today via `_set!`
      ([src/hparsearch.jl](../src/hparsearch.jl)) once the new value is accepted —
      add `"model.mb_formulation"` with `["theta_kinematic_wave",
      "muskingum_continuity"]` to
      [experiments/template_hparsearch.toml](../experiments/template_hparsearch.toml)
      as the A/B example. Confirm a search cell with the new value preprocesses,
      trains and evaluates end-to-end.
- **Implemented (all items except the validation A/B).**
  `MuskingumContinuityHybrid` / `build_muskingum_continuity` in
  [src/massbalance.jl](../src/massbalance.jl), selected by
  `mb_formulation = "muskingum_continuity"` (default unchanged), residual bound
  `mb_residual_bound` (default 0.5). Deviations from the text above, all recorded in
  [DECISIONS.md](DECISIONS.md): the backbone is an implicit-Euler *linear-reservoir cascade*
  (Muskingum with `x = 0`), not the three-coefficient MC form (daily steps give Courant
  numbers â‰« 1, so `Câ‚ƒ < 0` for physical `K` and the "abort on a negative coefficient" check
  would always fire; the reservoir backbone has spectrum `1/(1+dt/K) < 1` for any `K > 0`
  and `P â‰¥ 0`); the residual guarantee is bounded-input/bounded-state, not "cannot move the
  spectrum"; the encoder sees a zeroed `q` row (`encoder_state`) so `q` is out of the loop.
  The hybrid has no rollout/volume-budget layer record yet (`layer_mb_analysis` writes
  `metrics/mb_backbone_contraction_by_node.csv`); a volume budget for it is a follow-up.
  Tests: [test/test_mb_hybrid.jl](../test/test_mb_hybrid.jl) (44) and an end-to-end
  train/infer/analyze case in [test/test_pipeline.jl](../test/test_pipeline.jl).
- [ ] **Open design choices to revisit (not defects — the stability guarantee holds
      either way).** Two simplifications vs the §4.2 construction, worth a deliberate
      decision before/after the A/B validation:
      (a) **Level-pool backbone (Muskingum `x = 0`).** `P` is an implicit-Euler
      *linear-reservoir cascade* with a static per-node `K = μ_h/μ_q`, so it has the
      degenerate diffusive wedge: pure attenuation, no kinematic translation/sharpening.
      The `C₃ < 1` contraction is intact, but hydrograph-shape fidelity (travel time,
      peak attenuation) is the crudest MC case. Revisit if the A/B shows timing/peak
      bias that a true three-coefficient `(K, x)` operator — or a flow-dependent `K` —
      would fix. (Daily Courant ≫ 1 is why the physical three-coefficient form was
      dropped; a revisit would need a Courant-robust coefficient scheme.)
      (b) **Residual added *after* the backbone, not routed through it.** The code uses
      `h_pre = h_bb + residual_bound·tanh(Δ)`, whereas §4.2 wrote the residual inside
      the source, `S^{t+1} = P_MC·(S^t + source)`. Bounded-additive on a contraction is
      still BIBO-stable (why the guarantee survives), but the residual is not smoothed/
      routed by `P` and acts one step more directly on `h`. Decide whether to route it
      through `P` for a smoother, physically-sourced correction. Record the resolution
      of both in [DECISIONS.md](DECISIONS.md).
- [ ] **Validation (win conditions).** A/B the hybrid vs `ThetaKinematicWave` on
      `sava_small_v081`: (i) `amp` no longer geometric and fixed-horizon RMSE finite
      from stiff high-flow anchors; (ii) stable past the training horizon (≥1906
      steps, matching the current MB's one virtue) — the structural `C₃ < 1` claim;
      (iii) mass conservation still exact (volume-budget PBIAS/drift); (iv) `q`
      confirmed out of the loop (no `q`-row/column in the measured rollout
      Jacobian). Then repeat on full `sava_v081` where A diverges.
- Touch: `src/massbalance.jl` (new struct + interface), `src/sparse_ops.jl` /
  precompute (both static operators), `src/run.jl` (`_build_mb_formulation` Val
  dispatch), `src/gnn.jl` (`MB_FORMULATIONS`, decoder out-dim), `src/strategy.jl` +
  `src/rollout.jl` + `src/analysis.jl` (dispatched diagnostics), config templates +
  `scripts`/`experiments`; tests in `test/test_gnn.jl` (interface + conservation +
  contraction) and `test/test_training.jl` (a short hybrid run).
- **Note:** A and C are mutually-exclusive closures; the hybrid is additive (B
  backbone + C readout) and does **not** share state with `ThetaKinematicWave`, so
  it lands as a *parallel* formulation behind the toggle — no change to the legacy
  path, so existing checkpoints and A/B comparisons stay valid.

---

# Input feature normalization — selectable nonlinear scaling

**Why.** Physical river features span many orders of magnitude and are
heavy-tailed (discharge, upstream area, inwater), so a plain per-variable z-score
leaves the high-flow tail dominating the feature scale and compresses the dynamic
range the encoder sees at low/median flow. Static river geometry already gets a
`log1p` pre-standardization (`STATIC_TRANSFORMS`: `river_length/slope/width/depth`
via `_log1p_nonnegative!`) and `wflow_uparea` gets `_log_upstream_area_feature`,
but the **time-varying physical inputs** (`river_q`, `river_h`, `river_inwater`)
currently go straight to z-score (per-node for the `PER_NODE_NORM_VARS` set) with
no nonlinear squashing. Goal: let each physical feature pick a nonlinear transform
(e.g. `log1p`, signed-log, `asinh`) applied **before** z-score, selectable from
config and sweepable in the hpar search.

**Current state (grounding, [src/preprocess.jl](../src/preprocess.jl)).**
Normalization is hardcoded in two places: `STATIC_TRANSFORMS` (per-node static
transform before `standardize!`) and the time-series path, where
`PER_NODE_NORM_VARS` get `standardize_per_node!` and everything else gets plain
`standardize!`. `VAR_SCALERS` additionally applies the area/storage postscale and
records the per-node inverse multiplier in `postscale`. De-normalization
(`evaluate_trajectory` in [src/rollout.jl](../src/rollout.jl), the MB-residual path
in [src/analysis.jl](../src/analysis.jl)) undoes **z-score then `postscale`** — it
knows nothing about a nonlinear transform yet.

- [x] **Normalization-scheme abstraction.** Replace the hardcoded transform Dicts
      with a small selectable scheme per variable: at minimum `:zscore` (today's
      behaviour, the default), `:log1p_zscore`, `:signed_log_zscore`,
      `:asinh_zscore`. Each scheme is a `(forward, inverse)` pair so it is exactly
      invertible; z-score stays the outer affine step. Keep the per-node vs global
      z-score choice (`PER_NODE_NORM_VARS`) orthogonal to the nonlinear transform.
- [x] **Wire the inverse into every de-normalization path.** *(Resolved by design: encoder-only, no inverse path needed; see the Implemented note.)* The transform id must
      be persisted in `norm_stats` (→ `norm_stats.toml`) so `evaluate_trajectory`
      and the analysis/MB-residual code invert `nonlinear ∘ zscore ∘ postscale`
      symmetrically. Audit all readers of `norm_stats`/`postscale` (rollout.jl,
      analysis.jl, rollout fixed-horizon eval) — a missing inverse silently
      corrupts physical-unit output.
- [x] **Mass-balance correctness gate (critical).** The `ThetaKinematicWave` layer
      and the loss evaluate physics in **physical units** and the per-node z-score
      is deliberately kept **affine** to preserve routing additivity (see the
      `PER_NODE_NORM_VARS` comment in preprocess.jl and
      [src/massbalance.jl](../src/massbalance.jl)). A nonlinear transform on the
      **state** variables (`river_q`/`river_h`) would break that additivity, so
      scope the nonlinear option to **encoder input features** (forcing +
      static + the state *as fed to the encoder*) while the decoder/MB/loss keep
      operating on physical (affine-recoverable) `q`/`h`. Decide and document
      whether `river_inwater` (a forcing, not a looped state) may use a nonlinear
      transform freely. Add a test that the MB physical budget is unchanged when
      only input-feature normalization changes.
- [x] **Config switch (per variable).** Expose the scheme in the run config
      (data/model settings TOML), e.g. a `[normalization]` table mapping variable
      name → scheme, defaulting every variable to `:zscore` so existing configs are
      unchanged. Plumb through `DataSettings`/`ModelSettings`
      ([src/run.jl](../src/run.jl), [src/gnn.jl](../src/gnn.jl)) into
      `build_wflow_graph`'s kwargs (mirror how `sigma_floor_*` /
      `include_log_upstream_area` already flow via `preprocess_run`,
      [src/pipeline.jl](../src/pipeline.jl#L198)). Persist the choice so inference
      reuses it.
- [x] **Expose to the hpar search.** The box search already applies arbitrary
      dot-path overrides via `_set!` ([src/hparsearch.jl](../src/hparsearch.jl)), so
      a key like `"normalization.river_inwater"` is sweepable **once it exists in
      the parsed config**. Add the key(s) to the search-space and confirm `_set!`
      reaches the nested table (it indexes existing sub-dicts — pre-create the
      `[normalization]` table or teach `_set!` to create missing intermediate
      nodes). Add an example to
      [experiments/template_hparsearch.toml](../experiments/template_hparsearch.toml).
- **Implemented (all items except the validation A/B).** The nonlinear scaling is an
  **encoder-input-only** transform applied *inside the model*, not in the data
  pipeline, so the stored (affine) state/forcing, `norm_stats` `(mean, std)`,
  `postscale`, the mass-balance layer, the loss, `evaluate_trajectory` and the
  analysis code are untouched and no inverse is needed in any de-normalization path
  (the "wire the inverse" item is satisfied by construction; the MB budget is
  bit-unchanged, tested). Schemes (`NORMALIZATION_SCHEMES`): `:zscore` (default),
  `:log1p_zscore`, `:signed_log_zscore`, `:asinh_zscore`; `nonlinear_forward`
  / `nonlinear_inverse` are an exact pair (inverse used for tests). The scale `s`
  is fitted automatically (median non-zero `|x|` of the training window, per node for
  `PER_NODE_NORM_VARS`) because area-normalised `river_inwater` is ~1e-8, where a
  plain `log1p`/`asinh` is the identity; `f(x/s)` is re-standardised with
  training-window `nl_mean`/`nl_std`. `build_wflow_graph(...; normalization)`
  stores `scheme, scale, nl_mean, nl_std` in the stats of non-zscore variables only
  (also written to `norm_stats.toml`). The model prepends a non-trainable
  `EncoderInputTransform` to the encoder only when a scheme is set (default model and
  old checkpoints unchanged). Config: `[model.normalization]` (`variable = scheme`),
  validated, persisted in `model_settings.toml`; hpar key
  `model.normalization.river_inwater` works (`_set!` now creates missing tables).
  Scope: time-varying state/forcing variables; static variables keep
  `STATIC_TRANSFORMS`. Tests: [test/test_normalization.jl](../test/test_normalization.jl).
- [ ] **Validation A/B (open).** Run 
iver_inwater :zscore vs :log1p_zscore /
      :asinh_zscore on sava_small_v081 (config key above) and compare peak/low-flow
      skill.
- Touch: `src/preprocess.jl` (scheme abstraction, forward+inverse,
  `build_wflow_graph` kwargs), `src/rollout.jl` + `src/analysis.jl` (inverse in
  de-normalization), `src/run.jl` / `src/gnn.jl` (settings fields),
  `src/pipeline.jl` (`preprocess_run` plumbing), `src/hparsearch.jl` (`_set!`
  nested-create if needed), config templates + `norm_stats.toml` schema; tests in
  `test/test_preprocess.jl` (round-trip invertibility) + `test/test_gnn.jl` (MB
  budget unchanged).

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

## 3f. Outlet timeseries at the attribution outlets (in the analyze stage)

**Why.** The longitudinal `downstream_timeseries.png` nodes come from
`_upstream_percentile_nodes` (an upstream-area percentile ladder in
[src/plot.jl](../src/plot.jl)), which is chosen on a *different* criterion than the
subcatchment error-attribution outlets (confluence/gauge-defined in
[src/attribution.jl](../src/attribution.jl)). Only the single most-downstream node
(largest upstream area = attribution tree root) is guaranteed to coincide; the rest
differ in count and location. Plotting pred-vs-true discharge at **exactly** the
attribution outlets (plus the whole-basin outlet) lets the timeseries be read
cell-for-cell against the attribution eval.

- [x] **Explicit-node plotting.** Give `plot_downstream_timeseries` a `nodes`
      kwarg (graph-node indices) that bypasses `_upstream_percentile_nodes` and
      plots the given nodes in order. `upstream_area` then only labels. Existing
      callers (percentile ladder) keep the current default when `nodes === nothing`.
- [x] **Cell→node mapping.** Attribution stores outlets as **linear (column-major)
      cell indices** (`(col-1)*nrows+row`), the plotter works in **graph nodes**.
      Add a small inverse LUT (e.g. near `gather_nodes` in
      [src/postprocess.jl](../src/postprocess.jl) or in `src/analysis.jl`):
      `nodes_for_cells(grid, cells)` dropping cells with no active node.
- [x] **Cheap outlet derivation.** Add a `rank = false` kwarg to
      `subcatch_error_attribution` so the tree/outlets/decomposition are returned
      without the (5000-shuffle) permutation ranking — the analyze stage only needs
      the outlet set, not the attribute ranking.
- [x] **Wire into `analyze_run` by default.** Derive the retained attribution
      outlets + the largest-area basin root (`parent_id == 0`, listed first,
      de-duplicated), map to nodes, and write `<split>_outlet_timeseries*.png`/`.csv`.
      Make it **best-effort**: wrap in try/catch and skip with a notice when
      `river_q` is absent, `wflow_uparea` is missing (network route), or the rollout
      diverged — the analyze stage must not crash on it. Document the new output in
      the `analyze_run` docstring and [PROJECT.md](PROJECT.md) Evaluation Metrics.
- [x] **Open choices for the engineer.** (a) Expose `gauge_var`/`min_area_km2` from
      run config so the analyze-stage outlets can use Route A / a non-default size
      filter (default otherwise: network route, 100 km²). (b) Optionally cap to the
      main-stem outlets (`mainstem(tree)`) when the retained set is large (16 on
      `sava_small_v081`) to limit the number of figures. (c) Also emit for the
      optional date-range split.
- Touch: `src/plot.jl` (`nodes` kwarg), `src/attribution.jl` (`rank` kwarg),
  `src/analysis.jl` (`nodes_for_cells` + outlet-timeseries call in `analyze_run`);
  docstring + `docs/PROJECT.md`. No model-source change.
- **Implemented.** (a) done as `analyze_run` kwargs `outlet_gauge_var` / `outlet_min_area_km2` (default network route, 100 km^2), not yet in the run config / TOML; (b) not done (all retained outlets are plotted, whole-basin outlet first); (c) done (`_daterange` suffix). `nodes_for_cells` lives in `src/postprocess.jl`; outlet selection is `attribution_outlet_cells`; the diverged-rollout check is `rollout_diverged`. The whole-basin node (largest upstream area) is prepended to the attribution outlets, and files are un-prefixed like the other analyze-stage outputs (`outlet_timeseries*.png`).

## 3g. Subcatchment error-attribution plots are not produced by the pipeline / hpar search

**Symptom.** A hyperparameter search produces **no** subcatchment error-attribution
figures for its cells. Only the `outlet_timeseries*.png`/`.csv` (3f) appear in each
cell's `plots/`/`metrics/`; the actual attribution products — `<split>_subcatch_error_map.png`,
`<split>_subcatch_mainstem.png`, `<split>_subcatch_attributes.png` and
`<split>_subcatch_error_attribution.csv` / `<split>_subcatch_attribute_ranking.csv` —
are missing.

**Root cause.** The analyze stage ([`analyze_run`](../src/analysis.jl)) only calls
`subcatch_error_attribution(...; rank = false)` **internally**, to pick the outlet
*cells* for the 3f outlet timeseries. The attribution maps/ranking themselves are
written only by the standalone `subcatch_error_attribution_run(run_dir)` /
[scripts/subcatch_error_attribution.jl](../scripts/subcatch_error_attribution.jl),
which must be invoked **per run directory** post-hoc. The 4-stage pipeline
(`preprocess → train → infer → analyze`) — and therefore `hpar_search`, which runs
`run_wflow_gnn` per cell — never invokes it, so a search of N cells yields N run dirs
with no attribution plots. (A single run has the same gap; it only *looks* generated
if the user ran the CLI afterwards.)

- [x] **Pick the fix.** Either (a) call `subcatch_error_attribution_run`-equivalent
      plotting from inside `analyze_run` (reuse the already-computed `res` from the
      3f outlet-cell step so the 5000-shuffle ranking is the only extra cost; gate the
      ranking behind a flag since it is expensive and degenerate on small basins),
      **best-effort** (try/catch + `@info` skip, mirroring `_analyze_outlet_timeseries`),
      so every pipeline/hpar run emits the plots; or (b) add a post-search hook in
      `hpar_search` that runs `subcatch_error_attribution_run` over each cell's
      `run_dir` after training. Prefer (a): one attribution pass, consistent with how
      the outlet timeseries are already wired, and it also covers single runs.
- [x] **Avoid double work / divergence guard.** Share the single
      `subcatch_error_attribution` result between the 3f outlet-timeseries derivation
      and the new plotting; keep the same `rollout_diverged` early-out so diverged
      cells skip cleanly instead of plotting garbage.
- [x] **Expose the toggle** (and the expensive ranking on/off) in the run config so a
      search can turn it off for speed, defaulting to on for single runs. Document the
      new analyze-stage outputs in the `analyze_run` docstring and
      [PROJECT.md](PROJECT.md) Evaluation Metrics.
- Touch: `src/analysis.jl` (`analyze_run` wiring + `rank`/plot flag), possibly
  `src/hparsearch.jl` (if route (b)), config templates; no model-source change.
- **Implemented (route a).** `analyze_run` now computes one `subcatch_error_attribution`
  per rollout (`_analyze_subcatchments` in [src/analysis.jl](../src/analysis.jl)), uses it
  for the 3f outlet timeseries and, when `[eval] subcatch_attribution` is on (default),
  writes `val_subcatch_*` (and `val_daterange_subcatch_*`) via the new
  `write_subcatch_error_attribution` ([src/attribution.jl](../src/attribution.jl), shared with
  `subcatch_error_attribution_run`). `[eval] subcatch_rank` (default on) gates the
  permutation ranking, i.e. the attribute figure and ranking CSV. Both are new
  `TrainSettings` fields (TOML round trip, `[eval]` or `[train]` key, sweepable in an
  hpar search as `eval.subcatch_attribution`). Best effort with the same
  `rollout_diverged` early-out; each write is wrapped so a failure only skips that product.

## 3h. MB diagnostics assume the old q/h roles — stale under the Muskingum-continuity hybrid

**Why.** Several diagnostics were designed under the `ThetaKinematicWave` (Option A)
assumption that **`q` is the fed-back state** and **`h` the stiff derived readout**.
The hybrid (`MuskingumContinuityHybrid`) **switches those roles**: `h`/storage is the
carried state and `q` is a diagnostic continuity readout out of the loop. Reported
side-by-side in one hpar results CSV, the affected columns are either `NaN`/empty for
the hybrid arm or silently mean something different, so cross-arm (A/B) comparison is
misleading. This surfaced on the E15 A/B search
([experiments/sava_small_v081_e15_newfeatures_ab](../experiments/sava_small_v081_e15_newfeatures_ab)).

**Design (chosen; full rationale in [DECISIONS.md](DECISIONS.md) → PROPOSED
23-10-2026).** Separate the **axis of comparison** (scientific skill — must be
identical across arms) from the **mechanism diagnostic** (stability — must be
meaningful per mechanism). Scientific metrics use a single *works-for-both* path;
stability uses one measured *works-for-both* headline plus *auto-switched* analytic
companions. Scope is reporting/labelling + the results-CSV schema, **not** the hybrid
physics.

**Implemented (2026-10-23).** `rollout_contraction` ([src/strategy.jl](../src/strategy.jl))
is evaluated every epoch on all validation batches for every arm (`losses.val_contraction`).
`metrics.toml` gains `[formulation]` (kind, carried state / readout, effective
`h_loss_weight`) and `[stability]` (`rollout_contraction`, `rollout_contraction_margin`;
theta: `analytic_gain`, `q_to_h_amp`; hybrid: `backbone_rho_median`, `backbone_rho_max`);
`[mass_balance_residual]` gains layer-free `volume_pbias` / `volume_drift`
(`field_volume_bias`); the hpar CSV / summary replace `final_amp` with
`rollout_contraction` plus presence-based companions. ρ̂ is a one-step finite-difference
gain, see [DECISIONS.md](DECISIONS.md) (implementation notes). Skill-metric audit: all of
`river_q_performance_metrics`, the spatial medians and the fixed-horizon RMSE are computed
from the gridded fields / physical `q`, none is gated on the formulation. **Still open:**
the `increment` vs `absolute` default per formulation is not revisited, and
`plot_amplification` still labels the hybrid backbone contraction as "gain".

### Scientific-skill metrics — one formulation-agnostic path (no switching)

- [x] **Keep the field-based skill metrics arm-independent.** `q` is physical
      discharge in both arms, so `river_q_performance_metrics` (KGE, r/α/β, PBIAS,
      peak_ratio, FHV, per-node NSE), `river_h` RMSE/NSE/spatial medians, and the
      fixed-horizon RMSE on `q` are already comparable — just audit that none of them
      route through a layer-internal readout or get gated on the formulation.
- [x] **Unify conservation across arms.** Retire the theta-only *layer-internal*
      volume budget as the canonical conservation number and promote the **layer-free**
      `mb_residual_diagnostics` / `_analyze_mb_residual` (recomputed from the `q` & `h`
      fields; already "applies to any model") to the single conservation metric for
      both arms, writing the same `volume_pbias`/`drift` columns. The hybrid's exact
      continuity readout should drive its field residual ≈ 0; theta exposes its
      structural mismatch.

### Stability / amplification metrics — measured headline + auto-switched companions

- [x] **Add a works-for-both headline `rollout_contraction` ρ̂.** `ρ̂ =
      median_t ‖Δs_{t+1}‖/‖Δs_t‖` under teacher forcing, where `s` is **the
      formulation's own state row(s)** — dispatch only selects which row is the state
      (`q`-row for theta, `h`-row for the hybrid, via the existing `encoder_state`/
      state layout). Single orientation: **ρ̂ < 1 ⇒ contractive**, margin = 1 − ρ̂.
      This **replaces `final_amp`** as the sortable stability column present for every
      run, and fixes the current NaN-for-hybrid `final_amp`.
- [x] **Keep the analytic companions, namespaced + presence-based (drill-down only).**
      theta: `stability.analytic_gain = median(θ·dt·σ_q/σ_h)` (today's `gain`) and the
      realised `stability.q_to_h_amp` (today's `amp`); hybrid:
      `stability.backbone_rho = ρ(P)` (max + median backbone contraction, exact /
      horizon-independent) from `backbone_contraction(mb)`. A companion column **exists
      only for the arm it means something for** (not NaN for the other).
- [x] **`fixed_horizon` framing.** The fixed-horizon RMSE / `peak_ratio_frac_gt2` on
      `q` stay meaningful (physical), but any pass/fail gate or plot annotation keyed on
      the theta `amp` blow-up must be conditioned on the formulation (use ρ̂/margin, not
      `amp`, for the cross-arm gate).

### Results-CSV schema

- [x] **Headline always present, companions skipped-not-NaN.** `_read_run_scalar_metrics`
      / `_aggregate_repetition_rows` ([src/hparsearch.jl](../src/hparsearch.jl)) emit
      `rollout_contraction` for every run (the one stability column the summary
      averages) and the companions under their own keys, **skipping keys absent for an
      arm** (explicit not-applicable) so the aggregation never averages a NaN or
      compares incomparable quantities. Scientific columns stay identical-definition
      across arms.

### Loss weighting & labels (A/B fairness; not reported metrics)

- [x] **Keep the scientific loss identical across arms** (physical `q` and `h` errors)
      for a fair A/B; let each formulation declare its internal weighting via the
      existing `mb_h_loss_weight` dispatch (hybrid returns `nothing` = plain strategy
      weight; theta keeps the q→h-stiffness compensation that justified
      `h_loss_scale = "increment"` — a rationale that no longer holds when `h` is the
      primary state). Record the **effective** weight in `metrics.toml` so the
      comparison is auditable; revisit whether `increment` vs `absolute` is the right
      default per formulation.
- [x] **Annotate `loss_components`.** Keep the physical column names (`q_mse`,`h_mse`)
      but tag which row is the carried **state** vs the diagnostic **readout** for the
      active formulation, so logs/plots don't flip meaning silently between arms.
- **Scope:** reporting/labelling + the results-CSV schema, not the hybrid physics
  (which is correct). Record the chosen reporting convention in
  [DECISIONS.md](DECISIONS.md) (done: IMPLEMENTED 23-10-2026).
- Touch: `src/strategy.jl` (`_mb_amplification` → `rollout_contraction` headline +
  companions), `src/analysis.jl` / `src/postprocess.jl` (`layer_mb_analysis`,
  unified conservation via the layer-free residual, metrics.toml fields),
  `src/hparsearch.jl` (`_read_run_scalar_metrics` + summary aggregation: headline +
  skip-not-NaN companions), plotting/labels; no change to the MB layer itself.

---

# Resumable hpar search & training runs (opt-in CLI flag)

**Why.** An interrupted run restarts from scratch today. `hpar_search`
([src/hparsearch.jl](../src/hparsearch.jl)) unconditionally loops over every
combination and calls `run_wflow_gnn` ([src/run.jl](../src/run.jl)), which always runs
all four stages (`preprocess → train → infer → analyze`) from the beginning; there is
no completion check, progress file, or skip-if-exists logic. So relaunching an
interrupted search **reruns every cell** (overwriting the already-finished ones, since
the combo→folder map `<run_name>_hps<NN>[_s<seed>]` is deterministic) and, because the
aggregate `*_hps_results.csv` / `*_hps_summary.*` are only written at the very end, an
interrupted search leaves **no** results CSV. A single `scripts/train.jl` run has the
same all-or-nothing behaviour across its four stages.

**Scope decision (from discussion).** Resume at **cell / stage granularity** — *no*
checkpointing or warm-start of an individual training run; a cell that was mid-train is
simply **rerun from scratch**. Resumability is **off by default** and enabled by an
explicit CLI flag on both the hpar-search and train scripts (so default behaviour and
existing reproducibility are unchanged).

- [ ] **Define "cell complete".** A run directory counts as done iff its terminal
      artifact exists and is self-consistent — propose `metrics/metrics.toml` present
      (written last by `analyze_run` when `metrics_toml = true`) **and** the expected
      `output/<split>_{pred,true}.nc` present. A partial/aborted cell (no/stale
      `metrics.toml`) is treated as incomplete and **rerun from scratch** (its dir is
      overwritten), per the scope decision.
- [ ] **Hpar-search resume.** When the flag is set, before calling `run_wflow_gnn` for
      a cell, check the deterministic run dir (`joinpath(ds.runs_dir, ds.run_name)`): if
      complete, `@info` skip and `continue`; else run it. Keep the deterministic
      combo/seed → folder naming so the check is a pure function of the config (already
      the case).
- [ ] **Reconstruct the aggregate on resume.** The results CSV must still be produced
      even though skipped cells weren't run this invocation: read each finished cell's
      `metrics/metrics.toml` via the existing `_read_run_scalar_metrics` to rebuild its
      `rows` entry (metrics come from disk; `n_params`/durations that currently come
      from the in-memory `metrics` return need a disk fallback — persist them into
      `metrics.toml` if not already there, or mark them `NaN`/missing for skipped
      cells). Then `_write_hps_csv` / `_aggregate_repetition_rows` as today over the
      full set.
- [ ] **Singleton `run_wflow_gnn` / train resume.** Add the same opt-in skip to the
      four-stage `AllStages` fold so a resumed single run skips stages whose artifacts
      already exist (prep → `prep/` + datasets; train → `model/model.jld2`; infer →
      `output/<split>_{pred,true}.nc`; analyze → `metrics/metrics.toml`). A stage with
      a missing/stale artifact and **all** later stages are (re)run. This also makes the
      existing per-stage `scripts/pipeline.jl` entry points composable for manual resume.
      Still **no** intra-train checkpoint resume — `train` reruns fully when selected.
- [ ] **CLI flag on both scripts.** Add an opt-in flag (e.g. `--resume`) to
      [scripts/hparsearch.jl](../scripts/hparsearch.jl) and
      [scripts/train.jl](../scripts/train.jl) (and optionally
      [scripts/pipeline.jl](../scripts/pipeline.jl)). Both scripts currently parse
      **positional args only** (`ARGS` length checks), so add minimal flag parsing that
      keeps the existing positional usage working when the flag is absent; thread a
      `resume::Bool` kwarg through `hpar_search` / `run_wflow_gnn_from_toml` /
      `run_wflow_gnn`. Default `false` (today's behaviour). Update each script's usage
      string / header comment.
- **Caveats to document.** (a) Resume trusts on-disk artifacts — a config edited
      between runs is **not** detected, so a changed cell with a stale complete dir is
      skipped; note this (optionally hash the effective per-cell config into
      `metrics.toml` for a mismatch warning, out of scope for v1). (b) Skipped cells
      contribute their **persisted** metrics to the CSV, so columns sourced only from
      the in-memory return must gain a disk fallback (above).
- Touch: `src/hparsearch.jl` (`hpar_search` skip + CSV reconstruction),
  `src/run.jl` (`run_wflow_gnn` / `AllStages` stage-skip + `resume` kwarg),
  `src/pipeline.jl` (artifact-existence predicates, if not reused from the stage
  readers), `scripts/hparsearch.jl` + `scripts/train.jl` (+ optionally
  `scripts/pipeline.jl`) flag parsing; a test that a resumed search skips a
  pre-populated cell dir and still writes the full results CSV.

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