# Mass Balance — Rollout Stability, Soft/Hybrid Variants & the θ-Scheme

Forward-looking design notes exploring **how** the hard mass-balance constraint
stabilises the autoregressive rollout, and what happens if we relax it. Records
the reasoning behind soft and hybrid variants, the implicit/explicit/trapezoidal
(θ-method) family of discretisations, and the tuning of θ.

This is a companion to [docs/mass_balance_notes.md](mass_balance_notes.md) (the
*implemented* hard-constraint design, gradients and loss weighting) and
[docs/training_tuning_notes.md](training_tuning_notes.md) (LR curriculum,
`h_loss_scale`). Most of this is a design/analysis record; the **θ-method (§4–5)
is now implemented** (`MassBalanceLayer.θ` / `[model].mb_theta`, default `1.0`),
while the soft/hybrid governors (§2–3) and the Q rating prior (§7) remain
design proposals.

Source touchpoints: [src/gnn.jl](../src/gnn.jl) (`MassBalanceLayer`,
`mb_diagnostics`, `WflowGNN` forward, `enforce_mass_balance`),
[src/run.jl](../src/run.jl) (`build_gnn_model`, weight derivation),
[src/strategy.jl](../src/strategy.jl) (loss). Empirical basis:
`experiments/sava_small_v081_mb_sweep` (the MB on/off A/B) and the regime
analysis in [scripts/regime_analysis.jl](../scripts/regime_analysis.jl) +
[scripts/network_diameter.jl](../scripts/network_diameter.jl) (§6), run on both
`models/sava_small_v081` and the full `models/sava_v081`.

---

## 1. Empirical motivation: the constraint is a rollout governor

An A/B sweep varied **only** `enforce_mass_balance` (`true` vs `false`);
everything else identical (8 layers, hidden 64, mlp 2, batch 8, 250 epochs,
curriculum `steps = [1,2,5,8,10]`, `lr_start ≈ 1.35e-4`, `grad_clip = 1.0`,
`h_loss_scale = "increment"`). The `false` arm predicts `river_q` and `river_h`
as two independent decoder outputs (`out_dim = 2`).

| Metric | MB on (hps1) | MB off (hps2) |
|---|---|---|
| final train / val loss | 0.0285 / 0.0242 | 0.0441 / 0.0438 |
| date-range rollout Q RMSE | **27.5** | **≈ 7.5 × 10⁵** |
| date-range rollout H RMSE | **6.55** | **≈ 1.1 × 10⁴** |
| full val rollout (1906 steps) | finite throughout | **Float32 overflow → `Inf`** from step ≈ 1571 |

Teacher-forced training looked only ~1.8× worse without MB, but the **free-running
rollout diverged exponentially** and overflowed Float32. Diagnosis: with no
algebraic anchor, `q` and `h` are two free outputs fed back each step; a small
consistent positive bias compounds multiplicatively (positive feedback), an
unbounded runaway. The hard MB prevents this by (a) forcing `h` onto the
conservation manifold every step and (b) coupling `h` back to the routing
topology so local errors dissipate through the flux budget instead of
accumulating. **Key takeaway: the stabilisation is an inference-time property, not
a training-time one.**

---

## 2. Soft mass balance (loss penalty) — can it restrain the blowup?

A soft variant adds a physics-residual penalty instead of the hard decoder:

$$\mathcal{L} = \mathcal{L}_q + w_h\,\mathcal{L}_h + \lambda\,\big\lVert\, h_{t+1} - h_t - \tfrac{\Delta t}{w l}\big(\textstyle\sum Q_{\text{up}} + Q_{\text{in}} - Q_{\text{out}}\big) \big\rVert^2$$

**Verdict: it changes training incentives but removes the inference-time
governor, so it cannot be *expected* to restrain the rollout the way the hard
constraint does.** Reasons:

1. **Active only in training.** At inference the network runs unconstrained — the
   same two free autoregressive outputs, no algebraic projection. The runaway
   mode is still reachable.
2. **Penalty, not projection.** Minimising $\lVert r\rVert^2$ in expectation
   drives the *average* residual small but leaves a nonzero per-step, per-node
   error; in a closed loop even a tiny consistent bias compounds. A penalty
   attenuates the bias; it does not guarantee the loop gain stays `< 1`.
3. **Only sees trained horizons.** The penalty is enforced over the curriculum
   rollout length (here ≤ 10 steps). A 1906-step rollout extrapolates far beyond
   that; the hard constraint holds at step 1906 by construction, the soft one
   only where the loss reached.
4. **No positivity floor.** The hard layer also clamps `q, h ≥ 0`; a penalty
   gives neither the conservation projection nor the floor.

Expected behaviour: a well-weighted penalty **delays and softens** divergence
relative to the fully-unconstrained hps2 baseline, but is "sometimes stable,
sometimes still diverges, sensitive to λ and horizon" — not provably bounded.

*Trade it buys:* it **avoids** the stiff `∂h_norm/∂q_norm ≈ 22` gradient
amplification (see §5) because `q`/`h` gradients decouple — i.e. it moves the
difficulty from *inference stability* back to nothing, but loses the guarantee.

---

## 3. Hybrid: soft supervision + a cheap inference-time governor

The practical sweet spot keeps a **lightweight hard stabiliser in the rollout
loop** while moving conservation *supervision* into the loss. Three separable
pieces:

**(1) Decoder predicts both states independently** — same as
`enforce_mass_balance = false` today (`out_dim = 2`). Avoids the stiff-gradient
coupling of the hard decoder.

**(2) Inference-time governor** (the "hard" part that stays), two strengths:

- *Minimal — positivity clamp only:*
  ```julia
  q_new = max.(0f0, state[1:1, :] .+ Δq)
  h_new = max.(0f0, state[2:2, :] .+ Δh)
  ```
  Removes the negative-into-overflow path, nearly free; does **not** enforce
  conservation.
- *Stronger — partial projection toward the balance:* compute the
  mass-balance-implied depth `h_mb` (reuse the `MassBalanceLayer` math) and blend
  $$h_{\text{new}} = (1-\alpha)\,h_{\text{decoder}} + \alpha\,h_{\text{mb}}, \qquad \alpha \in [0,1]$$
  with `α = 1` = today's hard constraint, `α = 0` = pure soft, `α ∈ (0,1)`
  damping the free mode without fully clamping `h` to `q`.

**(3) Soft conservation penalty in the loss** (as in §2), almost free once `h_mb`
is computed for piece 2.

Mapping onto the existing toggle (would become an enum + two knobs):

| Mode | decoder out | inference governor | penalty λ |
|---|---|---|---|
| hard (today, hps1) | 1 (`Δq`) | full projection (`α = 1`) | — |
| soft-only | 2 | none | > 0 |
| **hybrid** | 2 | clamp or `α ∈ (0,1)` | > 0 |
| unconstrained (hps2) | 2 | none | 0 |

I.e. extend `enforce_mass_balance` → `mass_balance = "hard"|"soft"|"hybrid"|"off"`
plus `mb_projection_alpha` (α) and `mb_penalty_weight` (λ), all sweepable.

**Why it is the sweet spot:** recovers the inference-time governor a pure soft
penalty lacks (acts at every step incl. beyond the training horizon), while
avoiding the stiff-gradient training difficulty of the pure hard decoder, with a
tunable cost/benefit via α and λ.

**Cheapest first experiment:** take the MB-off config and add *only* the
positivity clamp (piece 2a) — no loss change, no projection. If the clamp alone
tames the overflow, most of hps2's blowup was the unbounded positive-feedback
mode rather than conservation drift, and the full projection may be unnecessary.

---

## 4. Implicit vs explicit vs trapezoidal: the θ-method

The current update is **fully implicit** (backward Euler) — it uses the
*just-predicted* discharge $q_{t+1}$ and next forcing:

$$h_{t+1}^{\text{impl}} = h_t + \frac{\Delta t}{w l}\,\underbrace{\big(\textstyle\sum Q_{\text{up}}(q_{t+1}) + Q_{\text{in},t+1} - q_{t+1}\big)}_{F_{t+1}}$$

The **explicit** (forward Euler) version uses only quantities known at time $t$:

$$h_{t+1}^{\text{expl}} = h_t + \frac{\Delta t}{w l}\,\underbrace{\big(\textstyle\sum Q_{\text{up}}(q_{t}) + Q_{\text{in},t} - q_{t}\big)}_{F_t}$$

Because both share the base $h_t$, a convex blend of the two depth estimates is
**algebraically identical** to θ-weighting the fluxes (θ-method):

$$\theta\,h_{t+1}^{\text{impl}} + (1-\theta)\,h_{t+1}^{\text{expl}} = h_t + \frac{\Delta t}{w l}\big[\theta F_{t+1} + (1-\theta)F_t\big]$$

- $\theta = 1$ → today's scheme (fully implicit)
- $\theta = 0$ → pure explicit
- $\theta = \tfrac12$ → **Crank–Nicolson / trapezoidal** (predictor–corrector /
  Heun): explicit as predictor, implicit as corrector.

"Average the two h estimates" and "use a θ-scheme" are the same knob.

**What blending buys:**

1. **Directly controls gradient stiffness** — a *structural* alternative to the
   `increment` loss hack (see §5). $\partial h/\partial q \propto \theta$.
2. **Smooths / de-noises h** — with $\theta < 1$ a single noisy $q_{t+1}$ has less
   leverage on $h_{t+1}$; genuine variance reduction on the depth channel.
3. **More accurate** — trapezoidal ($\theta = \tfrac12$) is 2nd-order in
   $\Delta t$; both endpoints beat either single endpoint.

**The catch — stability moves the opposite way.** The implicit term is exactly
the same-step negative feedback that provides the rollout governor (predict high
outflow → depth drops *now* → throttles next step). The explicit term uses only
past values, reintroducing a **one-step delay** in that feedback — the classic
route to oscillation / CFL-type growth at long horizons and high flows.

| θ | gradient stiffness | rollout stability | accuracy |
|---|---|---|---|
| 1 (now) | max (≈ 22×) | unconditional | 1st-order, diffusive |
| ½ (CN) | halved | strong, mild oscillation risk | 2nd-order |
| 0 (explicit) | none | conditional (CFL-like) | 1st-order |

Crucially, **even $\theta = 0$ still integrates physical fluxes**, so `h` stays on
the conservation structure — categorically better than the free-`h` hps2 blowup.
The question is only *how much* implicit damping to keep.

---

## 5. Tuning θ: the value that cancels the stiffness

The training stiffness lives in **normalised** space. Chaining the postscale
($q_\text{phys} = p_q(q_\text{norm}\sigma_q + \mu_q)$,
$h_\text{phys} = p_h(h_\text{norm}\sigma_h + \mu_h)$, with $p_q/p_h = w l$)
through the θ-weighted update:

$$\frac{\partial h_\text{norm}}{\partial q_\text{norm}} = \frac{1}{p_h\sigma_h}\cdot\Big(-\theta\,\frac{\Delta t}{w l}\Big)\cdot p_q\sigma_q = -\,\theta\,\frac{\Delta t}{w l}\cdot\frac{p_q}{p_h}\cdot\frac{\sigma_q}{\sigma_h}$$

Since $p_q/p_h = w l$, the **$w l$ cancels**:

$$\boxed{\;\frac{\partial h_\text{norm}}{\partial q_\text{norm}} = -\,\theta\,\frac{\Delta t\,\sigma_q}{\sigma_h}\;}$$

With $\Delta t = 86400$, $\sigma_q = 0.006244$, $\sigma_h = 24.582$:

$$\frac{\Delta t\,\sigma_q}{\sigma_h} = 21.95 \quad\Rightarrow\quad \frac{\partial h_\text{norm}}{\partial q_\text{norm}} = -21.95\,\theta$$

reproducing the ≈ −22 amplification at $\theta = 1$.

### Why θ = w·l/Δt is the wrong guess

Intuition "pick θ to cancel the amplification" is right, but $w l/\Delta t$ fails
on two counts:

- **Units.** θ is a dimensionless convex weight in $[0,1]$; $w l/\Delta t$ has
  units $\text{m}^2/\text{s}$. (It happens to be ≈ 0.19 for mean $w,l$, a
  dimensional coincidence.)
- **Per-node.** $w l$ differs per reach, so it cannot be a single blend weight.
- **Cancels the wrong factor.** The $w l$ already dropped out of the *normalised*
  gradient above; setting $\theta = w l/\Delta t$ would leave
  $-\theta\,(w l)\,\sigma_q/\sigma_h$, which is enormous.

### The θ that does cancel it

Set the bracket to −1:

$$\theta^\star = \frac{\sigma_h}{\Delta t\,\sigma_q} = \frac{1}{21.95} \approx 0.0456$$

Dimensionless, uniform across nodes, and — not a coincidence — **exactly `base`**,
the same quantity `h_loss_scale = "increment"` is built from ($w_h = \text{base}^2$,
see [docs/mass_balance_notes.md](mass_balance_notes.md) §3–4 and
[src/run.jl](../src/run.jl)). So $\theta = \text{base}$ makes
$\partial h_\text{norm}/\partial q_\text{norm} = -1$ **structurally in the forward
model**, rather than compensating in the loss after the fact.

**But $\theta^\star \approx 0.046$ means the scheme is 95.4 % explicit** — it hands
back most of the CFL-type rollout risk (§4), because the implicit fraction *is*
the governor. You cannot simultaneously null the stiffness and keep full implicit
damping; θ slides between them:

$$\theta \downarrow \;\Rightarrow\; \text{stiffness} \downarrow \;\text{and}\; \text{stabilisation} \downarrow$$

### Recommendation

- Make θ a config parameter (sweepable like the MB toggle), **default
  $\theta = \tfrac12$**, and evaluate on the **long** validation rollout — that is
  where the explicit component's weakness shows.
- Expect the sweet spot in $\theta \in [0.5, 0.8]$: most of the stiffness relief,
  most of the stabilisation. If it holds up, θ could largely supersede
  `h_loss_scale = "increment"` (structural cancellation vs loss-side
  compensation).
- **Best of both:** keep θ moderate for stability and let a mild `increment` mop
  up the residual stiffness — the two mechanisms act on the *same*
  $\Delta t\,\sigma_q/\sigma_h$ group from opposite sides, so they are
  complementary. `full-cancellation θ⋆` is generally **not** recommended (too
  explicit).

Implementation is cheap: the layer already computes $\sum Q_{\text{up}}$, inwater,
and holds $q_t$ in `state[1:1, :]`, so the explicit branch is one extra topology
multiply plus the θ blend.

**Status — implemented.** `MassBalanceLayer` now carries a `θ` field and the
forward pass evaluates $\theta F_{t+1} + (1-\theta)F_t$ (see
[src/gnn.jl](../src/gnn.jl)). The value is exposed as the `[model].mb_theta`
config key (default `1.0`, so existing runs are bit-identical) and is sweepable
via `"model.mb_theta"` in `[hparsearch.search_space]` (see
[experiments/template.toml](../experiments/template.toml) and
[experiments/template_hparsearch.toml](../experiments/template_hparsearch.toml)).
The explicit branch is skipped entirely when $\theta = 1$, and at $\theta = 1$
the update is numerically identical to the previous fully-implicit code.

---

## 6. Regime validity: CFL, quasi-steadiness & receptive field

Everything above assumes the daily update behaves like a **stiff, quasi-steady
routing relaxation**. This section tests that assumption against the wflow_sbm
reference runs for two catchments of very different size, because the conclusions
about receptive field and the multi-hop / global accumulation option depend on
it. Numbers from [scripts/regime_analysis.jl](../scripts/regime_analysis.jl)
(streamed over the 11.6 GB `sava_v081` output in 500-step blocks; validated by
exactly reproducing the small-catchment figures) and
[scripts/network_diameter.jl](../scripts/network_diameter.jl).

### 6.1 The three diagnostics

- **Courant number** — information travels at the kinematic-wave celerity
  $c = \mathrm{d}Q/\mathrm{d}A = \tfrac53 v$ (wide Manning channel), *not* the bulk
  velocity. Cells crossed per step: $\mathrm{Cr} = c\,\Delta t / L_\text{cell}$.
  This establishes the **regime**, not a stability bound.
- **Quasi-steady ratio** — storage change vs flux,
  $r = |w\,l\,\Delta h/\Delta t| / Q$. Quasi-steady $\Leftrightarrow r \ll 1$,
  i.e. $q \approx (I-A)^{-1} q_\text{lateral}$ each step.
- **Network diameter** — longest downstream path in hops = the cost (SpMVs) of a
  full upstream accumulation and the receptive field needed to see the whole
  contributing network.

### 6.2 Results — small vs full catchment

| metric | `sava_small_v081` | `sava_v081` (full) |
|---|---|---|
| flowing river cells | 102 | 868 |
| network nodes / **diameter** | 323 / **59 hops** | 8235 / **585 hops** |
| mean downstream path | 31.5 hops | 321.5 hops |
| peak Q median / max [m³/s] | 12.8 / 218 | 34.6 / **8616** |
| river width p90 [m] | 19.6 | **121.5** |
| peak depth max [m] | 11.4 | **32.4** |
| celerity median / p90 [m/s] | 1.95 / 5.5 | 2.73 / **34.9** |
| **travel $c\Delta t$ median [km]** | 168 | 236 |
| **Courant median / p90 / max** | 92 / 440 / 2466 | **152 / 2226 / 111727** |
| cells with Cr>1 | 100 % | 99 % |
| aggregate $r$ median | 0.0004 | 0.0017 |
| fraction $r<0.1$ | 99.8 % | 93.9 % |
| flood-front $r<0.1$ (g>0.4, Q≥5) | **100 %** | **97.0 %** |
| flood-front max $r$ | 0.047 | **2.378** |
| g>0.8 **wet** cells (n, max $r$) | 0, — | 14062, **2.38** |
| g>0.8 **dry** cells (n, max $r$) | 29, 4.3e3 | 46360, 4.1e4 |

### 6.3 What holds, and what breaks

1. **Stiffness — confirmed, stronger on the full basin.** Median Courant 92 → 152,
   99–100 % of cells Cr>1, max ~1.1×10⁵. The chain $\mathrm{Cr}\gg1 \Rightarrow$
   *stiff* $\Rightarrow$ explicit diverges / implicit governs holds a fortiori.
   The $\mathrm{Cr}$ number is the right diagnostic for the **regime**, not a
   wave-CFL stability limit (an implicit/steady scheme is unconditionally stable
   regardless of $\mathrm{Cr}$).

2. **Dry-headwater artifact — confirmed, identical in character.** The
   catastrophic $r$ (up to 4.1×10⁴) is *entirely* dry cells (Q<5 m³/s): the
   denominator $Q\to0$, not real storage. In the g>0.8 bin the full basin has
   46 360 such dry samples and **zero** wet samples on the small basin. This is
   the corner the `MassBalanceLayer` positivity floor exists to protect, and it
   is a small-signal artifact rather than a mass-flux failure.

3. **Quasi-steadiness is NOT airtight on large channels.** Unlike the headwater
   basin (flood fronts 100 % $r<0.1$, max 0.047), the full basin has **14 062
   genuinely flowing cells** (Q≥5) in the biggest-jump bin with $r$ up to **2.38**
   — cells where per-step storage change *exceeds* the flux. This is physically
   real: `sava_v081` has wide channels (width p90 121 m, depth up to 32 m) with
   **floodplain storage** the tiny basin lacks. ~3 % of main-stem flood fronts
   are genuinely **unsteady**.

4. **The "whole network couples per step" claim is a small-catchment artifact.**
   - Small: median travel **92 cells > diameter 59** → a perturbation crosses the
     *entire* network in one day.
   - Full: median travel **152 cells < diameter 585** → at baseflow a signal
     reaches only ~¼ of the main stem per step; **only during floods**
     (p90 celerity → Cr 2226 ≫ 585) does it traverse the whole network in a day.

### 6.4 Consequence for the multi-hop / global accumulation option

On a DAG, full upstream accumulation is the triangular solve
$q = (I-A)^{-1} q_\text{lateral}$ (forward substitution — a **direct solve**, not
an iteration), costing one triangular solve or `diameter` SpMVs (59 small / 585
full). Two design points sharpen with the full-basin data:

- **Features vs structural solve.** Feeding $A^2x, A^3x,\dots$ as extra decoder
  *inputs* does **not** stabilise a free/MB-off model — rollout stability is a
  property of the state→state Jacobian, and richer inputs give a free decoder
  *more* to amplify. Only using the accumulation as a hard conservation operation
  (the $(I-A)^{-1}$ solve) stabilises, and doing so is essentially rebuilding a
  global mass balance — the fully-implicit, whole-network end of the θ-spectrum.
- **On a realistic basin it is more physically *necessary during floods*,** not
  merely a nicer 1-hop sum: that is exactly when Cr ≫ diameter and storage
  matters. But it must be paired with genuine **dynamic (unsteady) storage** for
  the wide-channel cells where $r>1$ — a positivity floor alone (which only
  addresses the dry-cell artifact) is not enough. At baseflow, conversely, the
  cheap **1-hop upstream sum is *better* justified** on the large basin than the
  small one (the daily step does not globally couple headwater-to-outlet).

**Net:** the stiffness and dry-cell conclusions generalise; the receptive-field
argument does not — global accumulation earns its keep on floods of large basins,
and the real limit on the quasi-steady/hard-MB approximation is floodplain
storage unsteadiness ($r>1$ on wide wet channels), not the dry headwaters.

---

## 7. Bounds / guides on Q: the Manning rating curve

Lowering θ damps how strongly a *wrong* $q$ propagates into $h$, but it does not
by itself pull $q$ toward the right value — the observed failure mode also
includes **over-prediction of $q$**. A natural physics prior is the Manning
rating curve that couples discharge to depth:

$$Q_\text{rating} = \frac{1}{n}\,w\,H^{5/3}\,\sqrt{S} \quad(\text{wide channel},\ R\approx H)$$

or, keeping the hydraulic radius, $R = wH/(w+2H)$ (rectangular section). This is
attractive because it re-couples $q$ to $h$ (currently $h$ is slaved to $q$ but
nothing pushes $q$ back), giving a soft restoring force on discharge.

### 7.1 Conditions for validity

The single-valued rating $Q(H)$ holds only when **all** of the following are met:

1. **Quasi-steady / kinematic regime** ($r \ll 1$): storage change is negligible,
   so the friction slope ≈ bed slope and $Q$ is a single-valued function of $H$
   (no loop rating / hysteresis on the rising vs falling limb).
2. **In-bank flow** ($H \le$ bankfull `RiverDepth`): overbank flow has floodplain
   width and roughness, breaking the prismatic-channel geometry.
3. **Wide channel** ($R \approx H$): needed for the exponent-$5/3$ form; use the
   rectangular $R$ otherwise.
4. **Instantaneous law, but a daily-averaged target.** The rating is pointwise in
   time; wflow integrates it at sub-daily internal steps while the output — and
   the GNN target — is the **daily mean**. Averaging a nonlinear $Q \propto
   H^{5/3}$ over a rising/falling hydrograph (Jensen's inequality + hysteresis)
   decouples mean-$Q$ from mean-$H$.
5. Un-processed slope; no reservoirs/lakes.

### 7.2 How often it actually holds (empirical)

Test: $\rho = Q_\text{Manning}/Q_\text{actual}$ per river cell-step against the
wflow reference, streamed over the full record and broken down by the conditions
above ([scripts/rating_curve_test.jl](../scripts/rating_curve_test.jl)). "Valid"
≈ $\rho$ near 1; columns are the fraction of cell-steps within ±25 % and within a
factor of two.

| condition | small ±25 % / ×2 | full ±25 % / ×2 | full $\rho$ median |
|---|---|---|---|
| ALL | 13.7 % / 52.9 % | 12.3 % / 31.9 % | 1.12 |
| wet ($Q\ge5$) | 6.3 % / 44.7 % | 15.1 % / 36.5 % | 0.45 |
| quasi-steady ($r<0.1$) | 13.8 % / 53.1 % | 12.6 % / 33.0 % | 1.02 |
| unsteady ($r\ge0.1$) | 1.0 % / 1.0 % | 8.8 % / 15.9 % | **13.5** |
| in-bank ($H\le D_\text{bf}$) | 14.6 % / 55.5 % | 12.2 % / 32.9 % | 1.23 |
| overbank ($H>D_\text{bf}$) | 1.9 % / 17.1 % | 12.8 % / 27.6 % | 0.62 |
| high flow ($Q>500$) | — | 22.8 % / 46.3 % | 0.51 |
| rectangular $R$, ALL | 16.4 % (±25 %) | 11.9 % (±25 %) | 1.02–1.23 |

**Verdict: the rating is qualitatively right but quantitatively loose.** The
median is $O(1)$ (a real relationship exists) and the conditional structure is
exactly as theory predicts — it degrades most when unsteady ($\rho$ median 13.5)
and overbank, confirming conditions 1–2. But even in its best corner (wet,
quasi-steady, in-bank, high flow) only ~15–23 % of cell-steps land within 25 % of
wflow's $Q$, and only ~35–46 % within a factor of two; the p10→p90 spread is a
factor of 60+ on the full basin. The dominant culprit is **condition 4** (daily
averaging of the nonlinear law), which is intrinsic to the quantities being
constrained and does **not** wash out by re-calibrating $n$ or $S$.

### 7.3 Consequence for using it as a Q constraint

- **Do not impose the rating as a hard equality or a strongly-weighted two-sided
  penalty** — it would fight the reference data ~85 % of the time.
- If a discharge prior is wanted, prefer a **one-sided soft bound**: penalise only
  $Q$ *above* the rating (discourage discharge the storage cannot support),
  respecting the loose relationship while still capping runaway over-prediction.
- This keeps **θ as the primary, assumption-free lever** for the h-overshoot
  (it makes no claim that the rating holds; it only rescales how the real
  $q$-error propagates into $h$). Assumption-free routes to the $Q$
  over-prediction itself — extending the rollout curriculum and peak-weighting
  $\mathcal{L}_q$ — do not depend on the rating and should be preferred over a
  two-sided rating penalty.

---

## 8. Quick reference

| Question | Answer |
|---|---|
| What stabilises the rollout? | The **implicit** term's same-step negative feedback (inference-time) |
| Does MB-off diverge? | Yes — Float32 overflow in free rollout (sweep hps2) |
| Can a soft penalty replace it? | Partially; delays/softens but no bound (acts only in training) |
| Recommended relaxation? | **Hybrid** — free decoder + cheap inference governor (clamp or α-projection) + soft penalty |
| Cheapest thing to try | Positivity clamp on `q,h` in the forward loop; no retrain of loss |
| Implicit / explicit / trapezoidal? | θ-method; $\theta=1$ implicit (now), $0$ explicit, $\tfrac12$ Crank–Nicolson |
| Blend two h estimates ≡ ? | θ-weighting the fluxes (same base $h_t$) |
| Stiffness law | $\partial h_\text{norm}/\partial q_\text{norm} = -\theta\,\Delta t\,\sigma_q/\sigma_h = -21.95\,\theta$ |
| θ that cancels stiffness | $\theta^\star = \sigma_h/(\Delta t\,\sigma_q) \approx 0.046 = \text{base}$ (but 95 % explicit) |
| Is $\theta = w l/\Delta t$ correct? | No — dimensional, per-node, cancels a factor already gone from the normalised gradient |
| Suggested default θ | $\tfrac12$; sweep $[0.5, 0.8]$, evaluate on the long rollout |
| Is θ implemented? | Yes — `MassBalanceLayer.θ` / `[model].mb_theta` (default `1.0`), sweepable as `"model.mb_theta"` |
| Is the daily step quasi-steady? | Yes on flowing cells ($r$ median 4e-4 small / 2e-3 full); breaks on wide floodplain cells (full: 14k wet cells $r$ up to 2.4) and is a denominator artifact on dry headwaters |
| Does the whole network couple per step? | Small basin yes (travel 92 cells > diameter 59); full basin only during floods (baseflow travel 152 < diameter 585) |
| Is the Manning rating a valid Q law? | Qualitatively yes ($\rho$ median ~1), quantitatively loose — only ~15–23 % within ±25 %, ~35–46 % within ×2 even in the best regime; breaks when unsteady/overbank and under daily averaging |
| Use the rating to constrain Q? | Not as a hard/two-sided penalty (fights data ~85 %); at most a **one-sided** soft cap on over-prediction. Keep θ + rollout curriculum + peak-weighted $\mathcal{L}_q$ as the primary levers |

---

## References
- Crank & Nicolson, *A practical method for numerical evaluation of solutions of
  partial differential equations of the heat-conduction type*, Proc. Camb. Phil.
  Soc. 43 (1947) — the θ = ½ trapezoidal scheme.
- Courant, Friedrichs & Lewy, *Über die partiellen Differenzengleichungen der
  mathematischen Physik*, Math. Ann. 100 (1928) — CFL stability condition for
  explicit schemes.
- Raissi, Perdikaris & Karniadakis, *Physics-Informed Neural Networks*, J. Comp.
  Phys. 378 (2019); arXiv:1711.10561 — soft physics-residual penalties.
- Beucler et al., *Enforcing Analytic Constraints in Neural Networks Emulating
  Physical Systems*, Phys. Rev. Lett. 126 (2021); arXiv:1909.00912 — hard vs soft
  constraints.
- van Verseveld et al., *Wflow_sbm v0.7.3*, Geosci. Model Dev. 17 (2024) — the
  kinematic-wave routing being emulated.
- Ponce & Simons, *Shallow Wave Propagation in Open Channel Flow*, J. Hydraul.
  Div. ASCE 103 (1977) — kinematic-wave celerity $c = \tfrac53 v$ and the
  applicability regime for kinematic vs dynamic routing.
- Manning, *On the flow of water in open channels and pipes*, Trans. Inst. Civ.
  Eng. Ireland 20 (1891) — the $Q = \tfrac1n w H^{5/3}\sqrt S$ rating used as the
  discharge–depth prior in §7.
- Regime analysis scripts: [scripts/regime_analysis.jl](../scripts/regime_analysis.jl)
  (CFL, quasi-steady ratio, flood-front & wet/dry split — streaming),
  [scripts/network_diameter.jl](../scripts/network_diameter.jl) (DAG diameter) and
  [scripts/rating_curve_test.jl](../scripts/rating_curve_test.jl) (Manning
  rating-curve validity, §7).
- Companion: [docs/mass_balance_notes.md](mass_balance_notes.md),
  [docs/training_tuning_notes.md](training_tuning_notes.md).
