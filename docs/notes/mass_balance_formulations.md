# Mass-Balance Formulations — Three Conservative Closures

A comparative design note on **three ways** to enforce the kinematic-wave water
balance as a hard constraint inside the GNN decoder. All three conserve mass
exactly; they differ in **which state variable the network predicts freely** and
**which is pinned by conservation**, and therefore in their stability, their
regime of validity, their physical assumptions, and their cost.

This note is the synthesis of the design thread in
[mass_balance_stability_notes.md](mass_balance_stability_notes.md) (θ-method,
CFL/quasi-steady regime analysis §6, rating-curve validity §7) and
[mass_balance_notes.md](mass_balance_notes.md) (the implemented hard constraint).
It exists to hold all three options side by side before an implementation
decision is taken.

Source touchpoints for the **implemented** option: [src/gnn.jl](../src/gnn.jl)
(`MassBalanceLayer`, forward pass, `mb_diagnostics`). Empirical basis: the MB
on/off sweep (`experiments/sava_small_v081_mb_sweep`) and the regime analysis
([scripts/regime_analysis.jl](../scripts/regime_analysis.jl),
[scripts/network_diameter.jl](../scripts/network_diameter.jl)).

> **Design constraints assumed throughout** (from the design thread):
> 1. `(q, h)` must be the network **input** and the network **output**; the
>    decoder's raw output may represent anything *before* the MB layer.
> 2. The MB layer *preferably* contains **no iterative solve** — a single forward
>    apply. **This constraint is relaxed in the marked "⟳ iterative variant"
>    subsections below**, which trade a bounded, fixed number of inner iterations
>    for extra physical fidelity (nonlinear ratings, variable-parameter routing).
>    Every such relaxation is called out explicitly and its iteration count is
>    carried into the cost estimate (§4).
> 3. The adjacency `A` is **static**, so any inverse / prefactor involving `A`
>    may be **precomputed** once — but **only while the coefficients multiplying
>    `A` are also static.** Per-step (flow-dependent or learned) coefficients
>    restale the operator and forbid precompute; that is the boundary the
>    iterative variants cross.
> 4. **Train-short / roll-long gap must be preserved.** Training is limited to
>    short rollouts (curriculum max ≈ 10 steps) for feasibility, but inference
>    must stay stable over **multi-year** rollouts (1000s of steps) — the current
>    MB achieves this (stable at 1906 steps trained on ≤10). Any redesign must
>    inherit this generalisation *past the training horizon*. The structural
>    requirement (derived in §6): **every step must re-anchor the state to the
>    forcing-determined, in-distribution attractor** — i.e. the h-channel needs a
>    per-step restoring force ($\rho(\partial_h F)<1$), not just a contraction that
>    happens to hold at training horizons. This constraint is checked explicitly
>    for each option in its §_.3 and consolidated in §6.

---

## 0. Common notation

| symbol | meaning | units |
|---|---|---|
| $S_i$ | channel water storage in cell $i$ | m³ |
| $h_i$ | water depth, $S_i = h_i\,w_i\,l_i$ | m |
| $w_i, l_i$ | channel width, cell length (static) | m |
| $Q_{\text{out},i}$ | discharge leaving cell $i$ ( $\equiv$ `river_q` = $q_i$ ) | m³/s |
| $I_i$ | lateral inflow (`river_inwater`, from forcing) | m³/s |
| $A$ | routing adjacency; $A_{ij}=1$ if $j$ drains to $i$ | — |
| $A^\top q$ | gather of upstream discharge into each cell | m³/s |
| $\Delta t$ | timestep (1 day = 86 400 s) | s |
| $\sigma_q,\sigma_h$ | z-score scales of $q,h$ | — |

The **continuity equation** each cell must satisfy is the same for all three
formulations:

$$\frac{\mathrm{d}S_i}{\mathrm{d}t} \;=\; \underbrace{\sum_{j\to i} Q_{\text{out},j}}_{(A^\top Q_\text{out})_i} \;+\; I_i \;-\; Q_{\text{out},i}. \tag{C}$$

Summed over all cells, every internal $Q_\text{out}$ appears once as an inflow
and once as an outflow and telescopes, leaving

$$\sum_i S_i^{t+1} = \sum_i S_i^{t} + \Delta t\Big(\sum_i I_i - \sum_{\text{sinks}} Q_\text{out}\Big), \tag{C$'$}$$

i.e. **global conservation holds for any closure**, provided (C) is solved
exactly and $A$ is column-conservative (each cell routes to exactly one
downstream cell). The three options are three ways to supply the *second*
relation that closes the one-equation-two-unknowns system (C).

The **key regime numbers** (from
[mass_balance_stability_notes.md](mass_balance_stability_notes.md) §6, both
catchments) that every stability claim below refers to:

| quantity | `sava_small_v081` | `sava_v081` (full) |
|---|---|---|
| river nodes / network diameter | 323 / **59 hops** | 8235 / **585 hops** |
| Courant median / p90 / max | 92 / 440 / 2466 | 152 / 2226 / **111727** |
| quasi-steady ratio $r=|wl\,\Delta h/\Delta t|/Q$ median | 4×10⁻⁴ | 1.7×10⁻³ |
| fraction $r<0.1$ | 99.8 % | 93.9 % |
| genuinely unsteady wet cells ($r>1$) | 0 | ~14 000 (max $r$ 2.4) |

Two facts drive everything: (i) the daily step is **massively super-Courant**
(Cr ≫ 1 everywhere) so any *explicit* advance of a channel is under-resolved and
only an implicit/steady treatment is unconditionally stable; (ii) the flow is
**quasi-steady** ($r\ll1$) on ~94–99.8 % of wet cells, breaking only on the wide
floodplain cells of the large basin.

### 0.1 The rollout-Jacobian framework (used for every option)

Rollout blow-up is not about a single step's error — it is about whether errors
**grow** as the autoregressive map is iterated. The governing object is the
**Jacobian of the state-update map** $s^{t+1}=F(s^t,\text{forcing})$, where the
*state* $s$ is whatever is carried across steps. An error $\delta s^t$ evolves as
$\delta s^{t+1}\approx J\,\delta s^t$ with $J=\partial F/\partial s$, so over a
horizon $H$ it scales like $\|J\|^H$ (or $\rho(J)^H$ asymptotically). Hence:

$$\boxed{\ \rho(J) < 1 \Rightarrow \text{contraction (stable)};\quad \rho(J) = 1 \Rightarrow \text{marginal (drift)};\quad \rho(J) > 1 \Rightarrow \text{geometric blow-up}.\ }$$

The decisive design lever is **which variable is the carried state** (in the
loop) versus a **diagnostic readout** (computed each step, fed nothing forward).
A variable that is diagnostic-only has *no row/column in $J$* — its error cannot
compound, only appear once. Writing the two channels as a $2\times2$ block
Jacobian in $(q,h)$ and asking which blocks are zero is the single tool that
separates the three options below, and it is used verbatim in each §_.3. Let

$$D_q=\frac{\partial(\text{decoder out})}{\partial q},\quad D_h=\frac{\partial(\text{decoder out})}{\partial h},\quad \kappa=\frac{\Delta t}{wl},\quad \text{stiffness }\ \kappa(A^\top-I)\ \big(=-21.95\text{ in norm. units at }\theta=1\big).$$

### 0.2 Open downstream boundary (common to all three)

The river system is **open**: water leaves at the downstream-most cell(s). This
is already encoded in $A$ — a **sink** is a cell with *no downstream neighbour*,
so its outflow appears in no $(A^\top Q)_i$ and is precisely the
$\sum_{\text{sinks}} Q_\text{out}$ export term in (C′). No special case is
needed; the open outlet is the automatic consequence of the sink having no
out-edge. All three formulations are compatible, with three points to note:

- **Nilpotency ⇔ open DAG.** $(I-C_1A^\top)^{-1}$ (B) and $(I-A^\top)^{-1}$ (C)
  are finite, precomputable Neumann series **because $A^\top$ is nilpotent**, and
  $A^\top$ is nilpotent *precisely because the network is an open, acyclic DAG
  terminating at sinks*. The open boundary is therefore not merely tolerated by
  B and C — it is the **structural precondition** that makes their precomputed
  inverse well-posed. In C the accumulation naturally *terminates* at the outlet:
  the sink's readout is the whole-basin sum of $(I_\text{lat}-\mathrm{d}S/\mathrm{d}t)$,
  i.e. total basin outflow.
- **The outlet is disciplined only from upstream → needs loss supervision.**
  Interior cells' $q$ is doubly constrained (upstream inflow *and* being someone's
  upstream inflow); the sink's outflow is constrained only from upstream, so
  topology cannot catch outlet error — the **loss** must, which is convenient
  since the outlet is typically the gauged point. In C the sink concentrates the
  undamped linear-spatial accumulation error (§6), so the outlet is where a
  storage-rate bias shows up most; the **B/C hybrid mitigates this** ($C_3<1$
  backbone damps storage error before read-out).
- **Multi-sink caveat.** A clipped sub-basin has *several* boundary-exit cells
  (any cell whose true downstream neighbour was clipped). All three treat every
  no-out-edge cell as a sink identically, but the **outlet set is not a single
  node** — the same decision already open for the volume-budget outflow term
  (TODO §2d). B may also choose the sink reach's outlet condition (same $(K,x)$
  vs a transmissive zero-gradient $\partial Q/\partial x=0$).

---

## 1. Option A — **Current implementation**: free $q$, $h$ slaved by θ-balance

### 1.1 Formulation

The decoder predicts discharge $q^{t+1}$ freely (one output channel); depth is
**derived** from the θ-weighted discretisation of (C), solved algebraically for
$h$ (no coupling of the *new* $h$ across cells, because the balance is written
with $q$ known):

$$h_i^{t+1} = h_i^{t} + \frac{\Delta t}{w_i l_i}\Big[\theta\big(\underbrace{(A^\top q^{t+1})_i + I_i^{t+1} - q_i^{t+1}}_{\text{implicit net flux}}\big) + (1-\theta)\big((A^\top q^{t})_i + I_i^{t} - q_i^{t}\big)\Big], \tag{A}$$

followed by a positivity floor $h_i^{t+1} \leftarrow \max(0, h_i^{t+1})$.
$\theta\in[0,1]$ is the implicitness weight (`mb_theta`, default $1$ = fully
implicit / backward Euler). This is [src/gnn.jl](../src/gnn.jl)
`MassBalanceLayer`; `q` is the primary predicted state, `h` is diagnostic.

### 1.2 Conservation

Exact per (C′): a single $q_\text{out}$ per cell is gathered through $A^\top$.
**Caveat:** the positivity floor $\max(0,h)$ is a deliberate, small conservation
break to prevent dry-cell negatives (the denominator artifact of §6.3). It is
localised to $Q\to0$ headwater cells and is not a flux error.

### 1.3 Stability

- The **h-given-q** update is unconditionally stable (backward Euler at
  $\theta=1$) *for the h-channel* — it does not care about Courant.
- The **failure mode is the free $q$-channel.** Nothing pins $q$ to storage, so
  the learned $q$-map has an unconstrained state→state Jacobian. Rollout
  amplifies geometrically (`amp ≈ 12`, `peak_ratio ≈ amp^H`) from stiff
  high-flow-start anchors. The amplification is magnified by the **normalisation
  stiffness**

  $$\frac{\partial h_\text{norm}}{\partial q_\text{norm}} = -\theta\,\Delta t\,\frac{\sigma_q}{\sigma_h} = -21.95\,\theta,$$

  so any $q$-error is ~22× loud in $h$ at $\theta=1$. Critically this blow-up is
  observed on the **small basin, which is airtight quasi-steady** ($r<0.1$ at
  100 % of flood fronts) — proving the instability is the *free-q Jacobian*, not
  a CFL or quasi-steadiness breakdown. Lowering $\theta$ damps the stiffness but
  95 %-explicit ($\theta^\star\approx0.046$) to fully cancel it, which
  reintroduces the explicit-advance risk.
- MB-**off** (both $q,h$ free) diverges to Float32 overflow (sweep hps2),
  confirming the constraint is a genuine rollout governor.

#### Jacobian / spectrum

The carried state is $s=(q,h)$ (both fed back as decoder inputs). With
$q^{t+1}=D(q^t,h^t,f)$ and $h^{t+1}=h^t+\kappa(A^\top-I)q^{t+1}$ the block
Jacobian is

$$J_A=\begin{bmatrix} D_q & D_h \\[2pt] \kappa(A^\top-I)D_q & \; I+\kappa(A^\top-I)D_h \end{bmatrix}.$$

The stiffness $\kappa(A^\top-I)$ sits in the **bottom row** and creates an
eigenvalue $>1$ by coupling with $D_h$ (the decoder's sensitivity to the **h
input**) and $D_q$. Because $\kappa(A^\top-I)$ has the −21.95 magnitude, even a
modest $D_h,D_q$ pushes $\rho(J_A)>1$ — this is the amp≈12 mechanism, and it is
**present at every step regardless of regime** (the small-basin blow-up under
$r<0.1$ is $\rho(J_A)>1$ with the *quasi-steady* $A^\top$, so the amplifier is the
Jacobian, not the physics). There is no zero block to break the coupling: this is
why A is structurally unstable rather than merely hard to train.

**Train-short / roll-long (constraint 4):** achieved *conditionally*. The gap
works because the physics bakes in a per-step relaxation to the forcing-set
attractor (h derived from a re-anchored q + implicit same-step feedback), so the
state never leaves the training distribution as the horizon grows — stable to
1906 steps trained on ≤10. But it holds only from benign starts; stiff high-flow
anchors escape the attractor and $\rho(J_A)>1$ takes over (amp≈12). So A **mostly**
satisfies constraint 4 and is the empirical proof the property is achievable, but
not unconditionally.

### 1.4 Regime of validity

- **CFL:** unconditionally stable in $h$ at $\theta=1$; the free-$q$ channel is
  the unresolved one and is what destabilises.
- **Quasi-steady:** the 1-hop gather $A^\top q$ is the *complete* upstream
  accumulation **only because** $r\ll1$ makes a neighbour's steady $q$ already
  equal to its full upstream sum. Holds on ~94–99.8 % of wet cells; degrades
  where $r\to1$.

### 1.5 Physics assumptions

Kinematic wave; quasi-steady 1-hop conservation; depth positivity floor;
`h` fully slaved to `q` (no independent storage dynamics — no wedge, no lag).

### 1.6 Cost

Per step: one SpMM $A^\top q$ (+ a second for the explicit term if $\theta<1$),
all else elementwise. **~1–2 sparse matvecs/step, zero iterations.** This is the
baseline all others are measured against.

### 1.7 Input/diagnostic variants of A (can it be rescued?)

The roles of *what is fed back* can be varied without changing the formulation.
The useful case is **making $h$ diagnostic-only** — keep $q$ as the free predicted
state but *remove $h$ from the decoder input* ($D_h=0$), so $h$ is computed by (A)
purely for output. This **block-triangularises** the Jacobian:

$$J_{A'}=\begin{bmatrix} D_q & 0 \\[2pt] \kappa(A^\top-I)D_q & I \end{bmatrix} \quad\Rightarrow\quad \mathrm{spec}(J_{A'}) = \mathrm{spec}(D_q)\;\cup\;\{1\}.$$

- **What it fixes (provably):** the stiffness $\kappa(A^\top-I)$ drops out of the
  eigenvalues entirely — the −21.95 amplifier is **structurally decoupled** from
  the rollout, exactly the amp≈12 loop. Same trick as C (dead-end one variable).
- **Why it still does not save A — two residual failures.**
  1. **The fragile channel is still the free driver.** Stability now rests on
     $\rho(D_q)\le1$, but $q$ is the orders-of-magnitude channel and $D_q$ is a
     free learned map with no structural bound — the MB-off overflow (hps2) *is*
     a free-$q$ map blowing up. Removed the loop, not the driver.
  2. **$h$ becomes an undamped integrator:** the bottom-right block is exactly
     $I$ (eigenvalue $1$), so any systematic $q$-bias makes $h$ **drift linearly
     in time** with no restoring force — conservation *forbids* a
     $\partial h/\partial h<1$ term.
- **Plus a physical cost:** discharge genuinely depends on storage
  ($q\propto h^{5/3}$); stripping $h$ from the input removes the one signal that
  would let $q$ self-regulate toward what storage supports.
- **Verdict — A′ is the mirror of C, dominated by C.** Both use the
  "diagnostic-not-fed-back" trick; A′ puts the **fragile** variable ($q$) in the
  driver's seat and forces the **tame** one ($h$) to be an *undampable*
  integrator (eig $=1$), whereas C carries the **tame** variable ($h$, dampable,
  eig $<1$ possible) and makes the fragile one an *exact* readout. If you are
  willing to make one variable diagnostic-only, make it $q$ (→ Option C), not
  $h$. A′ turns an exponential blow-up into a marginal drifter — better, but
  strictly worse than C.

---

## 2. Option B — **Muskingum–Cunge**: free routing params, $q$ pinned by a rating

### 2.1 Formulation

Swap the roles: **storage becomes the conserved state** ($h\equiv S/(wl)$) and
**discharge becomes a diagnostic of storage** with a residence time — a storage
closure, not a free output. The general storage–discharge relation ladder is:

$$Q_i = \frac{S_i}{K_i}\ \text{(linear reservoir)} \;\;\longrightarrow\;\; Q_i = k_i h_i^{m_i}\ \text{(Manning, }m=\tfrac53\text{)} \;\;\longrightarrow\;\; S_i = K_i\big[x_i Q_{\text{in},i} + (1-x_i)Q_{\text{out},i}\big]\ \text{(wedge).}$$

The **wedge** form (full Muskingum) discretises to the classic three-coefficient
reach law, which in network form (with $Q_\text{in}=A^\top Q_\text{out}$) is

$$(I - C_1 A^\top)\,Q_\text{out}^{t+1} = (C_2 A^\top + C_3 I)\,Q_\text{out}^{t}, \qquad C_1+C_2+C_3=1, \tag{B}$$

with $C_{1,2,3}(K,x,\Delta t)$. **If $K,x$ are static**, $C_1 A^\top$ is the same
nilpotent lower-triangular DAG operator as $A$, so its inverse is finite and
sparse and the whole step is **one precomputed matvec**:

$$Q_\text{out}^{t+1} = \underbrace{(I - C_1 A^\top)^{-1}(C_2 A^\top + C_3 I)}_{P_\text{MC}\ \text{(precompute once)}}\; Q_\text{out}^{t}.$$

Learned freedom (per constraint 1: decoder = "anything") goes into a **bounded
explicit residual** $r(S^t,z^t)$ added to $Q_\text{out}$ in the *source* term (so
the system stays linear in the unknown and needs no iteration), and optionally a
**bank of precomputed $P_\text{MC}^{(m)}$** at reference $(K,x)$ pairs with
decoder-emitted convex weights to approximate flow-dependent (variable-parameter)
routing at fixed cost.

#### ⟳ Iterative variant B-i: true variable-parameter Muskingum–Cunge

**Relaxing constraint 2.** The blended bank only *approximates* flow-dependent
$(K,x)$. The exact variable-parameter scheme recomputes $K_i,x_i$ each step from
the local celerity and discharge (e.g. $x_i=\tfrac12(1-Q_i/(B_i c_i S_{0,i}\Delta x))$),
so $C_{1,2,3}$ — and hence $M=I-C_1 A^\top$ — **change every step and cannot be
precomputed.** Because $C_1$ still multiplies the *nilpotent* $A^\top$, the linear
solve is a single **forward-substitution sweep in topological order**, not a true
iteration; but the coefficients themselves depend on the solution (celerity is a
function of $Q$), which closes a nonlinear loop resolved by **Picard iteration**:
seed $(K,x)$ from $Q^t$, sweep, recompute $(K,x)$ from the new $Q$, repeat. In
practice **2–3 Picard sweeps** suffice at a daily step because $(K,x)$ vary
slowly relative to $Q$. Conservation is preserved at every sweep ($\sum C=1$
holds coefficient-by-coefficient); only the *rating timing* is iterated to
convergence. This buys the genuine $r>1$ floodplain regime that the static bank
only brackets.

#### ⟳ Iterative variant B-ii: nonlinear stage–discharge rating

**Relaxing constraint 2.** Using the physical Manning rating $Q_i=k_i h_i^{5/3}$
(or a learned power law $k_i h_i^{m_i}$) as the storage closure makes continuity
*nonlinear* in $S^{t+1}$. Solve implicitly by **Newton iteration**: each Newton
step evaluates $\partial Q/\partial S=(5/3)Q/S$ and solves one linear triangular
system (a sweep) against the current Jacobian. **2–4 Newton steps** reach a tight
tolerance because the rating is smooth and monotone (guaranteed if
$k>0,\ m\ge1$). Conservation is exact **only at convergence** (or if the final
iterate is projected back onto continuity); a truncated Newton solve leaves a
mass residual of order the step tolerance — so if this variant is used, either
iterate to a conservation tolerance or add a final continuity projection. This
variant recovers the convex high-flow rating (steep drainage at peak stage) that
the linear reservoir misses.

### 2.2 Conservation

Exact: $C_1+C_2+C_3=1$ **is** the discrete mass-balance condition; the network
telescoping (C′) holds for any static coefficients. A blended bank is
conservative iff each member satisfies $\sum C=1$ and the weights are convex.

### 2.3 Stability

- $M=I-C_1A^\top$ is a diagonally-dominant **M-matrix**, so $\rho(P_\text{MC})<1$
  **unconditionally, for any Courant number** — the contraction is *structural*,
  not learned. This is the strongest stability guarantee of the three.
- **Precompute-time constraint:** require $C_2\ge0$ (i.e. $\Delta t/K\ge 2x$) to
  avoid the Muskingum negative-coefficient oscillation; reject violating
  $(K,x)$ pairs when building $P_\text{MC}$. No runtime cost.
- The bounded residual $r$ cannot create an amplifying eigenvalue (clamped), so
  learning does not spoil the contraction.

#### Jacobian / spectrum

The carried state is $S$ ($\equiv h$); $q=Q_\text{out}$ is the diagnostic. The
linear step is $S^{t+1}=P_\text{MC}\,S^{t}+P_\text{MC}(\text{source}(z^t,r))$, so
the homogeneous rollout Jacobian is **exactly the precomputed operator**

$$J_B=\frac{\partial S^{t+1}}{\partial S^{t}}=P_\text{MC}=(I-C_1A^\top)^{-1}(C_2A^\top+C_3I).$$

Because $A^\top$ is nilpotent (strictly lower-triangular on the DAG), its
eigenvalues are all $0$, so $\mathrm{spec}(C_1A^\top)=\{0\}$ and
$\mathrm{spec}(A^\top)=\{0\}$ — hence $\mathrm{spec}(P_\text{MC})=\{C_3\}$ with
$C_3=1-C_1-C_2\in[0,1)$ whenever the $C_2\ge0$ / $C_1,C_2$ conditions hold. So
$\rho(J_B)=C_3<1$ **by construction, independent of the network and of Cr** — a
nilpotent-plus-diagonal spectrum is the algebraic reason the M-matrix contracts.
The learned residual enters only through the *source* (a bounded additive term),
not through $J_B$, so it **cannot move the spectrum** — the contraction is
provably learning-proof. This is the one option whose $\rho(J)<1$ is a theorem
rather than a hope.

**Train-short / roll-long (constraint 4):** **satisfied structurally** — the
strongest of the three. $\rho(J_B)=C_3<1$ is a horizon-independent theorem that
does not depend on the network, so a model trained on 10 steps contracts
identically at 10 000 steps. This is the *only* option that guarantees the gap
rather than earning it in training.

> **⟳ iterative variants:** for B-i/B-ii the coefficients depend on the state, so
> $J$ picks up a $\partial C/\partial S$ term and the clean $\{C_3\}$ spectrum
> becomes state-dependent; contraction then holds only where the local $C_3<1$
> and the Picard/Newton loop has converged. Still far from A's amplifier, but no
> longer a single closed-form eigenvalue.

### 2.4 Regime of validity

- **CFL:** unconditionally stable (implicit M-matrix), any Cr.
- **Quasi-steady:** **widest of the three.** The wedge term makes outflow depend
  on inflow, so it captures **wave translation** and **loop-rating / hysteresis**
  — the leading-order $r>1$ unsteady behaviour a single-valued $g(S)$ cannot
  represent. Constant-parameter form covers up to moderate unsteadiness; the true
  $r>1$ floodplain regime needs variable $(K,x)$ or a nonlinear rating —
  approximated at fixed cost by the blended bank, or captured exactly by the
  **⟲ iterative variants B-i / B-ii** (2–4 inner solves/step) if constraint 2 is
  relaxed.

### 2.5 Physics assumptions

Muskingum–Cunge routing (kinematic wave + diffusive wedge); **constant**
(or bank-discretised) $K,x$; rating relation between storage and discharge; the
daily-averaging bias of the nonlinear rating (§7 condition 4) absorbed into the
bounded residual.

### 2.6 Cost

- **Static / bank (no relaxation):** **one SpMV** ($P_\text{MC}$, precomputed,
  ~2.6 M nnz on the full basin) for the single-operator form; **$m$ SpMVs** for
  an $m$-member blended bank (m≈3–4); plus one cheap gather for the residual.
  **Zero iterations.** Precompute $P_\text{MC}$ once at model-build (a sparse
  triangular inverse per reference $(K,x)$).
- **⟳ B-i variable-parameter (relaxes constraint 2):** **no precompute**
  possible; each step is **2–3 Picard sweeps**, each sweep a topological
  forward-substitution ($O(\text{diameter})$ sequential sparse ops, or `diameter`
  SpMVs if expressed as a truncated Neumann apply). Cost ≈ **2–3 × (a triangular
  solve)** per step — roughly the same order as a handful of SpMVs on the small
  basin, but the sequential depth (585 hops) makes it markedly less GPU-friendly
  on the full basin.
- **⟳ B-ii nonlinear rating (relaxes constraint 2):** **2–4 Newton steps** per
  step, each one triangular solve against the current Jacobian, plus the rating
  and its derivative (elementwise). Cost ≈ **2–4 × (a triangular solve)** per
  step, same sequential-depth caveat, plus a convergence/projection check to keep
  conservation exact.

---

## 3. Option C — **Continuity readout**: free $h$ (i.e. $\mathrm{d}S/\mathrm{d}t$), $q$ diagnosed exactly

### 3.1 Formulation

The dual of Option B. The network predicts **depth** $h^{t+1}$ (equivalently
$\mathrm{d}S/\mathrm{d}t$, since $S=h\,wl$); discharge is then **uniquely
determined by continuity** with the known lateral inflow — no rating, no g(S):

$$(I - A^\top)\,Q_\text{out} = I_\text{lat} - \frac{\mathrm{d}S}{\mathrm{d}t} \quad\Longrightarrow\quad \boxed{\,Q_\text{out} = (I - A^\top)^{-1}\Big(I_\text{lat} - \tfrac{\mathrm{d}S}{\mathrm{d}t}\Big)\,}. \tag{C-readout}$$

$(I-A^\top)^{-1}$ is the **undamped flow-accumulation matrix** — finite, sparse,
static, **precomputed once** (nilpotent $A^\top$ on the DAG). `h` is the primary
predicted state (through a positivity transform, e.g. softplus); `q` is a **pure
readout** that feeds nothing forward — the rollout operator becomes **h→h only**.

Decompose the readout to see its structure:

$$Q_\text{out} = \underbrace{(I-A^\top)^{-1} I_\text{lat}}_{\text{steady accumulation of known forcing}} \;-\; \underbrace{(I-A^\top)^{-1}\tfrac{\mathrm{d}S}{\mathrm{d}t}}_{\text{unsteady correction (network)}}.$$

Since $|\mathrm{d}S/\mathrm{d}t|/Q = r \approx 4\times10^{-4}$–$2\times10^{-3}$,
**~99.8 % of $Q$ is the exact accumulation of the known lateral forcing** and the
network supplies only the sub-percent unsteady correction (rising to O(1) exactly
at the $r>1$ flood fronts where a learned correction is wanted).

### 3.2 Conservation

Exact by construction: (C-readout) *is* continuity (C) rearranged; every
$Q_\text{out}$ that exists satisfies the balance identically. No positivity floor
needed on the flux (a softplus on `h` keeps storage ≥ 0 directly).

### 3.3 Stability

- **`q` leaves the dynamical loop.** The amp≈12 mechanism of Option A *cannot
  exist* because `q` is a deterministic readout of `h`'s history, not a fed-back
  degree of freedom, and the q→h stiffness (−21.95θ) has no loop to act in.
- The h→h map is a **free learned operator** — stability is **empirical, not
  structural** (unlike Option B's M-matrix guarantee). The bet: a free **h**
  predictor is far tamer than a free **q** predictor, because `h` is the bounded,
  low-dynamic-range, smooth channel and no longer inherits `q`'s amplified error.
  Physics supports the bet (h is the tame channel) but does not prove it.
- **Undamped spatial error accumulation:** $(I-A^\top)^{-1}$ sums
  $\mathrm{d}S/\mathrm{d}t$ over the whole upstream network with coefficient 1
  (no decay), so a systematic bias in predicted $\mathrm{d}S/\mathrm{d}t$
  accumulates *linearly* toward the outlet (~8000 cells, full basin). This is a
  bounded-per-step **spatial** accumulation, not a geometric **temporal**
  blow-up — much milder than Option A's failure — but high-upstream-area `q`
  accuracy depends on basin-wide `h` fidelity. A loss placed directly on `q`
  ties basin-wide $\mathrm{d}S/\mathrm{d}t$ back to observed discharge and
  controls it.

#### Jacobian / spectrum

**C is not unconditionally stable** — it removes A's *specific* amplifier, not
all blow-up risk. The carried state is $h$; $q$ is diagnostic. With
$h^{t+1}=F(h^t,f)$ the rollout Jacobian is simply

$$J_C=\frac{\partial h^{t+1}}{\partial h^t}=\partial_h F,$$

and $q$ contributes **no** term because it is not fed back — the entire $q$-row
and $q$-column of the block Jacobian are zero. So $q$'s error, which does enter
through $(I-A^\top)^{-1}$, is a **dead end**: it appears once in the output and
never seeds the next step. Blow-up is governed purely by $\rho(\partial_h F)$,
living on the **tame** channel. The residual risk is real and lives in two
places:

- **Increment vs absolute — the actual blow-up knob.** If $F$ predicts $h^{t+1}$
  *directly* (absolute), $\partial_h F$ is free and can be a contraction. If $F$
  integrates a predicted rate (increment,
  $h^{t+1}=h^t+\kappa\,g(h^t)$), then
  $$\partial_h F = I + \kappa\,\partial_h g \;\ge\; I \quad\text{unless the net learns damping }\partial_h g<0,$$
  so a biased rate becomes a **random-walk drift in storage**. The
  "increment beats absolute" finding was measured under A (h slaved to q) and
  **may invert here** — under C, increment is the *less* stable choice.
- **Spatial gain in the readout** ($(I-A^\top)^{-1}$, coefficient-1 accumulation)
  — bounded per step, linear in space, not compounding in time on its own.

So the honest claim is: C **downgrades** the worst case from *geometric-in-time*
(A) to *empirical h-map Jacobian + linear-in-space readout*, and **removes the
demonstrated −21.95 amplifier** — but does not deliver B's structural guarantee.

**Train-short / roll-long (constraint 4):** **not guaranteed — the weak point of
pure C.** Because $q$ is a dead-end diagnostic it provides *zero* feedback to
$h$, so nothing structural bounds $h$ over 1000s of steps; the gap survives only
if the learned $\partial_h F$ is a contraction, a *training bet* weaker than A's
baked-in attractor. Two sub-cases (see §6): **absolute-$h$** can approximately
recover the gap if the net learns $\partial_h F\approx0$ (re-anchor to forcing
each step); **increment-$h$** ($\partial_h F=I+\kappa\partial_h g\ge I$) is an
undamped integrator and **breaks** the gap — a slow drift visible only at
multi-year horizons. Pure C therefore needs an *engineered* restoring force
(anchored/relaxation $h$, §6) or the B/C hybrid to meet constraint 4.

#### Input/diagnostic variant: feeding $q$ back as an input

Making `q` a decoder **input** (while it remains the conservation-pinned output)
is a design choice that **re-opens the severed feedback path**. Since
$q^t=\Phi(h^t,\dots)=(I-A^\top)^{-1}(I_\text{lat}-\mathrm{d}S/\mathrm{d}t)$ is a
function of the state, the chain rule adds a term to $J_C$:

$$J_{C+q}=\partial_h F \;+\; \partial_q F\cdot\underbrace{(I-A^\top)^{-1}\!\left(-\frac{\partial\,\mathrm{d}S/\mathrm{d}t}{\partial h}\right)}_{\text{reintroduced path, carries the accumulation gain}}.$$

- The new term carries the **network-wide accumulation gain** of
  $(I-A^\top)^{-1}$ (large at high-upstream-area cells) — the very direction that
  was previously a dead end now feeds forward. Its size is $\partial_q F$,
  **learned**, so q-stability drops from *structurally absent* (pure C) to
  *learned-to-be-small*.
- It is **not** a return to A: the *output* $q$ is still the exact continuity
  readout, and the fixed −21.95 stiffness stays gone; the reintroduced gain is
  learned and bounded-by-training, not a fixed multiplier applied every step.
  Ordering: **A** (q is state, fixed amplifier) $\gg$ **C+q-input** (q is input,
  learned gain) $>$ **pure C** (no path).
- **When to include it anyway:** `q` compactly encodes flow-regime / upstream-
  area context that can *improve* $\partial_h F$. Do it **defensively** —
  small/regularised q-input weights or a stop-gradient on the q-input during
  early curriculum — so $\partial_q F\approx0$ (recovering pure C's safety) unless
  the net demonstrably earns the dependence.

### 3.4 Regime of validity

- **CFL:** the readout itself is exact at any Cr (it is an algebraic identity,
  not a time advance); stability rests on the learned h-dynamics rather than a
  Courant condition.
- **Quasi-steady:** does **not require** quasi-steadiness for *conservation*
  (the identity is exact), but the *accuracy* split above means it degrades
  gracefully — the dominant steady-accumulation term is always exact, and only
  the small unsteady correction must be learned well, precisely where the
  network should focus ($r\to1$ floodplains).

### 3.5 Physics assumptions

**Minimal — accounting only.** Continuity + known lateral inflow + a
column-conservative `A`. **No routing physics** in the operator: wave celerity,
attenuation, translation and hysteresis are *all* delegated to the learned
h-dynamics. This is the fewest assumptions of the three and the most that is
asked of the network.

### 3.6 Cost

Per step: **one SpMV** ($(I-A^\top)^{-1}$, precomputed, same sparsity class as
$P_\text{MC}$) plus a positivity transform. **Zero iterations, zero learned
parameters in the operator.** Cheapest operator of the three; precompute the
accumulation inverse once.

**No iterative variant is needed or useful here:** the continuity readout is an
exact linear identity, so relaxing constraint 2 buys Option C nothing — it stays
a single precomputed SpMV regardless. (The only nonlinearity, the softplus on
`h`, is elementwise and outside the solve.) This is a structural advantage of C:
it reaches full fidelity *without* the iteration that Options B-i/B-ii need,
because it delegates routing physics to the learned h-dynamics rather than to an
in-layer nonlinear solve.

---

## 4. Side-by-side comparison

| | **A — current** | **B — Muskingum–Cunge** | **C — continuity readout** |
|---|---|---|---|
| network predicts (free) | $q^{t+1}$ | routing params + bounded residual | $h^{t+1}$ ($\equiv \mathrm{d}S/\mathrm{d}t$) |
| pinned by conservation | $h$ | $q$ (via rating) | $q$ (exact) |
| primary state in loop | $q$ | $S$ ($\equiv h$) | $h$ |
| mass conservation | exact (– floor) | exact | exact |
| carried state / diagnostic | state $q$; diag $h$ | state $S{\equiv}h$; diag $q$ | state $h$; diag $q$ |
| rollout Jacobian $J$ | $\begin{smallmatrix}D_q & D_h\\ \kappa(A^\top{-}I)D_q & I{+}\kappa(A^\top{-}I)D_h\end{smallmatrix}$ | $P_\text{MC}$ | $\partial_h F$ |
| $\rho(J)$ | $>1$ (stiffness×$D_{q,h}$) — **blows up** | $\{C_3\}<1$ **provably** | $\rho(\partial_h F)$ — empirical, tame |
| stability guarantee | none (free $q$) — **fails, amp≈12** | **structural M-matrix, any Cr** | empirical (free $h$, tame channel) |
| CFL regime | uncond. stable in $h$; free $q$ unresolved | unconditional | identity (exact); stability via learned $h$ |
| quasi-steady regime | needs $r\ll1$ (1-hop = full accum.) | **widest** (wedge → loop rating, $r\!\to\!1$) | conservation exact ∀$r$; accuracy graceful |
| routing physics | 1-hop kinematic, $h$ slaved | Muskingum wedge (translation+attenuation) | **none** (all learned) |
| dominant-$Q$ exactness | — | rating-approximate | **~99.8 % exact** (forcing accumulation) |
| positivity floor needed | yes (small break) | no (if $g(0)=0$) | no (softplus $h$) |
| operator cost / step | 1–2 SpMM | **1 SpMV** static / $m$ bank / **2–4 solves** if ⟳ | **1 SpMV** |
| learned params in operator | — | $K,x$ (static) / bank weights / per-step ⟳ | **none** |
| iterations | 0 | **0** static / bank; **2–3 Picard (B-i)** or **2–4 Newton (B-ii)** if relaxed | 0 |
| precompute | — | $P_\text{MC}$ (+ bank), $C_2\ge0$ check; **none if ⟳** | $(I-A^\top)^{-1}$ |
| widest regime needs iteration? | — | **yes** ($r>1$ ⇒ B-i/B-ii) | **no** (learned $h$) |
| **train-short / roll-long (constraint 4)** | **conditional** (attractor baked in; escapes from stiff starts) | **structural** ($\rho{=}C_3{<}1$, horizon-independent theorem) | **not guaranteed** (learned $\partial_h F$; increment-$h$ breaks it) |

### Reading the table

- **A is the diagnosed failure:** the only option with a *free discharge* in the
  loop, and the amplification is a property of that free-$q$ Jacobian × the
  σ_q/σ_h stiffness — demonstrated on the quasi-steady small basin.
- **B is the maximum-physics / maximum-stability option:** the wedge covers the
  widest regime (including loop-rating unsteadiness) and the M-matrix gives an
  unconditional contraction, at the price of more machinery and `q` pinned to a
  rating whose daily-averaged validity is loose (§7).
- **C is the minimum-assumption option:** removes `q` from the loop entirely and
  makes the bulk of `q` exact, at the price of trading B's *structural* stability
  guarantee for the (physically well-motivated) *empirical* bet that a free `h`
  predictor is well-behaved.

**A and C are mutually exclusive closures** (predicting both a Q-relation and
$\mathrm{d}S/\mathrm{d}t$ over-determines continuity). A hybrid is possible: use
$P_\text{MC}$ (B) as a **stability prior / initialisation** for the h-dynamics
while diagnosing `q` by the continuity readout (C).

### 4.1 The input/diagnostic design axis (orthogonal to the closure choice)

Independently of *which closure* is used, each of $q$ and $h$ can play three
roles: **carried state** (predicted and fed back — has rows/columns in $J$),
**decoder input only** (informs the prediction but is itself conservation-derived
— adds a chain-rule term to $J$), or **pure diagnostic** (computed for output,
fed nothing forward — *absent* from $J$). The Jacobian rule from §0.1 makes the
stability consequence mechanical: **a variable in the state can compound its
error; a pure diagnostic cannot; an input-only variable can compound only through
a learned $\partial(\cdot)F$ gain.**

| variable roles | closure | $J$ structure | spectrum / risk |
|---|---|---|---|
| $q$ state, $h$ input | **A** (current) | full $2\times2$, stiffness in bottom row | $\rho>1$ — amp≈12 blow-up |
| $q$ state, $h$ **pure diagnostic** | **A′** (§1.7) | block-triangular | $\mathrm{spec}(D_q)\cup\{1\}$ — drift, dominated by C |
| $h$ state, $q$ **pure diagnostic** | **C** (pure) | $\partial_h F$ only | tame; $q$-error is a dead end |
| $h$ state, $q$ **input** | **C+q-input** (§3.3) | $\partial_h F + \partial_q F\,\Phi'$ | tame + *learned* reintroduced gain |
| $S{\equiv}h$ state, $q$ diagnostic-by-rating | **B** | $P_\text{MC}$ | $\{C_3\}<1$ structural |

Three rules of thumb fall out:

1. **Put the *tame* variable ($h$) in the state and make the *fragile* one ($q$)
   the diagnostic** — this is the single most important lever, and it is exactly
   what separates C (good) from A′ (dominated). Roles matter more than the
   closure algebra.
2. **A pure diagnostic is "free" stability-wise** (no $J$ entry) but forfeits its
   value as *context* for the prediction; feeding it back as an **input** buys
   context at the price of a *learned* Jacobian term — acceptable if that term is
   regularised toward zero (defensive init / stop-gradient).
3. **Never make the fragile variable a fed-back *state* with an algebraic
   amplifier in the same loop** — that is A, and the −21.95 stiffness guarantees
   $\rho(J)>1$.

### 4.2 Option B/C — the hybrid (recommended)

The one-line aside above is important enough to state as a first-class option,
because it is the **only design that satisfies every constraint simultaneously**,
including the train-short / roll-long gap (constraint 4) that pure C fails.

**Construction.** Carry $h$ ($\equiv S$) as the state; advance it with the
**Muskingum backbone $P_\text{MC}$** (Option B) so the h-channel inherits the
structural contraction $\rho(J)=C_3<1$; add the decoder's freedom as a **bounded
residual in the source** (cannot move the spectrum); and **diagnose $q$ by the
continuity readout** (Option C) so the fragile discharge is *never a state and
never fed back*.

$$S^{t+1}=\underbrace{P_\text{MC}\,S^{t}}_{\text{contracting backbone (B)}}+\underbrace{P_\text{MC}\,\text{source}(z^t,r)}_{\text{bounded learned residual}},\qquad q^{t+1}=\underbrace{(I-A^\top)^{-1}(I_\text{lat}-\tfrac{\mathrm{d}S}{\mathrm{d}t})}_{\text{exact readout (C), out of the loop}}.$$

**Why it wins on every axis:**
- **Constraint 4 (the decisive one):** the backbone gives $\rho(\partial_h F)=C_3<1$
  as a horizon-independent theorem — train on 10 steps, contract at 10 000. It
  delivers *structurally* what current A only approximates and pure C only
  bets on.
- **Fragile channel removed:** $q$ is the exact continuity readout, so it cannot
  overflow at any horizon (inherits C's key win), and the −21.95 stiffness loop
  never exists.
- **Conservation:** exact (both B backbone and C readout are exact; the residual
  enters conservatively via the source).
- **Cost:** two precomputed SpMVs/step ($P_\text{MC}$ and $(I-A^\top)^{-1}$),
  zero iterations.
- **Regime:** the backbone carries real routing physics (translation +
  attenuation) so it degrades gracefully toward $r\to1$; the residual + optional
  bank extend it further.

**Cost of the hybrid:** it reintroduces Option B's rating assumption *on the
backbone* (the $P_\text{MC}$ coefficients encode a storage–discharge relation),
so it is not as assumption-light as pure C — but that is precisely the price of
turning the train/rollout gap from a bet into a theorem. This is the recommended
target architecture.

---

## 5. Open questions / decision points

- **Primary bet:** structural stability (B) vs minimal assumptions + `q` out of
  the loop (C). C attacks the *diagnosed* mechanism most directly and is cheapest;
  B is safest by construction.
- **For C:** is `q` a **pure readout** (decoder ignores input `q`) or a
  **redundant input feature**? And does the "increment beats absolute" `h`
  finding (measured under A, where `h` was slaved to `q`) still hold when `h` is
  the free state?
- **For B:** single static $P_\text{MC}$ prototype vs straight-to-bank; choice of
  reference $(K,x)$ grid; whether the bounded residual is one- or two-sided; and
  **whether to relax constraint 2** for the variable-parameter (B-i) or nonlinear
  rating (B-ii) variants — i.e. whether the $r>1$ floodplain regime is worth
  2–4 inner solves/step (and the sequential-depth cost on the full basin) versus
  the fixed-cost bank approximation.
- **Shared:** all three need the loss placed (at least partly) on `q` to control
  the respective accumulation/rating error; the volume-budget diagnostic
  (proposed separately) is the natural monitor for whichever is chosen.

---

## 6. Preserving the train-short / roll-long gap (constraint 4)

The current MB's most valuable property is that a model **trained on ≤10-step
rollouts stays stable over multi-year (1000s-step) inference** (stable to 1906
steps). Any redesign must inherit this. This section explains *why* it holds,
states the requirement precisely, and checks each option — it is the
consolidation referenced from the per-option §_.3.

### 6.1 Why current MB generalises past its training horizon

It is **not** that the learned map is robust to horizon. It is two
horizon-independent structural facts:

1. **The quasi-steady forcing attractor.** Because $r\ll1$, the true state each
   step is slaved to *that day's forcing*: $q\approx(I-A)^{-1}q_\text{lateral}$,
   $h$ the consistent storage. Every step is a **bounded relaxation toward a
   manifold set by the forcing**, not a free extrapolation. Crucially, **forcing
   at step 1000 is as in-distribution as forcing at step 5** (weather is
   weather), so the target state never leaves the training distribution no matter
   how long the rollout — the horizon is long but the *state stays in-domain*.
   Current MB bakes this in by *deriving* $h$ from a re-anchored $q$.
2. **Implicit same-step negative feedback** (θ=1 backward Euler; stability notes
   §8) damps within-step overshoot every step, independent of count.

### 6.2 The requirement, stated precisely

> Constraint 4 holds **iff every step applies a restoring force toward the
> forcing-determined, in-distribution attractor** — i.e. the state-update's
> Jacobian on the carried channel is a contraction *by construction*
> ($\rho(\partial_h F)<1$), not merely at training horizons. Equivalently: the
> state must be **re-anchored to (in-distribution) forcing each step**, so error
> cannot accumulate over the horizon gap between train (≤10) and inference (1000s).

### 6.3 Check against each option

| option | restoring force on carried channel | constraint 4 |
|---|---|---|
| **A** (current) | attractor baked in via $h$-from-$q$ + implicit feedback | **conditional** — holds from benign starts (proof it's achievable), escapes from stiff high-flow anchors ($\rho(J_A)>1$) |
| **A′** ($h$ diagnostic) | none on $q$; $h$ self-block $=I$ | **fails** — $h$ is an undamped integrator, drifts over long horizons |
| **C pure, absolute-$h$** | only if net learns $\partial_h F\approx0$ | **soft** — recoverable but a training bet, not structural |
| **C pure, increment-$h$** | $\partial_h F=I+\kappa\partial_h g\ge I$ | **fails** — integrator drift visible only at multi-year horizons |
| **C + anchored/relaxation $h$** | engineered $\partial_h F<1$ (below) | **holds** (engineered) |
| **B** (Muskingum) | $\rho(J_B)=C_3<1$ theorem | **structural** — horizon-independent, network-independent |
| **B/C hybrid** (§4.2) | $P_\text{MC}$ backbone $C_3<1$ | **structural** *and* $q$ out of the loop |

### 6.4 How to get the gap structurally while keeping $q$ out of the loop

$q$ must stay diagnostic (for the stability win), so the restoring force must live
in the **h-channel** without using $q$ as feedback. Two ways:

1. **Anchored / relaxation $h$-prediction.** Predict a *bounded deviation from a
   physics reference*: $h^{t+1}=h_\text{ref}(\text{forcing})+\text{clamp}(\text{net})$,
   or a relaxation form $h^{t+1}=h^t+\kappa\big(g(h^\star)-g(h^t)\big)$ with
   $h^\star$ a forcing-set target. Both make $\partial_h F<1$ structurally (a
   bounded correction around a re-anchored reference), so the attractor is built
   in and $q$ is still never fed back. This upgrades **pure C** from "soft" to
   "holds".
2. **The B/C hybrid (§4.2) — recommended.** The Muskingum backbone $P_\text{MC}$
   supplies $\rho=C_3<1$ as a *theorem* (horizon- and network-independent),
   diagnosing $q$ by continuity. This is the only option that gives constraint 4
   as a guarantee **and** removes the fragile discharge from the dynamics.

### 6.5 Consequence for the redesign

- **Absolute-$h$ (or anchored-$h$) is strongly preferred over increment-$h$**
  under any $h$-state option; increment integrates rate error and specifically
  breaks the multi-year gap. (The "increment beats absolute" result was measured
  under A, where $h$ was slaved to $q$, and does **not** transfer.)
- **The multi-year with/without-MB validation** the current setup relies on is
  preserved by A (conditionally), B, the hybrid, and anchored-C; it is **broken**
  by A′ and increment-pure-C. Whichever option is prototyped, the multi-year
  rollout is the diagnostic that would expose an integrator drift a short
  training rollout hides — so it must be run past the training horizon.

---

## 7. References

- Kinematic wave & celerity $c=\tfrac53 v$: Ponce & Simons, *J. Hydraul. Div.
  ASCE* 103 (1977).
- Muskingum–Cunge constant/variable parameter routing: Cunge, *J. Hydraul. Res.*
  7 (1969); Ponce, *Engineering Hydrology* (1989), ch. 9.
- CFL condition: Courant, Friedrichs & Lewy, *Math. Ann.* 100 (1928).
- Companion notes: [mass_balance_stability_notes.md](mass_balance_stability_notes.md)
  (θ-method, regime analysis §6, rating validity §7),
  [mass_balance_notes.md](mass_balance_notes.md) (implemented hard constraint).
