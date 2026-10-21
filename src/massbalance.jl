using Flux
using Functors
using GraphNeuralNetworks
using SparseArrays
using Statistics

"""
    MassBalanceFormulation

Abstract supertype of the mass-balance formulations a [`WflowGNN`](@ref) can be
built around. The model never branches on *whether* it has a mass balance; it
dispatches on the formulation instead. A formulation implements:

- `decoder_outputs(f, n_state)`   : number of state channels the decoder predicts
- `extra_decoder_features(f, g, state, forcing)` : extra decoder inputs, a tuple of
                                   matrices (`()` for none)
- `extra_decoder_dim(f)`          : total row count of `extra_decoder_features`
- `floor_state(f, g, raw)`        : physical-space floor of the raw predicted q
- `apply_formulation(f, g, state, forcing, forcing_next, Δ)` : next full state
- `has_mass_balance(f)`           : `true` if it constrains `river_h`
- `mb_h_loss_weight(f, h_loss_scale)`: h-loss weight implied by its stiffness (or `nothing`)
- `precompute_batched(f, B)`, `Flux.gpu`, `Flux.cpu`

Concrete formulations: [`NoMassBalance`](@ref), [`ThetaKinematicWave`](@ref).
"""
abstract type MassBalanceFormulation end

"""
    NoMassBalance()

Null-object formulation: no physics constraint. The model predicts the full state
increment directly (`state .+ decoder(h)`), with no `river_h` hard constraint.
Used for non-river domains and whenever the mass balance is disabled.
"""
struct NoMassBalance <: MassBalanceFormulation end

"""
    ThetaKinematicWave

A non-trainable layer that enforces the kinematic-wave mass balance as a hard
constraint in the river routing forward pass.

Given the predicted normalised discharge `q_norm_new` it computes the new
water depth `h_norm_new` deterministically using a θ-weighted (mixed
implicit/explicit) time discretisation of the kinematic-wave storage equation:

    h_phys_new = h_phys_old
               + dt/(w·l) · [ θ·(ΣQ_up[t+1] + Q_iw[t+1] − Q_out[t+1])          # implicit
                            + (1−θ)·(ΣQ_up[t] + Q_iw[t] − Q_out[t]) ]          # explicit

where `1/(w·l) = postscale_h / postscale_q` and all Q values are in m³/s.

- `θ = 1` recovers the fully-implicit (backward-Euler) scheme — the historical
  default. `h[t+1]` then depends on the predicted `q[t+1]` with full weight, so
  the mapping `∂h_norm/∂q_norm = −θ·dt·σ_q/σ_h` has its largest magnitude and the
  rollout is most prone to h-overshoot amplification (see
  `docs/mass_balance_stability_notes.md`).
- `θ = 0` is fully-explicit (forward-Euler): `h[t+1]` uses only the known state
  at `t`, decoupling it from the current prediction.
- `0 < θ < 1` blends the two; `θ = 0.5` is Crank–Nicolson. Lowering θ shrinks the
  stiff gradient gain by the factor θ, damping overshoots at the cost of a small
  temporal lag in the storage response.

Fields (all per-node constants, not optimised):
- `postscale_q`   : upstream area per node [m²] (inverse of the river_q pre-scaling)
- `postscale_h`   : `a/(w·l)` per node (inverse of the river_h pre-scaling)
- `μ_q`, `σ_q`   : per-node z-score statistics of (scaled) river_q
- `μ_h`, `σ_h`   : per-node z-score statistics of (scaled) river_h
- `μ_inwater`, `σ_inwater` : per-node z-score statistics of river_inwater
- `postscale_inwater` : per-node multiplier that turns the denormalised lateral
                    inflow into m³/s. Preprocessing divides `river_inwater` by the
                    upstream area `a` before the z-score (like `river_q`), so this
                    is `a` (`postscale["river_inwater"]`); all-ones reproduces the
                    pre-R5b behaviour that treated the area-scaled inflow as m³/s
- `dt`            : model timestep in seconds
- `θ`             : implicitness weight in `[0, 1]` (default `1` = fully implicit)
- `smooth_h_floor`: if `true`, uses a smooth non-negative floor for `h_phys_new`
- `h_floor_softness`: smooth-floor softness in physical meters (`> 0`)
"""
struct ThetaKinematicWave{V <: AbstractVector{Float32}} <: MassBalanceFormulation
    postscale_q       :: V
    postscale_h       :: V
    ph_over_pq        :: V   # postscale_h ./ postscale_q — precomputed constant
    μ_q               :: V
    σ_q               :: V
    μ_h               :: V
    σ_h               :: V
    μ_inwater         :: V
    σ_inwater         :: V
    postscale_inwater :: V   # per-node inflow → m³/s multiplier (upstream area, or ones)
    dt                :: Float32
    A_routing         :: AbstractMatrix{Float32}  # (N×N) upstream adjacency, no self-loops
    A_routing_batched :: Union{Nothing, AbstractMatrix{Float32}}  # block-diagonal (B·N×B·N)
    batch_size        :: Int                                       # B for A_routing_batched; 0 = none
    θ                 :: Float32                                    # implicitness weight ∈ [0,1]
    smooth_h_floor    :: Bool
    h_floor_softness  :: Float32
end

# Pre-R2 name of the θ-kinematic-wave formulation, kept for compatibility.
const MassBalanceLayer = ThetaKinematicWave

# Non-negative floor in physical space. Default is hard clamp; optional
# softplus branch preserves a small gradient near dry-channel conditions.
function _nonnegative_floor(x, smooth::Bool, softness::Float32)
    if !smooth
        return max.(0f0, x)
    end
    β = max(Float32(softness), 1f-6)
    z = x ./ β
    # Numerically stable softplus WITHOUT a positive-side cap:
    #   softplus(z) = max(z, 0) + log1p(exp(-|z|)).
    # The exp argument is always ≤ 0, so it cannot overflow; for large z the
    # output → β·z = x (no artificial ceiling). A naive `β·log1p(exp(clamp(z,-40,40)))`
    # instead capped h at β·40 (e.g. 2.0 m at softness 0.05) — an implementation bug.
    return β .* (max.(z, 0f0) .+ log1p.(exp.(.-abs.(z))))
end

Flux.@layer ThetaKinematicWave
Flux.trainable(::ThetaKinematicWave) = (;)  # physics constants, not optimised

# Per-node constant `v` as a `(1, n)` row, tiled over the `n_rep` graphs of a batch
# (a zero-copy reshape when `n_rep == 1`, i.e. always during rollout).
_tile_node_const(v::AbstractVector, n_rep::Int, n::Int) =
    n_rep == 1 ? reshape(v, 1, n) : reshape(repeat(v, n_rep), 1, n)
# Restrict Functors traversal to the per-node vectors; exclude A_routing
# (not writable in-place on GPU). Device transfer is handled by the explicit
# Flux.gpu / Flux.cpu overloads below.
Functors.@functor ThetaKinematicWave (
    postscale_q, postscale_h, ph_over_pq,
    μ_q, σ_q, μ_h, σ_h, μ_inwater, σ_inwater, postscale_inwater,
)
function Flux.gpu(l::ThetaKinematicWave)
    ThetaKinematicWave(
        Flux.gpu(l.postscale_q),
        Flux.gpu(l.postscale_h),
        Flux.gpu(l.ph_over_pq),
        Flux.gpu(l.μ_q), Flux.gpu(l.σ_q), Flux.gpu(l.μ_h), Flux.gpu(l.σ_h),
        Flux.gpu(l.μ_inwater), Flux.gpu(l.σ_inwater), Flux.gpu(l.postscale_inwater), l.dt,
        _to_cusparse(l.A_routing),
        isnothing(l.A_routing_batched) ? nothing : _to_cusparse(l.A_routing_batched),
        l.batch_size,
        l.θ,
        l.smooth_h_floor,
        l.h_floor_softness,
    )
end

function Flux.cpu(l::ThetaKinematicWave)
    ThetaKinematicWave(
        Flux.cpu(l.postscale_q),
        Flux.cpu(l.postscale_h),
        Flux.cpu(l.ph_over_pq),
        Flux.cpu(l.μ_q), Flux.cpu(l.σ_q), Flux.cpu(l.μ_h), Flux.cpu(l.σ_h),
        Flux.cpu(l.μ_inwater), Flux.cpu(l.σ_inwater), Flux.cpu(l.postscale_inwater), l.dt,
        _to_cpu_sparse(l.A_routing),
        isnothing(l.A_routing_batched) ? nothing : _to_cpu_sparse(l.A_routing_batched),
        l.batch_size,
        l.θ,
        l.smooth_h_floor,
        l.h_floor_softness,
    )
end

"""
    build_massbalance(norm_stats, postscale, src_edges, tgt_edges, dt;
                      θ, smooth_h_floor, h_floor_softness,
                      unscale_inwater = true) -> ThetaKinematicWave

Factory for the river-domain [`ThetaKinematicWave`](@ref). Pulls the per-node
`river_q`/`river_h`/`river_inwater` z-score statistics out of `norm_stats`
(scalar stats are broadcast to every node, per-node vector stats are checked
against the node count), the inverse scalers out of `postscale`, and builds the
routing-only adjacency (no self-loops; `A_routing[i,j] = 1` means `j` flows
into `i`) from the graph edge lists. `dt` is the model timestep in seconds.

Preprocessing divides `river_inwater` by the upstream area before the z-score, so
the lateral inflow in m³/s is `(z·σ + μ) · postscale["river_inwater"]`. With
`unscale_inwater = true` (default) the layer applies that factor; `false` uses
ones, reproducing the pre-R5b behaviour (kept only to evaluate old checkpoints).

`n_nodes` is taken from `length(postscale["river_q"])`. The batched
block-diagonal routing matrix is left unset; call
[`precompute_batched`](@ref) afterwards.
"""
function build_massbalance(norm_stats, postscale,
                           src_edges::AbstractVector{<:Integer},
                           tgt_edges::AbstractVector{<:Integer},
                           dt;
                           θ::Real,
                           smooth_h_floor::Bool,
                           h_floor_softness::Real,
                           unscale_inwater::Bool = true)
    pq_vec  = postscale["river_q"]
    ph_vec  = postscale["river_h"]
    n_nodes = length(pq_vec)
    piw_vec = unscale_inwater ? Float32.(postscale["river_inwater"]) :
                                ones(Float32, n_nodes)

    stat_vec(vname::String, field::Symbol) = begin
        raw = getfield(norm_stats[vname], field)
        if raw isa AbstractVector
            length(raw) == n_nodes ||
                throw(ArgumentError("norm_stats[$vname].$field length $(length(raw)) != n_nodes $n_nodes"))
            Float32.(raw)
        else
            fill(Float32(raw), n_nodes)
        end
    end

    μ_q = stat_vec("river_q", :mean)
    σ_q = stat_vec("river_q", :std)
    μ_h = stat_vec("river_h", :mean)
    σ_h = stat_vec("river_h", :std)
    μ_inwater = stat_vec("river_inwater", :mean)
    σ_inwater = stat_vec("river_inwater", :std)
    # Routing-only adjacency (no self-loops): A_routing[i,j]=1 means j flows into i.
    A_routing = sparse(tgt_edges, src_edges,
                       ones(Float32, length(src_edges)), n_nodes, n_nodes)
    return ThetaKinematicWave(
        pq_vec,
        ph_vec,
        ph_vec ./ pq_vec,
        μ_q,
        σ_q,
        μ_h,
        σ_h,
        μ_inwater,
        σ_inwater,
        piw_vec,
        dt,
        A_routing,
        nothing,  # A_routing_batched — set via precompute_batched
        0,        # batch_size
        Float32(θ),
        smooth_h_floor,
        Float32(h_floor_softness),
    )
end

"""
    (l::ThetaKinematicWave)(g, state, forcing, forcing_next, q_norm_new) -> h_norm_new

    Compute normalised water depth at the next timestep by enforcing the
    θ-weighted (mixed implicit/explicit) kinematic-wave mass balance:

        h[t+1] = h[t] + dt/(w·l) · [ θ·(Σq[t+1] + iw[t+1] − q[t+1])
                                   + (1−θ)·(Σq[t]   + iw[t]   − q[t]) ]

    With `θ = 1` (default) this is the fully-implicit scheme and the explicit
    branch is skipped entirely (baseline behaviour, bit-identical).

- `state`         : normalised state matrix `(2, n_nodes)` — rows are river_q, river_h at t.
- `forcing`       : normalised forcing matrix at t; row 1 is river_inwater[t]
                    (used only for the explicit term when `θ < 1`).
- `forcing_next`  : normalised forcing matrix at t+1; row 1 is river_inwater[t+1].
- `q_norm_new`    : normalised predicted discharge `(1, n_nodes)` at t+1.

Returns normalised `h_norm_new` of shape `(1, n_nodes)`.

Handles batched `GNNGraph`s automatically: per-node constants are tiled to
match the total node count of the batch.
"""
function (l::ThetaKinematicWave)(g            ::GNNGraph,
                               state        ::AbstractMatrix,
                               forcing      ::AbstractMatrix,
                               forcing_next ::AbstractMatrix,
                               q_norm_new   ::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    # Tile per-node constants to cover the whole batch  (1 × n_nodes).
    # When n_rep == 1 (always true during rollout), reshape is a zero-copy view.
    if n_rep == 1
        pq  = reshape(l.postscale_q,  1, n)
        ph  = reshape(l.postscale_h,  1, n)
        phr = reshape(l.ph_over_pq,   1, n)
        μq  = reshape(l.μ_q,          1, n)
        σq  = reshape(l.σ_q,          1, n)
        μh  = reshape(l.μ_h,          1, n)
        σh  = reshape(l.σ_h,          1, n)
        μiw = reshape(l.μ_inwater,    1, n)
        σiw = reshape(l.σ_inwater,    1, n)
    else
        pq  = reshape(repeat(l.postscale_q,  n_rep), 1, n)
        ph  = reshape(repeat(l.postscale_h,  n_rep), 1, n)
        phr = reshape(repeat(l.ph_over_pq,   n_rep), 1, n)
        μq  = reshape(repeat(l.μ_q,          n_rep), 1, n)
        σq  = reshape(repeat(l.σ_q,          n_rep), 1, n)
        μh  = reshape(repeat(l.μ_h,          n_rep), 1, n)
        σh  = reshape(repeat(l.σ_h,          n_rep), 1, n)
        μiw = reshape(repeat(l.μ_inwater,    n_rep), 1, n)
        σiw = reshape(repeat(l.σ_inwater,    n_rep), 1, n)
    end

    # Physical discharge at current and predicted timesteps  [m³/s]
    # q_phys_new is floored at 0 in physical space (z-scored 0 ≠ physical 0).
    q_phys_new  = max.(0f0, pq .* (q_norm_new .* σq .+ μq))

    # Lateral inflow at t+1  [m³/s]  (row 1 = river_inwater)  — implicit term.
    # `river_inwater` was divided by the upstream area before the z-score, so the
    # denormalised value is scaled back by `postscale_inwater`.
    piw          = _tile_node_const(l.postscale_inwater, n_rep, n)
    inwater_phys = (forcing_next[1:1, :] .* σiw .+ μiw) .* piw

    # Sum upstream Q into each node via CuSPARSE SpMM.
    # Three dispatch paths mirror SparseConv:
    #  1. Single graph (n_rep==1): direct (N×N) SpMM.
    #  2. Batched with precomputed block-diagonal: single (B·N×B·N) SpMM.
    #  3. Batched fallback: reshape trick (no precomputation needed).
    N_per = length(l.postscale_q)
    route(x) =
        if n_rep == 1
            (_topology_mul(l.A_routing, x'))'
        elseif !isnothing(l.A_routing_batched) && n == l.batch_size * N_per
            (_topology_mul(l.A_routing_batched, x'))'
        else
            reshape(_topology_mul(l.A_routing, reshape(x, N_per, n_rep))', 1, n)
        end

    upstream_q = route(q_phys_new)

    # Physical h at current step  [m]
    # h_phys = postscale_h · (norm_h · σ_h + μ_h)
    h_phys_curr = ph .* (state[2:2, :] .* σh .+ μh)

    # θ-weighted net flux  [m³/s].  θ = 1 ⇒ pure implicit (baseline).
    net_flux_impl = upstream_q .+ inwater_phys .- q_phys_new
    if l.θ == 1f0
        net_flux = net_flux_impl
    else
        # Explicit side: all quantities are known at t (no dependence on the
        # current prediction), which is what damps the stiff q→h gradient.
        q_phys_curr     = max.(0f0, pq .* (state[1:1, :] .* σq .+ μq))
        inwater_curr    = (forcing[1:1, :] .* σiw .+ μiw) .* piw
        upstream_q_curr = route(q_phys_curr)
        net_flux_expl   = upstream_q_curr .+ inwater_curr .- q_phys_curr
        net_flux        = l.θ .* net_flux_impl .+ (1f0 - l.θ) .* net_flux_expl
    end

    # Mass balance  [m]:  Δh = dt · (1/(w·l)) · net_flux
    #   1/(w·l) = a/(w·l) / a = postscale_h / postscale_q  (precomputed as ph_over_pq)
    h_phys_new = h_phys_curr .+ l.dt .* phr .* net_flux

    # Water depth cannot be negative (dry-channel floor)
    h_phys_new = _nonnegative_floor(h_phys_new, l.smooth_h_floor, l.h_floor_softness)

    # Re-normalise:  scaled_h = h_phys / postscale_h  →  norm_h = (scaled_h - μ_h) / σ_h
    return (h_phys_new ./ ph .- μh) ./ σh
end

# Floor predicted discharge in physical space and map back to normalized q.
# This keeps river_q physically non-negative in the propagated model state.
function _floor_q_norm_nonnegative(l::ThetaKinematicWave,
                                   g::GNNGraph,
                                   q_norm::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per
    pq, μq, σq = if n_rep == 1
        reshape(l.postscale_q, 1, n), reshape(l.μ_q, 1, n), reshape(l.σ_q, 1, n)
    else
        reshape(repeat(l.postscale_q, n_rep), 1, n),
        reshape(repeat(l.μ_q, n_rep), 1, n),
        reshape(repeat(l.σ_q, n_rep), 1, n)
    end
    q_phys = pq .* (q_norm .* σq .+ μq)
    q_phys = max.(0f0, q_phys)
    return (q_phys ./ pq .- μq) ./ σq
end

# Number of physics-derived features appended to the decoder input when
# `mb_augment_decoder` is enabled (see `mb_decoder_features`).
const MB_DECODER_FEATURES = 2

"""
    mb_decoder_features(l, g, state, forcing) -> (upstream_q_norm, h_expl_norm)

Compute two physics-derived node features from quantities known at the current
timestep `t`, for use as **additional decoder inputs** when `mb_augment_decoder`
is enabled. Both are non-circular (they do not depend on the discharge the model
is about to predict) and differentiable w.r.t. `state`/`forcing`, so gradients
flow through them during BPTT rollout. The hard mass-balance output is unchanged.

- `upstream_q_norm` : sum of upstream discharge routed into each node at `t`,
                      re-expressed in the model's normalised discharge space.
- `h_expl_norm`     : the fully-explicit (forward-Euler, θ=0) estimate of the
                      next-step water depth, in normalised depth space — a cheap
                      physics warm-start the decoder can correct.

Both are `(1, n_nodes)`. Batched `GNNGraph`s are handled like the layer's
forward pass (per-node constants tiled, routing dispatched three ways).
"""
function mb_decoder_features(l           ::ThetaKinematicWave,
                             g           ::GNNGraph,
                             state       ::AbstractMatrix,
                             forcing     ::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    if n_rep == 1
        pq  = reshape(l.postscale_q,  1, n)
        ph  = reshape(l.postscale_h,  1, n)
        phr = reshape(l.ph_over_pq,   1, n)
        μq  = reshape(l.μ_q,          1, n)
        σq  = reshape(l.σ_q,          1, n)
        μh  = reshape(l.μ_h,          1, n)
        σh  = reshape(l.σ_h,          1, n)
        μiw = reshape(l.μ_inwater,    1, n)
        σiw = reshape(l.σ_inwater,    1, n)
    else
        pq  = reshape(repeat(l.postscale_q,  n_rep), 1, n)
        ph  = reshape(repeat(l.postscale_h,  n_rep), 1, n)
        phr = reshape(repeat(l.ph_over_pq,   n_rep), 1, n)
        μq  = reshape(repeat(l.μ_q,          n_rep), 1, n)
        σq  = reshape(repeat(l.σ_q,          n_rep), 1, n)
        μh  = reshape(repeat(l.μ_h,          n_rep), 1, n)
        σh  = reshape(repeat(l.σ_h,          n_rep), 1, n)
        μiw = reshape(repeat(l.μ_inwater,    n_rep), 1, n)
        σiw = reshape(repeat(l.σ_inwater,    n_rep), 1, n)
    end

    N_per = length(l.postscale_q)
    route(x) =
        if n_rep == 1
            (_topology_mul(l.A_routing, x'))'
        elseif !isnothing(l.A_routing_batched) && n == l.batch_size * N_per
            (_topology_mul(l.A_routing_batched, x'))'
        else
            reshape(_topology_mul(l.A_routing, reshape(x, N_per, n_rep))', 1, n)
        end

    # All quantities are known at t (explicit / forward-Euler) — no dependence
    # on the discharge about to be predicted, so they are safe to feed as inputs.
    q_phys_curr     = max.(0f0, pq .* (state[1:1, :] .* σq .+ μq))
    piw             = _tile_node_const(l.postscale_inwater, n_rep, n)
    inwater_curr    = (forcing[1:1, :] .* σiw .+ μiw) .* piw
    upstream_q_curr = route(q_phys_curr)
    net_flux_expl   = upstream_q_curr .+ inwater_curr .- q_phys_curr

    h_phys_curr = ph .* (state[2:2, :] .* σh .+ μh)
    h_phys_expl = _nonnegative_floor(h_phys_curr .+ l.dt .* phr .* net_flux_expl,
                                     l.smooth_h_floor, l.h_floor_softness)

    # Re-express both features in the model's normalised (z-scored) spaces so
    # they share the scale of the other inputs.
    upstream_q_norm = (upstream_q_curr ./ pq .- μq) ./ σq
    h_expl_norm     = (h_phys_expl ./ ph .- μh) ./ σh
    return upstream_q_norm, h_expl_norm
end

"""
    mb_diagnostics(l, g, state, forcing, q_norm_new) -> NamedTuple

Compute all intermediate physical quantities of the mass balance for one step.
Non-differentiable; intended for validation/debugging only.

Returns a `NamedTuple` with `Vector{Float32}` per node (physical units):
- `q_phys_curr`  [m³/s]: discharge from input state (denorm + postscale)
- `q_phys_new`   [m³/s]: predicted discharge (floored at 0)
- `upstream_q`   [m³/s]: sum of direct upstream neighbours' discharge
- `inwater_phys` [m³/s]: lateral inflow
- `net_flux`     [m³/s]: upstream_q + inwater - q_phys_new
- `h_phys_curr`  [m]:    current water depth
- `h_phys_raw`   [m]:    next water depth before the ≥0 floor
- `h_phys_new`   [m]:    next water depth after floor
"""
function mb_diagnostics(l            ::ThetaKinematicWave,
                        g            ::GNNGraph,
                        state        ::AbstractMatrix,
                        forcing      ::AbstractMatrix,
                        forcing_next ::AbstractMatrix,
                        q_norm_new   ::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    # Force everything to CPU plain arrays for the diagnostic
    pq  = reshape(repeat(Array(l.postscale_q), n_rep), 1, n)
    ph  = reshape(repeat(Array(l.postscale_h), n_rep), 1, n)
    μq  = reshape(repeat(Array(l.μ_q), n_rep), 1, n)
    σq  = reshape(repeat(Array(l.σ_q), n_rep), 1, n)
    μh  = reshape(repeat(Array(l.μ_h), n_rep), 1, n)
    σh  = reshape(repeat(Array(l.σ_h), n_rep), 1, n)
    μiw = reshape(repeat(Array(l.μ_inwater), n_rep), 1, n)
    σiw = reshape(repeat(Array(l.σ_inwater), n_rep), 1, n)
    piw = reshape(repeat(Array(l.postscale_inwater), n_rep), 1, n)
    st  = Array(state)
    fn  = Array(forcing_next)
    qn  = Array(q_norm_new)

    q_phys_curr  = pq .* (st[1:1, :] .* σq .+ μq)   # for diagnostics only
    q_phys_new   = max.(0f0, pq .* (qn .* σq .+ μq))
    inwater_phys = (fn[1:1, :] .* σiw .+ μiw) .* piw
    A_cpu        = _to_cpu_sparse(l.A_routing)
    upstream_q   = (A_cpu * q_phys_new')'
    h_phys_curr  = ph .* (st[2:2, :] .* σh .+ μh)
    net_flux_impl = upstream_q .+ inwater_phys .- q_phys_new
    if l.θ == 1f0
        net_flux = net_flux_impl
    else
        fc              = Array(forcing)
        q_phys_curr_f   = max.(0f0, pq .* (st[1:1, :] .* σq .+ μq))
        inwater_curr    = (fc[1:1, :] .* σiw .+ μiw) .* piw
        upstream_q_curr = (A_cpu * q_phys_curr_f')'
        net_flux_expl   = upstream_q_curr .+ inwater_curr .- q_phys_curr_f
        net_flux        = l.θ .* net_flux_impl .+ (1f0 - l.θ) .* net_flux_expl
    end
    h_phys_raw   = h_phys_curr .+ l.dt .* (ph ./ pq) .* net_flux
    h_phys_new   = _nonnegative_floor(h_phys_raw, l.smooth_h_floor, l.h_floor_softness)

    return (q_phys_curr  = vec(q_phys_curr),
            q_phys_new   = vec(q_phys_new),
            upstream_q   = vec(upstream_q),
            inwater_phys = vec(inwater_phys),
            net_flux     = vec(net_flux),
            h_phys_curr  = vec(h_phys_curr),
            h_phys_raw   = vec(h_phys_raw),
            h_phys_new   = vec(h_phys_new))
end

"""
    precompute_batched(layer::ThetaKinematicWave, batch_size::Int) -> ThetaKinematicWave

Return a new `ThetaKinematicWave` with a pre-computed block-diagonal routing matrix
`A_routing_batched = blockdiag(A_routing, ..., A_routing)` (`batch_size` copies).
Mirrors `precompute_batched(::SparseConv, B)`. Call **before** `Flux.gpu`.
"""
function precompute_batched(l::ThetaKinematicWave, B::Int)
    A_cpu = _to_cpu_sparse(l.A_routing)
    A_blk = SparseMatrixCSC{Float32, Int32}(blockdiag(fill(A_cpu, B)...))
    ThetaKinematicWave(
        l.postscale_q, l.postscale_h, l.ph_over_pq,
        l.μ_q, l.σ_q, l.μ_h, l.σ_h, l.μ_inwater, l.σ_inwater, l.postscale_inwater, l.dt,
        A_cpu, A_blk, B, l.θ,
        l.smooth_h_floor,
        l.h_floor_softness,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Formulation interface (see `MassBalanceFormulation`)
# ──────────────────────────────────────────────────────────────────────────────

# `NoMassBalance` carries no data: it has no children, no trainable parameters,
# and is its own CPU/GPU/batched version.
Flux.@layer NoMassBalance
Flux.gpu(f::NoMassBalance) = f
Flux.cpu(f::NoMassBalance) = f
precompute_batched(f::NoMassBalance, ::Int) = f

# Old checkpoints saved before the formulation hierarchy stored
# `mass_balance = nothing` for models without a mass balance.
Flux.loadmodel!(::NoMassBalance, ::Nothing; kw...) = nothing

# Checkpoints saved before R5b have no `postscale_inwater` in the layer state. Keep
# the destination's own value (all-ones for a legacy `mb_inwater_unscale = false`
# settings file), so such checkpoints reload with their original behaviour.
function Flux.loadmodel!(dst::ThetaKinematicWave, src::NamedTuple; kw...)
    if !haskey(src, :postscale_inwater)
        src = merge(src, (postscale_inwater = dst.postscale_inwater,))
    end
    invoke(Flux.loadmodel!, Tuple{Any, Any}, dst, src; kw...)
end

has_mass_balance(::NoMassBalance)       = false
has_mass_balance(::ThetaKinematicWave)  = true

# Number of state channels the decoder predicts (q only when `river_h` is derived
# from the mass balance; every state channel otherwise).
decoder_outputs(::NoMassBalance, n_state::Int)      = n_state
decoder_outputs(::ThetaKinematicWave, ::Int)        = 1

# Extra decoder inputs (a tuple of `(1, n_nodes)` matrices) and their row count.
extra_decoder_dim(::NoMassBalance)      = 0
extra_decoder_dim(::ThetaKinematicWave) = MB_DECODER_FEATURES
extra_decoder_features(::NoMassBalance, g, state, forcing) = ()
extra_decoder_features(f::ThetaKinematicWave, g, state, forcing) =
    mb_decoder_features(f, g, state, forcing)

# Physical-space floor of the raw predicted discharge (normalised in/out).
floor_state(::NoMassBalance, g, raw)          = raw
floor_state(f::ThetaKinematicWave, g, raw)    = _floor_q_norm_nonnegative(f, g, raw)

"""
    apply_formulation(f, g, state, forcing, forcing_next, Δ) -> next_state

Map the decoder output `Δ` to the next normalised state `(n_state, n_nodes)`.
`NoMassBalance` is the plain residual update `state .+ Δ`. `ThetaKinematicWave`
treats `Δ` as the discharge increment, floors the propagated q, and derives
`river_h` deterministically from the mass balance.
"""
apply_formulation(::NoMassBalance, g, state, forcing, forcing_next, Δ) = state .+ Δ

function apply_formulation(f::ThetaKinematicWave, g, state, forcing, forcing_next, Δ)
    q_new = floor_state(f, g, state[1:1, :] .+ Δ)
    h_new = f(g, state, forcing, forcing_next, q_new)
    return vcat(q_new, h_new)
end

# Weight of the water-depth loss term implied by the formulation's stiffness
# (`nothing` = leave the strategy's weight untouched).
mb_h_loss_weight(::NoMassBalance, ::Symbol) = nothing

function mb_h_loss_weight(mb::ThetaKinematicWave, h_loss_scale::Symbol)
    finite_median(v::AbstractVector{<:Real}) = begin
        fv = Float32[x for x in v if isfinite(x)]
        isempty(fv) ? NaN32 : Float32(median(fv))
    end
    finite_max(v::AbstractVector{<:Real}) = begin
        fv = Float32[x for x in v if isfinite(x)]
        isempty(fv) ? NaN32 : Float32(maximum(fv))
    end
    # Base weight balances the q- and h-loss magnitudes; ∂h_norm/∂q_norm of
    # the hard mass-balance decoder equals `dt·σ_q/σ_h = 1/base`, so the
    # h-branch injects a q-gradient amplified by `1/base` at :absolute.
    # :increment squares the weight (measures h on the `dt·σ_q` increment
    # scale), cancelling that amplification so ∂h_norm/∂q_norm ≈ O(1).
    base_weight_vec = mb.σ_h ./ (mb.dt .* mb.σ_q)
    base_weight = finite_median(base_weight_vec)
    h_weight    = h_loss_scale === :increment ? base_weight^2 : base_weight
    @info "Mass balance h_loss_weight = $(round(h_weight; sigdigits=3)) " *
          "[scale=$(h_loss_scale)]  " *
          "(median σ_h=$(round(finite_median(mb.σ_h); sigdigits=3)), " *
          "median σ_q=$(round(finite_median(mb.σ_q); sigdigits=3)), dt=$(mb.dt) s)"
    gain_vec = mb.θ .* mb.dt .* mb.σ_q ./ mb.σ_h
    @info "Mass balance per-node gain |∂h_norm/∂q_norm| summary: " *
          "median=$(round(finite_median(gain_vec); sigdigits=3)) " *
          "max=$(round(finite_max(gain_vec); sigdigits=3))"
    if mb.θ != 1f0
        @info "Mass balance θ = $(mb.θ) (mixed implicit/explicit; " *
              "θ=1 fully implicit, θ=0 fully explicit). Effective stiff " *
              "gain ∂h_norm/∂q_norm scaled by θ."
    end
    if mb.smooth_h_floor
        @info "Mass balance smooth h-floor enabled (softness=$(mb.h_floor_softness) m)"
    end
    return h_weight
end

# ──────────────────────────────────────────────────────────────────────────────
# Reservoir backbone + continuity readout hybrid
# ──────────────────────────────────────────────────────────────────────────────

"""
    MuskingumContinuityHybrid

Mass-balance formulation in which the **storage** (water depth `h`) is the only
carried state and discharge `q` is a *readout*, never fed back.

Per step, with `S = h·w·l` the channel storage [m³] and `I` the lateral inflow [m³/s]:

1. **Backbone** — an implicit-Euler *linear-reservoir cascade* (`S_i = K_i·Q_i`):

       S_bb[t+1] = P · ( S[t] + dt·I[t+1] ),    P = (I + dt·D − dt·A·D)⁻¹,   D = diag(1/K)

   `A[i,j] = 1` means `j` flows into `i`. `P` is the exact solution of
   `S[t+1] − S[t] = dt·(I + (A − I)·S[t+1]/K)`; it is **non-negative** (`I + dt·D − dt·A·D`
   is an M-matrix) and, because `A` is nilpotent, lower-triangular in topological
   order, so its spectrum is its diagonal `1/(1 + dt/K_i) < 1` for *any* `K_i > 0` —
   unconditional contraction, no Courant condition. (The Muskingum–Cunge three-
   coefficient form needs `K(1−x) ≥ dt/2`; with daily steps and Courant numbers ≫ 1 that
   fails for physical `K`, which is why the implicit reservoir — Muskingum with `x = 0` —
   is used.) `K_i = μ_h/μ_q` is the per-node mean storage/discharge ratio from the
   training statistics, clamped to `[0.05, 100]·dt`.

2. **Bounded learned residual** — the decoder output `Δ` enters as
   `h_norm = h_bb_norm + ρ_max·tanh(Δ)` (`ρ_max = residual_bound`). The residual is
   bounded in value, so the state stays bounded for any weights (bounded input ⇒ bounded
   state of a contracting linear system). This is *not* a claim that the learned map
   cannot change the local Jacobian; it can, but it cannot make the state diverge.
   A non-negative floor (hard or softplus) is then applied to the physical depth.

3. **Continuity readout** — discharge is solved from the *predicted* storage change:

       Q[t+1] = (I − A)⁻¹ · ( I[t+1] − (S[t+1] − S[t])/dt ),

   which satisfies `dS/dt = A·Q + I − Q` **exactly** at every node for whatever `S[t+1]`
   the network produced. The only conservation break is the output floor `Q ≥ 0`, which is
   outside the dynamics (`q` is not fed back: the encoder sees a zeroed `q` row, see
   [`encoder_state`](@ref)).

Both `P` and `(I − A)⁻¹` are precomputed sparse matrices (no dense inverse); a batched
block-diagonal copy is built by [`precompute_batched`](@ref).

Fields: per-node constants as in [`ThetaKinematicWave`](@ref) plus `K` [s],
`contraction = 1/(1 + dt/K)` (diagonal of `P`), `P`, `R = (I−A)⁻¹`, their batched
counterparts, `residual_bound`, and the smooth-floor settings.
"""
struct MuskingumContinuityHybrid{V <: AbstractVector{Float32}} <: MassBalanceFormulation
    postscale_q       :: V
    postscale_h       :: V
    ph_over_pq        :: V
    μ_q               :: V
    σ_q               :: V
    μ_h               :: V
    σ_h               :: V
    μ_inwater         :: V
    σ_inwater         :: V
    postscale_inwater :: V
    K                 :: V   # reservoir time constant per node [s]
    contraction       :: V   # 1/(1 + dt/K): diagonal (= spectrum) of P
    dt                :: Float32
    P                 :: AbstractMatrix{Float32}   # (N×N) backbone operator
    R                 :: AbstractMatrix{Float32}   # (N×N) readout operator (I − A)⁻¹
    P_batched         :: Union{Nothing, AbstractMatrix{Float32}}
    R_batched         :: Union{Nothing, AbstractMatrix{Float32}}
    batch_size        :: Int
    residual_bound    :: Float32
    smooth_h_floor    :: Bool
    h_floor_softness  :: Float32
end

Flux.@layer MuskingumContinuityHybrid
Flux.trainable(::MuskingumContinuityHybrid) = (;)  # physics constants, not optimised

Functors.@functor MuskingumContinuityHybrid (
    postscale_q, postscale_h, ph_over_pq,
    μ_q, σ_q, μ_h, σ_h, μ_inwater, σ_inwater, postscale_inwater,
    K, contraction,
)

# Re-create `l` with every per-node vector mapped by `fv` and every sparse operator by `fm`.
function _mch_move(l::MuskingumContinuityHybrid, fv, fm)
    MuskingumContinuityHybrid(
        fv(l.postscale_q), fv(l.postscale_h), fv(l.ph_over_pq),
        fv(l.μ_q), fv(l.σ_q), fv(l.μ_h), fv(l.σ_h),
        fv(l.μ_inwater), fv(l.σ_inwater), fv(l.postscale_inwater),
        fv(l.K), fv(l.contraction), l.dt,
        fm(l.P), fm(l.R),
        isnothing(l.P_batched) ? nothing : fm(l.P_batched),
        isnothing(l.R_batched) ? nothing : fm(l.R_batched),
        l.batch_size, l.residual_bound, l.smooth_h_floor, l.h_floor_softness,
    )
end
Flux.gpu(l::MuskingumContinuityHybrid) = _mch_move(l, Flux.gpu, _to_cusparse)
Flux.cpu(l::MuskingumContinuityHybrid) = _mch_move(l, Flux.cpu, _to_cpu_sparse)

# Bounds of the reservoir constant, as multiples of the timestep.
const MCH_K_MIN_DT = 0.05f0
const MCH_K_MAX_DT = 100f0

"""
    _tree_inverse_operator(src, tgt, n, c, d) -> SparseMatrixCSC{Float32}

Sparse inverse of the lower-triangular operator `L = diag(d) − A·diag(c)` on the
routing DAG (`A[i,j] = 1` for each edge `src[k]=j → tgt[k]=i`), computed row by row in
topological order from `row_i = (e_i + Σ_{j→i} c_j·row_j) / d_i`. Each node's row is
merged into exactly one downstream row, so the cost is `O(nnz(L⁻¹))` — no dense inverse.
`c = d = 1` gives the accumulation operator `(I − A)⁻¹` (`1` where `j` is `i` or upstream
of it). Throws `ArgumentError` when the graph has a cycle.
"""
function _tree_inverse_operator(src::AbstractVector{<:Integer}, tgt::AbstractVector{<:Integer},
                                n::Int, c::AbstractVector, d::AbstractVector)
    ups  = [Int[] for _ in 1:n]
    down = [Int[] for _ in 1:n]
    remaining = zeros(Int, n)
    for (s, t) in zip(src, tgt)
        push!(ups[t], s); push!(down[s], t); remaining[t] += 1
    end
    queue = findall(==(0), remaining)
    rows_idx = Vector{Vector{Int32}}(undef, n)
    rows_val = Vector{Vector{Float32}}(undef, n)
    acc     = zeros(Float64, n)
    mark    = falses(n)
    touched = Int[]
    head = 1
    while head <= length(queue)
        i = queue[head]; head += 1
        empty!(touched)
        acc[i] = 1.0; mark[i] = true; push!(touched, i)
        for j in ups[i]
            cj = Float64(c[j])
            for (k, v) in zip(rows_idx[j], rows_val[j])
                if !mark[k]
                    mark[k] = true; push!(touched, k)
                end
                acc[k] += cj * v
            end
        end
        di = Float64(d[i])
        rows_idx[i] = Int32.(touched)
        rows_val[i] = Float32[acc[k] / di for k in touched]
        for k in touched
            acc[k] = 0.0; mark[k] = false
        end
        for t in down[i]
            remaining[t] -= 1
            remaining[t] == 0 && push!(queue, t)
        end
    end
    length(queue) == n ||
        throw(ArgumentError("routing graph contains a cycle ($(n - length(queue)) nodes unreachable in topological order)"))
    nnz_total = sum(length, rows_idx)
    I = Vector{Int}(undef, nnz_total); J = Vector{Int}(undef, nnz_total)
    V = Vector{Float32}(undef, nnz_total)
    p = 0
    for i in 1:n
        for (k, v) in zip(rows_idx[i], rows_val[i])
            p += 1; I[p] = i; J[p] = k; V[p] = v
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    build_muskingum_continuity(norm_stats, postscale, src_edges, tgt_edges, dt;
                               residual_bound, smooth_h_floor, h_floor_softness,
                               unscale_inwater = true) -> MuskingumContinuityHybrid

Factory for [`MuskingumContinuityHybrid`](@ref); statistics, inverse scalers and `dt`
as in [`build_massbalance`](@ref). The reservoir constant is `K_i = μ_h/μ_q` (mean
scaled storage over mean scaled discharge, in seconds), clamped to
`[$(MCH_K_MIN_DT), $(MCH_K_MAX_DT)]·dt`; nodes without a positive finite ratio fall back
to `K = dt`. Aborts (`ArgumentError`) on a non-finite or negative `residual_bound` or a
cyclic routing graph. The batched operators are left unset; call
[`precompute_batched`](@ref).
"""
function build_muskingum_continuity(norm_stats, postscale,
                                    src_edges::AbstractVector{<:Integer},
                                    tgt_edges::AbstractVector{<:Integer},
                                    dt;
                                    residual_bound::Real,
                                    smooth_h_floor::Bool,
                                    h_floor_softness::Real,
                                    unscale_inwater::Bool = true)
    isfinite(residual_bound) && residual_bound >= 0 ||
        throw(ArgumentError("mb_residual_bound must be finite and ≥ 0, got $residual_bound"))
    pq_vec  = postscale["river_q"]
    ph_vec  = postscale["river_h"]
    n_nodes = length(pq_vec)
    piw_vec = unscale_inwater ? Float32.(postscale["river_inwater"]) :
                                ones(Float32, n_nodes)
    stat_vec(vname::String, field::Symbol) = begin
        raw = getfield(norm_stats[vname], field)
        if raw isa AbstractVector
            length(raw) == n_nodes ||
                throw(ArgumentError("norm_stats[$vname].$field length $(length(raw)) != n_nodes $n_nodes"))
            Float32.(raw)
        else
            fill(Float32(raw), n_nodes)
        end
    end
    μ_q = stat_vec("river_q", :mean);  σ_q = stat_vec("river_q", :std)
    μ_h = stat_vec("river_h", :mean);  σ_h = stat_vec("river_h", :std)
    μ_iw = stat_vec("river_inwater", :mean); σ_iw = stat_vec("river_inwater", :std)

    dtf = Float32(dt)
    K = μ_h ./ μ_q
    n_fallback = count(k -> !(isfinite(k) && k > 0f0), K)
    K[.!(isfinite.(K) .& (K .> 0f0))] .= dtf
    n_clipped = count(k -> k < MCH_K_MIN_DT * dtf || k > MCH_K_MAX_DT * dtf, K)
    clamp!(K, MCH_K_MIN_DT * dtf, MCH_K_MAX_DT * dtf)
    λ = dtf ./ K
    contraction = 1f0 ./ (1f0 .+ λ)
    all(c -> 0f0 < c < 1f0, contraction) ||
        throw(ArgumentError("backbone contraction left (0, 1): K must be positive and finite"))

    P = _tree_inverse_operator(src_edges, tgt_edges, n_nodes, λ, 1f0 .+ λ)
    R = _tree_inverse_operator(src_edges, tgt_edges, n_nodes,
                               ones(Float32, n_nodes), ones(Float32, n_nodes))
    @info "Muskingum-continuity hybrid: median K/dt = $(round(median(K) / dtf; sigdigits = 3)), " *
          "backbone contraction (spectral radius) = $(round(maximum(contraction); sigdigits = 4)); " *
          "$n_fallback node(s) K→dt fallback, $n_clipped clamped; nnz(P) = $(nnz(P))"
    return MuskingumContinuityHybrid(
        pq_vec, ph_vec, ph_vec ./ pq_vec, μ_q, σ_q, μ_h, σ_h, μ_iw, σ_iw, piw_vec,
        K, contraction, dtf, P, R, nothing, nothing, 0,
        Float32(residual_bound), smooth_h_floor, Float32(h_floor_softness),
    )
end

function precompute_batched(l::MuskingumContinuityHybrid, B::Int)
    blk(M) = SparseMatrixCSC{Float32, Int32}(blockdiag(fill(_to_cpu_sparse(M), B)...))
    MuskingumContinuityHybrid(
        l.postscale_q, l.postscale_h, l.ph_over_pq,
        l.μ_q, l.σ_q, l.μ_h, l.σ_h, l.μ_inwater, l.σ_inwater, l.postscale_inwater,
        l.K, l.contraction, l.dt,
        _to_cpu_sparse(l.P), _to_cpu_sparse(l.R), blk(l.P), blk(l.R), B,
        l.residual_bound, l.smooth_h_floor, l.h_floor_softness,
    )
end

# `M · x` for a `(1, n)` row `x`, with the same three dispatch paths as the θ layer
# (single graph / precomputed block-diagonal / reshape fallback).
function _mch_apply(l::MuskingumContinuityHybrid, M, Mb, x::AbstractMatrix, n::Int, n_rep::Int)
    N_per = length(l.postscale_q)
    if n_rep == 1
        return (_topology_mul(M, x'))'
    elseif !isnothing(Mb) && n == l.batch_size * N_per
        return (_topology_mul(Mb, x'))'
    else
        return reshape(_topology_mul(M, reshape(x, N_per, n_rep)), 1, n)
    end
end

"""
    (l::MuskingumContinuityHybrid)(g, state, forcing_next, Δ) -> next_state

One step of the hybrid (see [`MuskingumContinuityHybrid`](@ref)): backbone, bounded
residual `Δ`, depth floor, continuity readout. `state` is the normalised `(2, n)` state
at `t` (only its `river_h` row is used); returns the normalised `(2, n)` state at `t+1`
(`river_q` from the readout, `river_h` the carried storage).
"""
function (l::MuskingumContinuityHybrid)(g::GNNGraph, state::AbstractMatrix,
                                        forcing_next::AbstractMatrix, Δ::AbstractMatrix)
    n     = g.num_nodes
    n_rep = n ÷ length(l.postscale_q)
    tile(v) = _tile_node_const(v, n_rep, n)
    pq, ph = tile(l.postscale_q), tile(l.postscale_h)
    μq, σq = tile(l.μ_q), tile(l.σ_q)
    μh, σh = tile(l.μ_h), tile(l.σ_h)
    μiw, σiw, piw = tile(l.μ_inwater), tile(l.σ_inwater), tile(l.postscale_inwater)

    # Storage S = pq · scaled_h [m³]; lateral inflow at t+1 [m³/s].
    S_curr = pq .* (state[2:2, :] .* σh .+ μh)
    I_next = (forcing_next[1:1, :] .* σiw .+ μiw) .* piw

    # 1. backbone (implicit-Euler reservoir cascade), in normalised depth space
    S_bb  = _mch_apply(l, l.P, l.P_batched, S_curr .+ l.dt .* I_next, n, n_rep)
    h_bb  = (S_bb ./ pq .- μh) ./ σh
    # 2. bounded residual + ≥ 0 depth floor
    h_pre = h_bb .+ l.residual_bound .* tanh.(Δ)
    h_phys = _nonnegative_floor(ph .* (h_pre .* σh .+ μh), l.smooth_h_floor, l.h_floor_softness)
    scaled_h_new = h_phys ./ ph
    S_new = pq .* scaled_h_new
    # 3. continuity readout of discharge (exact for the storage change above)
    q_phys = _mch_apply(l, l.R, l.R_batched, I_next .- (S_new .- S_curr) ./ l.dt, n, n_rep)
    q_phys = max.(0f0, q_phys)
    return vcat((q_phys ./ pq .- μq) ./ σq, (scaled_h_new .- μh) ./ σh)
end

has_mass_balance(::MuskingumContinuityHybrid)   = true
decoder_outputs(::MuskingumContinuityHybrid, ::Int) = 1
extra_decoder_dim(::MuskingumContinuityHybrid)  = 0
extra_decoder_features(::MuskingumContinuityHybrid, g, state, forcing) = ()
floor_state(::MuskingumContinuityHybrid, g, raw) = raw
apply_formulation(f::MuskingumContinuityHybrid, g, state, forcing, forcing_next, Δ) =
    f(g, state, forcing_next, Δ)

function mb_h_loss_weight(mb::MuskingumContinuityHybrid, ::Symbol)
    @info "Muskingum-continuity hybrid: h_loss_weight left at the strategy value " *
          "(h is a carried state, not a stiff q→h readout)"
    return nothing
end

"""
    encoder_state(f, state) -> state

State rows the encoder may see. Identity for every formulation except
[`MuskingumContinuityHybrid`](@ref), whose `river_q` row (a readout, not a state) is
zeroed so the dynamics depend only on the carried storage `h` and the forcing. The row
count is unchanged, so the encoder input width and checkpoints keep their layout.
"""
encoder_state(::MassBalanceFormulation, state) = state
function encoder_state(::MuskingumContinuityHybrid, state)
    q0 = Flux.ignore_derivatives(() -> zero(state[1:1, :]))
    return vcat(q0, state[2:2, :])
end

# Diagonal (= spectrum) of the backbone operator, per node: `1/(1 + dt/K_i)`.
backbone_contraction(f::MuskingumContinuityHybrid) = Array(f.contraction)
