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
# Restrict Functors traversal to the per-node vectors; exclude A_routing
# (not writable in-place on GPU). Device transfer is handled by the explicit
# Flux.gpu / Flux.cpu overloads below.
Functors.@functor ThetaKinematicWave (
    postscale_q, postscale_h, ph_over_pq,
    μ_q, σ_q, μ_h, σ_h, μ_inwater, σ_inwater,
)
function Flux.gpu(l::ThetaKinematicWave)
    ThetaKinematicWave(
        Flux.gpu(l.postscale_q),
        Flux.gpu(l.postscale_h),
        Flux.gpu(l.ph_over_pq),
        Flux.gpu(l.μ_q), Flux.gpu(l.σ_q), Flux.gpu(l.μ_h), Flux.gpu(l.σ_h),
        Flux.gpu(l.μ_inwater), Flux.gpu(l.σ_inwater), l.dt,
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
        Flux.cpu(l.μ_inwater), Flux.cpu(l.σ_inwater), l.dt,
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
                      θ, smooth_h_floor, h_floor_softness) -> ThetaKinematicWave

Factory for the river-domain [`ThetaKinematicWave`](@ref). Pulls the per-node
`river_q`/`river_h`/`river_inwater` z-score statistics out of `norm_stats`
(scalar stats are broadcast to every node, per-node vector stats are checked
against the node count), the inverse scalers out of `postscale`, and builds the
routing-only adjacency (no self-loops; `A_routing[i,j] = 1` means `j` flows
into `i`) from the graph edge lists. `dt` is the model timestep in seconds.

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
                           h_floor_softness::Real)
    pq_vec  = postscale["river_q"]
    ph_vec  = postscale["river_h"]
    n_nodes = length(pq_vec)

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
        dt,
        A_routing,
        nothing,  # A_routing_batched — set via precompute_batched
        0,        # batch_size
        θ,
        smooth_h_floor,
        h_floor_softness,
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

    # Lateral inflow at t+1  [m³/s]  (row 1 = river_inwater)  — implicit term
    inwater_phys = forcing_next[1:1, :] .* σiw .+ μiw

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
        inwater_curr    = forcing[1:1, :] .* σiw .+ μiw
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
    inwater_curr    = forcing[1:1, :] .* σiw .+ μiw
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
    st  = Array(state)
    fn  = Array(forcing_next)
    qn  = Array(q_norm_new)

    q_phys_curr  = pq .* (st[1:1, :] .* σq .+ μq)   # for diagnostics only
    q_phys_new   = max.(0f0, pq .* (qn .* σq .+ μq))
    inwater_phys = fn[1:1, :] .* σiw .+ μiw
    A_cpu        = _to_cpu_sparse(l.A_routing)
    upstream_q   = (A_cpu * q_phys_new')'
    h_phys_curr  = ph .* (st[2:2, :] .* σh .+ μh)
    net_flux_impl = upstream_q .+ inwater_phys .- q_phys_new
    if l.θ == 1f0
        net_flux = net_flux_impl
    else
        fc              = Array(forcing)
        q_phys_curr_f   = max.(0f0, pq .* (st[1:1, :] .* σq .+ μq))
        inwater_curr    = fc[1:1, :] .* σiw .+ μiw
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
        l.μ_q, l.σ_q, l.μ_h, l.σ_h, l.μ_inwater, l.σ_inwater, l.dt,
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
