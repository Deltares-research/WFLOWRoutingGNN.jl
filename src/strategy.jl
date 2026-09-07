import TOML
using Flux
using GraphNeuralNetworks

"""
    TrainingStrategy

Defines the multi-step rollout curriculum and input noise regularization used
during training.

Fields:
- `steps`         : prediction horizons (steps ahead) for each training phase.
- `durations`     : number of epochs for each phase; must match `length(steps)`.
- `noise_scale`   : std dev of Gaussian noise added to `state` and `forcing` inputs
                    at each rollout step (default `0f0`, i.e. disabled).
- `current_steps` : active prediction horizon; updated by the training loop to
                    reflect the current position in the schedule.

The phases are executed in order: `steps[1]` steps ahead for `durations[1]`
epochs, then `steps[2]` steps ahead for `durations[2]` epochs, and so on.
Once all phases are exhausted the last phase is repeated indefinitely.
"""
mutable struct TrainingStrategy
    steps          :: Vector{Int}
    durations      :: Vector{Int}
    noise_scale    :: Float32
    h_loss_weight  :: Float32
    loss_type      :: Symbol
    peak_delta     :: Float32
    peak_lambda    :: Float32
    peak_gamma     :: Float32
    peak_w_max     :: Float32
    current_steps  :: Int
end

"""
    TrainingStrategy(steps, durations, noise_scale = 0) -> TrainingStrategy

Construct a `TrainingStrategy`. `current_steps` is initialised to `steps[1]`.
"""
function TrainingStrategy(steps, durations, noise_scale = 0;
                          h_loss_weight = 1f0,
                          loss_type = :mse,
                          peak_delta = 1f0,
                          peak_lambda = 0f0,
                          peak_gamma = 1f0,
                          peak_w_max = 4f0)
    length(steps) == length(durations) ||
        throw(ArgumentError("steps and durations must have the same length"))
    isempty(steps) &&
        throw(ArgumentError("steps must not be empty"))
    all(>(0), steps) ||
        throw(ArgumentError("all steps must be positive"))
    all(>(0), durations) ||
        throw(ArgumentError("all durations must be positive"))
    noise_scale >= 0 ||
        throw(ArgumentError("noise_scale must be non-negative"))
    loss_type in (:mse, :huber) ||
        throw(ArgumentError("loss_type must be :mse or :huber"))
    peak_delta > 0 ||
        throw(ArgumentError("peak_delta must be positive"))
    peak_lambda >= 0 ||
        throw(ArgumentError("peak_lambda must be non-negative"))
    peak_gamma > 0 ||
        throw(ArgumentError("peak_gamma must be positive"))
    peak_w_max >= 1 ||
        throw(ArgumentError("peak_w_max must be >= 1"))
    steps_v = convert(Vector{Int}, steps)
    TrainingStrategy(steps_v,
                     convert(Vector{Int}, durations),
                     Float32(noise_scale),
                     Float32(h_loss_weight),
                     loss_type,
                     Float32(peak_delta),
                     Float32(peak_lambda),
                     Float32(peak_gamma),
                     Float32(peak_w_max),
                     steps_v[1])
end

function Base.show(io::IO, s::TrainingStrategy)
    println(io, "TrainingStrategy:")
    println(io, "  steps          : ", s.steps)
    println(io, "  durations      : ", s.durations)
    println(io, "  noise_scale    : ", s.noise_scale)
    println(io, "  h_loss_weight  : ", s.h_loss_weight)
    println(io, "  loss_type      : ", s.loss_type)
    println(io, "  peak_delta     : ", s.peak_delta)
    println(io, "  peak_lambda    : ", s.peak_lambda)
    println(io, "  peak_gamma     : ", s.peak_gamma)
    println(io, "  peak_w_max     : ", s.peak_w_max)
    print(  io, "  current_steps  : ", s.current_steps)
end

"""
    save_training_strategy(path, strategy)

Write `strategy` to a TOML file at `path`. `current_steps` is not persisted
(it is always re-initialised from `steps[1]` on load).
"""
function save_training_strategy(path::String, s::TrainingStrategy)
    dict = Dict(
        "steps"       => s.steps,
        "durations"   => s.durations,
        "noise_scale" => Float64(s.noise_scale),
        "loss_type"   => String(s.loss_type),
        "peak_delta"  => Float64(s.peak_delta),
        "peak_lambda" => Float64(s.peak_lambda),
        "peak_gamma"  => Float64(s.peak_gamma),
        "peak_w_max"  => Float64(s.peak_w_max),
    )
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_training_strategy(path) -> TrainingStrategy

Read a `TrainingStrategy` from the TOML file at `path`.
"""
function load_training_strategy(path::String)
    d = TOML.parsefile(path)
    return TrainingStrategy(
        convert(Vector{Int}, d["steps"]),
        convert(Vector{Int}, d["durations"]),
        Float32(get(d, "noise_scale", 0.0));
        loss_type = Symbol(get(d, "loss_type", "mse")),
        peak_delta = Float32(get(d, "peak_delta", 1.0)),
        peak_lambda = Float32(get(d, "peak_lambda", 0.0)),
        peak_gamma = Float32(get(d, "peak_gamma", 1.0)),
        peak_w_max = Float32(get(d, "peak_w_max", 4.0)),
    )
end

# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

"""
    update_steps!(strategy, epoch)

Update `strategy.current_steps` to the horizon for `epoch` (1-based).
After all phases are exhausted the last phase is repeated indefinitely.
"""
function update_steps!(strategy::TrainingStrategy, epoch::Int)
    cumulative = 0
    for (nsteps, dur) in zip(strategy.steps, strategy.durations)
        cumulative += dur
        if epoch <= cumulative
            strategy.current_steps = nsteps
            return
        end
    end
    strategy.current_steps = strategy.steps[end]
end

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

function _huber_element(residual::Real, delta::Real)
    abs_r = abs(residual)
    return abs_r <= delta ? 0.5f0 * abs_r * abs_r : delta * (abs_r - 0.5f0 * delta)
end

# Broadcast helper for `peak_weighted_huber_loss`'s node-varying threshold/scale.
# `u`/`s` given as a plain `AbstractVector` (length `nvar`, one scalar per
# channel) is reshaped to a `(nvar, 1)` column so it broadcasts across all
# `nnode` nodes; given as an `AbstractMatrix` of shape `(nvar, nnode)`, it
# supplies a distinct threshold/scale **per node** and is used as-is. Purely
# non-mutating (no in-place `.=`), so it stays differentiable under Zygote.
# Keep all intermediate arrays on the same device as the inputs so the Huber
# loss remains valid under CUDA/Zygote, where host-side `Matrix` arguments are
# rejected by GPU kernels.
_peak_threshold_broadcast(u::AbstractVector) = reshape(Float32.(u), :, 1)
_peak_threshold_broadcast(u::AbstractMatrix) = Float32.(u)

_same_device_fill(x::AbstractArray, value::Real; T::Type = eltype(x)) = Flux.ignore_derivatives() do
    out = similar(x, T)
    fill!(out, T(value))
    out
end

_same_device_fill(x::AbstractArray, value::Real, dims::Tuple{Vararg{Int}}; T::Type = eltype(x)) = Flux.ignore_derivatives() do
    out = similar(x, T, dims...)
    fill!(out, T(value))
    out
end

_same_device_falses(x::AbstractArray) = Flux.ignore_derivatives() do
    out = similar(x, Bool)
    fill!(out, false)
    out
end

_peak_weight_and_score(target::AbstractMatrix,
                      u::Union{AbstractVector,AbstractMatrix},
                      s::Union{AbstractVector,AbstractMatrix},
                      lambda::Real, gamma::Real, w_max::Real) = begin
    u_b = _peak_threshold_broadcast(u)
    s_b = max.(_peak_threshold_broadcast(s), eps(Float32))
    peak_scores = max.(0f0, (target .- u_b) ./ s_b)
    weights = min.(1f0 .+ Float32(lambda) .* (peak_scores .^ Float32(gamma)), Float32(w_max))
    (weights, peak_scores)
end

# Shared by `loss_function` and `peak_epoch_diagnostics`: resolves the
# q/h peak threshold (`u`) and scale (`s`) for a batch given `peak_stats`
# (per-node, from `peak_node_stats`) or `nothing` (coarse per-batch fallback).
# When `peak_stats` is supplied, its single-graph per-node vectors are tiled
# across the batch dimension to match the block-diagonal node concatenation
# order produced by `GNNGraphs.batch` (batchsize may be > 1, and the last
# batch of an epoch may be a short remainder).
function _resolve_peak_thresholds(peak_stats, q_target::AbstractMatrix, h_target::AbstractMatrix)
    if peak_stats === nothing
        q_u = _same_device_fill(q_target, maximum(abs, q_target), (1,); T = Float32)
        q_s = _same_device_fill(q_target, maximum(abs, q_target) + eps(Float32), (1,); T = Float32)
        h_u = _same_device_fill(h_target, maximum(abs, h_target), (1,); T = Float32)
        h_s = _same_device_fill(h_target, maximum(abs, h_target) + eps(Float32), (1,); T = Float32)
        return q_u, q_s, h_u, h_s
    end
    n_graph_nodes = length(peak_stats.q.u)
    n_cols        = size(q_target, 2)
    n_cols % n_graph_nodes == 0 ||
        throw(ArgumentError("batch node count ($n_cols) is not a multiple of " *
                            "peak_stats node count ($n_graph_nodes)"))
    batch_reps = n_cols ÷ n_graph_nodes
    q_u = reshape(repeat(Float32.(peak_stats.q.u), batch_reps), 1, :)
    q_s = reshape(repeat(Float32.(peak_stats.q.s), batch_reps), 1, :)
    h_u = reshape(repeat(Float32.(peak_stats.h.u), batch_reps), 1, :)
    h_s = reshape(repeat(Float32.(peak_stats.h.s), batch_reps), 1, :)
    return q_u, q_s, h_u, h_s
end

"""
    peak_weighted_huber_loss(pred, target, u, s; delta = 1.0, lambda = 0.0,
                             gamma = 1.0, w_max = 4.0) -> Float32

Peak-weighted Huber loss. `u`/`s` may be given either as:
- a plain `AbstractVector` of length `nvar` — one scalar peak threshold /
  scale per channel, applied uniformly across all nodes; or
- an `AbstractMatrix` of shape `(nvar, nnode)` — a distinct per-**node**
  threshold / scale for each channel (the `u_i`/`s_i` of the design notes),
  e.g. precomputed by [`peak_node_stats`](@ref) on the training split.
"""
function peak_weighted_huber_loss(pred::AbstractMatrix, target::AbstractMatrix,
                                 u::Union{AbstractVector,AbstractMatrix},
                                 s::Union{AbstractVector,AbstractMatrix};
                                 delta::Real = 1.0f0, lambda::Real = 0.0f0,
                                 gamma::Real = 1.0f0, w_max::Real = 4.0f0)
    size(pred) == size(target) || throw(ArgumentError("pred and target must have the same shape"))
    residual = pred .- target
    if lambda > 0f0
        row_weights, _ = _peak_weight_and_score(target, u, s, lambda, gamma, w_max)
        total_w = sum(row_weights)
        row_weights = total_w > 0f0 ? row_weights ./ total_w : row_weights
    else
        row_weights = _same_device_fill(pred, 1f0, size(pred); T = Float32)
    end
    return Float32(sum(_huber_element.(residual, delta) .* row_weights))
end

"""
    peak_quantile_stats(target) -> (u, s)

Compute per-channel peak and spread statistics for the peak-weighted Huber
loss: `u` is the approximate 98th percentile, and `s` is the interquartile range
(IQR) used as a robust scale. These values are intended for train-split-only
precomputation and are safe to use as scalar summaries for Huber tuning.
"""
function peak_quantile_stats(target::AbstractMatrix)
    size(target, 2) > 0 || return Float32[], Float32[]
    u = Float32[]
    s = Float32[]
    for i in 1:size(target, 1)
        x = Float32.(view(target, i, :))
        q98 = quantile(x, 0.98)
        q25 = quantile(x, 0.25)
        q75 = quantile(x, 0.75)
        push!(u, Float32(max(q98, eps(Float32))))
        push!(s, Float32(max(q75 - q25, eps(Float32))))
    end
    return u, s
end

"""
    estimate_peak_loss_parameters(target; delta_scale = 1.5, lambda = 0.5,
                                 gamma = 1.0, w_max = 4.0) -> NamedTuple

Compute a lightweight first pass for Huber hyperparameters from the target data:
- `u` and `s` are the 98th-percentile / IQR estimates per channel.
- `delta` is set from the median absolute residual scale in the target channel.
- `w_max` is capped from the empirical peak-weight distribution.
"""
function estimate_peak_loss_parameters(target::AbstractMatrix;
                                      delta_scale::Real = 1.5f0,
                                      lambda::Real = 0.5f0,
                                      gamma::Real = 1.0f0,
                                      w_max::Real = 4.0f0)
    u, s = peak_quantile_stats(target)
    row_medians = [median(Float32.(view(target, i, :))) for i in 1:size(target, 1)]
    abs_resid = [abs.(Float32.(view(target, i, :)) .- row_medians[i]) for i in 1:size(target, 1)]
    delta = Float32(max(median(vcat(abs_resid...)), eps(Float32)) * Float32(delta_scale))
    return (; delta = delta, lambda = Float32(lambda), gamma = Float32(gamma),
            w_max = Float32(max(w_max, 1.0f0)), u = Float32.(u), s = Float32.(s))
end

"""
    peak_loss_summary(pred, target, u, s; delta = 1.0, lambda = 0.0,
                      gamma = 1.0, w_max = 4.0) -> NamedTuple

Tier-1 diagnostic summary for peak-focused loss tuning. Returns the weighted
Huber loss together with mean and max peak weights, which helps the operator
judge whether the loss is disproportionately emphasising the high-end tail.
`u`/`s` accept the same per-channel-vector or per-node-matrix shapes as
[`peak_weighted_huber_loss`](@ref).
"""
function peak_loss_summary(pred::AbstractMatrix, target::AbstractMatrix,
                          u::Union{AbstractVector,AbstractMatrix},
                          s::Union{AbstractVector,AbstractMatrix};
                          delta::Real = 1.0f0, lambda::Real = 0.0f0,
                          gamma::Real = 1.0f0, w_max::Real = 4.0f0)
    size(pred) == size(target) || throw(ArgumentError("pred and target must have the same shape"))
    if lambda > 0f0
        weights, _ = _peak_weight_and_score(target, u, s, lambda, gamma, w_max)
    else
        weights = _same_device_fill(pred, 1f0, size(pred); T = Float32)
    end
    loss = peak_weighted_huber_loss(pred, target, u, s;
                                   delta = delta, lambda = lambda,
                                   gamma = gamma, w_max = w_max)
    return (; loss = Float32(loss), w_mean = Float32(mean(weights)),
            w_max = Float32(maximum(weights)), w_min = Float32(minimum(weights)))
end

"""
    peak_weight_matrix(target, u, s; delta = 1.0, lambda = 0.0, gamma = 1.0,
                       w_max = 4.0) -> (weights::Matrix{Float32}, mask::BitMatrix)

Forward-only (non-differentiable) computation of the per-element peak weights
`w_i,t` used by [`peak_weighted_huber_loss`](@ref), plus a `BitMatrix` marking
which elements are "peak" cells (`target` strictly above the `u`/`s` threshold,
i.e. `peak_score > 0`). `u`/`s` accept the same per-channel `AbstractVector` or
per-node `AbstractMatrix` (`(nvar, nnode)`) shapes as
[`peak_weighted_huber_loss`](@ref).

Intended for Tier-1 loss-tuning diagnostics (`C_peak`, weight summary stats,
`RMSE_high`/`MAE_high`) where a separate, mutation-free computation of the
weights/mask is convenient (e.g. inside `Flux.ignore_derivatives`).
"""
function peak_weight_matrix(target::AbstractMatrix,
                           u::Union{AbstractVector,AbstractMatrix},
                           s::Union{AbstractVector,AbstractMatrix};
                           lambda::Real = 0.0f0, gamma::Real = 1.0f0, w_max::Real = 4.0f0)
    if lambda > 0f0
        weights, peak_scores = _peak_weight_and_score(target, u, s, lambda, gamma, w_max)
        mask = peak_scores .> 0f0
    else
        weights = _same_device_fill(target, 1f0, size(target); T = Float32)
        mask    = _same_device_falses(target)
    end
    return weights, mask
end

"""
    loss_function(model, batch, strategy, static; peak_stats = nothing) -> Float32

Multi-step rollout loss on a collated batch (`Vector{GNNGraph}` of length
`strategy.current_steps + 1`, each element a batched `GNNGraph`).

At each step `t`:
1. Optionally add Gaussian noise (std `strategy.noise_scale`) to state/forcing.
2. Forward the model to get `pred_state`.
3. Accumulate MSE (or, when `strategy.loss_type == :huber`, the peak-weighted
   Huber loss) against `batch[t+1].ndata.state`.
4. Carry `pred_state` forward; ground-truth forcing from `batch[t+1]` is used next.

`peak_stats`, used only when `strategy.loss_type == :huber`, supplies the
per-node peak threshold/scale (`u_i`/`s_i`) precomputed on the training split
by [`peak_node_stats`](@ref): a `NamedTuple` `(q = (u, s), h = (u, s))` with
each `u`/`s` an `AbstractVector{Float32}` of length `n_nodes` (the node count
of a *single* graph, i.e. `graphs[1].num_nodes` — NOT the batched/collated
node count). When `batch` is a collated multi-graph batch (`batchsize > 1`),
the per-node vector is tiled across the batch dimension to match the
block-diagonal node concatenation order used by `GNNGraphs.batch`. When
`nothing` (the default), a coarse per-batch fallback threshold
(`maximum(abs, target)`, shared by every node) is used instead — this keeps
the `:huber` path usable without precomputed stats, but the per-node stats are
what the design notes call `u_i`/`s_i` and should be supplied for real
training runs.

Returns mean loss across all steps.
"""
function loss_function(model      ::WflowGNN,
                       batch      ::Vector{<:GNNGraph},
                       strategy   ::TrainingStrategy,
                       static     ::AbstractMatrix;
                       peak_stats = nothing)
    nsteps      = strategy.current_steps
    noise_scale = strategy.noise_scale
    length(batch) >= nsteps + 1 ||
        throw(ArgumentError("batch length ($(length(batch))) must be >= current_steps+1 ($(nsteps+1))"))

    g_topo, state, forcings, forcings_next, targets = Flux.ignore_derivatives() do
        g    = batch[1]
        st   = g.ndata.state
        fs   = [batch[t].ndata.forcing     for t in 1:nsteps]
        fsn  = [batch[t + 1].ndata.forcing for t in 1:nsteps]
        tgts = [batch[t + 1].ndata.state   for t in 1:nsteps]
        g, st, fs, fsn, tgts
    end

    use_ckpt = nsteps > 1

    loss = 0f0
    for t in 1:nsteps
        forcing      = forcings[t]
        forcing_next = forcings_next[t]
        if noise_scale > 0f0
            state   = state   .+ noise_scale .* randn(Float32, size(state))
            forcing = forcing .+ noise_scale .* randn(Float32, size(forcing))
        end
        pred_state = use_ckpt ?
            Flux.Zygote.checkpointed(model, g_topo, state, forcing, static, forcing_next) :
            model(g_topo, state, forcing, static, forcing_next)

        if strategy.loss_type == :huber
            q_target = targets[t][1:1, :]
            h_target = targets[t][2:2, :]
            q_pred = pred_state[1:1, :]
            h_pred = pred_state[2:2, :]
            q_u, q_s, h_u, h_s = _resolve_peak_thresholds(peak_stats, q_target, h_target)
            q_loss = peak_weighted_huber_loss(q_pred, q_target, q_u, q_s;
                                              delta = strategy.peak_delta,
                                              lambda = strategy.peak_lambda,
                                              gamma = strategy.peak_gamma,
                                              w_max = strategy.peak_w_max)
            h_loss = strategy.h_loss_weight * peak_weighted_huber_loss(h_pred, h_target, h_u, h_s;
                                                                       delta = strategy.peak_delta,
                                                                       lambda = strategy.peak_lambda,
                                                                       gamma = strategy.peak_gamma,
                                                                       w_max = strategy.peak_w_max)
            loss += q_loss + h_loss
        else
            loss += Flux.mse(pred_state[1:1, :], targets[t][1:1, :]) +
                    strategy.h_loss_weight * Flux.mse(pred_state[2:2, :], targets[t][2:2, :])
        end
        state = pred_state
    end
    return loss / nsteps
end

"""
    one_step_loss(model, batch) -> Float32

1-step-ahead MSE on a collated batch. Uses `batch[1]` as input and
`batch[2].ndata.state` as target.
"""
function one_step_loss(model::WflowGNN, batch::Vector{<:GNNGraph},
                       static::AbstractMatrix,
                       h_loss_weight::Float32 = 1f0)
    g, state, forcing, forcing_next, target = Flux.ignore_derivatives() do
        g = batch[1]
        g, g.ndata.state, g.ndata.forcing, batch[2].ndata.forcing, batch[2].ndata.state
    end
    pred = model(g, state, forcing, static, forcing_next)
    Flux.mse(pred[1:1, :], target[1:1, :]) +
        h_loss_weight * Flux.mse(pred[2:2, :], target[2:2, :])
end

"""
    loss_components(model, batch) -> (q_mse, h_mse)

Compute unweighted 1-step-ahead MSE separately for q (row 1) and h (row 2) of
the state. Returns `(nothing, nothing)` when the model has no mass balance.

Runs entirely under `Flux.ignore_derivatives` — for diagnostic reporting only.
"""
function loss_components(model::WflowGNN, batch::Vector{<:GNNGraph}, static::AbstractMatrix)
    isnothing(model.mass_balance) && return (nothing, nothing)
    Flux.ignore_derivatives() do
        g      = batch[1]
        pred   = model(g, g.ndata.state, g.ndata.forcing, static, batch[2].ndata.forcing)
        target = batch[2].ndata.state
        Flux.mse(pred[1:1, :], target[1:1, :]),
        Flux.mse(pred[2:2, :], target[2:2, :])
    end
end

"""
    mb_amplification(model, batch, static) -> (amp, gain)

Diagnostic: how strongly a one-step discharge (`q`) error is amplified into a
water-depth (`h`) error by the hard mass-balance decoder.

This is teacher-forced (input state/forcing are ground truth), so the
counterfactual depth `h_ref = MB(q_true, …)` isolates the h error caused
**purely by the q error** from the wflow structural mismatch of the mass-balance
assumption. Because the model's own `h_pred = MB(q_pred, …)`, the difference
`h_pred − h_ref` is exactly the mass balance's response to the q error, including
upstream routing and the ≥0 floors.

Returns (normalised units):
- `amp`  : `RMS(h_pred − h_ref) / RMS(q_pred − q_true)` — the realised
           amplification. `amp > 1` ⇒ q errors are magnified into h.
- `gain` : the analytic per-node self-gain `|∂h_norm/∂q_norm| = θ·dt·σ_q/σ_h`,
           a reference the realised `amp` can be compared against.

Returns `(NaN32, NaN32)` when the model has no mass balance. Non-differentiable;
for diagnostic reporting only.
"""
function mb_amplification(model::WflowGNN, batch::Vector{<:GNNGraph}, static::AbstractMatrix)
    mb = model.mass_balance
    isnothing(mb) && return (NaN32, NaN32)
    Flux.ignore_derivatives() do
        g            = batch[1]
        state        = g.ndata.state
        forcing      = g.ndata.forcing
        forcing_next = batch[2].ndata.forcing
        target       = batch[2].ndata.state

        pred   = model(g, state, forcing, static, forcing_next)
        q_pred = pred[1:1, :]
        h_pred = pred[2:2, :]
        q_true = target[1:1, :]
        h_ref  = mb(g, state, forcing, forcing_next, q_true)  # counterfactual: perfect Q

        dq    = q_pred .- q_true
        dh    = h_pred .- h_ref
        denom = sqrt(mean(abs2, dq))
        amp   = denom > 0f0 ? Float32(sqrt(mean(abs2, dh)) / denom) : NaN32
        gain  = Float32(mb.θ * mb.dt * mb.σ_q / mb.σ_h)
        (amp, gain)
    end
end


