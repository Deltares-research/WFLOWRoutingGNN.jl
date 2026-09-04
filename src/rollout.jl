"""
    rollout(model, g0, static, forcing; device = :cpu, timesteps = nothing) -> Array{Float32, 3}

Perform an autoregressive rollout over `timesteps` steps (default: all steps
in `forcing`).

Arguments:
- `model`     : a `WflowGNN`.
- `g0`        : initial `GNNGraph` with `ndata.state` (n_state × n_nodes) giving
                the state at t = 0, and the graph topology.
- `static`    : `AbstractMatrix{Float32}` of shape `(n_static, n_nodes)` with
                time-invariant node features (shared across all timesteps).
- `forcing`   : `AbstractArray` of shape `(n_forcing, n_nodes, T)` with the
                forcing inputs for timesteps t = 1 … T.
- `device`    : `:cpu` or `:gpu`. The model, graph, static, and forcing are moved
                to this device before the rollout; results are always returned on CPU.
- `timesteps` : number of autoregressive steps to perform. Must be ≤ `T`.
                `nothing` (default) means all `T` steps.
- `return_timing` : when `true`, return a `NamedTuple`
                `(states, step_times, total_time, n_steps)` instead of just the
                state array — `step_times` is the per-timestep wall time (s).

Returns an `Array{Float32, 3}` of shape `(n_state, n_nodes, timesteps)` on the CPU
(or the timing `NamedTuple` when `return_timing = true`).
"""
function rollout(model, g0::GNNGraph, static::AbstractMatrix{Float32},
                 forcing::AbstractArray{<:Real, 3};
                 device::Symbol = :cpu,
                 timesteps::Union{Int, Nothing} = nothing,
                 return_timing::Bool = false)
    device in (:cpu, :gpu) || throw(ArgumentError("device must be :cpu or :gpu"))
    dev_fn = device == :gpu ? Flux.gpu : Flux.cpu

    model_d   = dev_fn(model)
    g0_d      = dev_fn(g0)
    static_d  = dev_fn(Array{Float32}(static))
    forcing_d = dev_fn(Array{Float32}(forcing))

    T_max = size(forcing_d, 3)
    T     = isnothing(timesteps) ? T_max :
            (1 ≤ timesteps ≤ T_max ? timesteps :
             throw(ArgumentError("timesteps ($timesteps) must be between 1 and $T_max")))

    state  = g0_d.ndata.state

    n_state = size(state, 1)
    n_nodes = size(state, 2)

    states_d = similar(state, n_state, n_nodes, T)

    # Device synchronisation: on GPU, model calls launch asynchronously, so
    # per-step wall-clock deltas are meaningless without a sync.  No-op on CPU.
    sync_dev() = device == :gpu ? CUDA.synchronize() : nothing

    # Warm-up: run (and discard) one step so the reported timings exclude
    # first-call JIT compilation, which otherwise inflates the mean.
    let fwarm = forcing_d[:, :, min(2, T_max)]
        model_d(g0_d, state, forcing_d[:, :, 1], static_d, fwarm)
    end
    sync_dev()

    # Use `time_ns()` (not `time()`): GPU steps are only a few ms and Windows
    # `time()` quantises to ~1 ms, which would corrupt them.  Same nanosecond
    # counter + sync-after-call convention as scripts/benchmark_inference_vs_train.jl,
    # so per-step numbers here are conceptually identical to that benchmark.
    t_start = time_ns()
    step_times = Vector{Float64}(undef, T)

    for t in 1:T
        t_step = time_ns()
        forcing_next_t    = forcing_d[:, :, min(t + 1, T_max)]
        state             = model_d(g0_d, state, forcing_d[:, :, t], static_d, forcing_next_t)
        states_d[:, :, t] = state
        sync_dev()
        step_times[t] = (time_ns() - t_step) / 1e9
    end

    t_total = (time_ns() - t_start) / 1e9
    med_step  = median(step_times)
    mean_step = sum(step_times) / T
    std_step  = sqrt(sum((step_times .- mean_step).^2) / T)
    @info @sprintf("rollout: %d steps  total=%.3f s  median/step=%.4f s  mean/step=%.4f s  std/step=%.4f s  min=%.4f s  max=%.4f s",
                   T, t_total, med_step, mean_step, std_step, minimum(step_times), maximum(step_times))

    states_cpu = Array{Float32}(Flux.cpu(states_d))
    return return_timing ?
        (states = states_cpu, step_times = step_times, total_time = t_total, n_steps = T) :
        states_cpu
end

"""
    rollout_ensemble(model, g0, static, forcing;
                     states0 = nothing, device = :cpu, timesteps = nothing) -> Array{Float32, 4}

Perform `B` autoregressive rollouts in parallel — one per ensemble member —
using the block-diagonal batching machinery.  All members share the graph
topology of `g0`; they differ in their per-member forcing (and, optionally,
initial state).  Because the members occupy disjoint diagonal blocks of the
batched adjacency, they never interact — each block is an independent rollout,
but every timestep is evaluated with a single batched forward pass, amortising
kernel-launch latency across the ensemble (the dominant per-step cost on GPU).

The block-diagonal adjacency for `B` is precomputed once so every step takes the
single-SpMM path; the reshape-fallback path is deliberately not used here.

Arguments:
- `model`   : a `WflowGNN`.
- `g0`      : initial `GNNGraph` giving the shared topology.  Its `ndata.state`
              is used as the initial condition for every member when `states0`
              is `nothing`.
- `static`  : `(n_static, n_nodes)` time-invariant node features (shared).
- `forcing` : `(n_forcing, n_nodes, T, B)` per-member forcing for t = 1 … T.
              The ensemble size `B` is `size(forcing, 4)`.

Keyword arguments:
- `states0`    : optional `(n_state, n_nodes, B)` per-member initial state.
                 `nothing` (default) replicates `g0.ndata.state` across members.
- `device`     : `:cpu` or `:gpu`.
- `timesteps`  : number of steps (≤ `T`); `nothing` means all `T`.
- `return_timing` : when `true`, return a `NamedTuple`
                 `(states, step_times, total_time, n_steps, n_members)` instead
                 of just the state array — `step_times` is the per-timestep wall
                 time (s) for the whole batched step (all `B` members together).

Returns an `Array{Float32, 4}` of shape `(n_state, n_nodes, timesteps, B)` on CPU
(or the timing `NamedTuple` when `return_timing = true`).
"""
function rollout_ensemble(model, g0::GNNGraph, static::AbstractMatrix{Float32},
                          forcing::AbstractArray{<:Real, 4};
                          states0::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                          device::Symbol = :cpu,
                          timesteps::Union{Int, Nothing} = nothing,
                          return_timing::Bool = false)
    device in (:cpu, :gpu) || throw(ArgumentError("device must be :cpu or :gpu"))
    dev_fn = device == :gpu ? Flux.gpu : Flux.cpu

    N = g0.num_nodes
    B = size(forcing, 4)
    size(forcing, 2) == N ||
        throw(ArgumentError("forcing has $(size(forcing, 2)) nodes but g0 has $N"))

    T_max = size(forcing, 3)
    T     = isnothing(timesteps) ? T_max :
            (1 ≤ timesteps ≤ T_max ? timesteps :
             throw(ArgumentError("timesteps ($timesteps) must be between 1 and $T_max")))

    # Base per-member initial state (n_state × n_nodes).
    base_state = isnothing(states0) ? g0.ndata.state : nothing
    if !isnothing(states0)
        size(states0, 2) == N || throw(ArgumentError("states0 has $(size(states0, 2)) nodes but g0 has $N"))
        size(states0, 3) == B || throw(ArgumentError("states0 has $(size(states0, 3)) members but forcing has $B"))
    end
    n_state = isnothing(states0) ? size(base_state, 1) : size(states0, 1)

    # Batched B·N-node graph: B disjoint copies of g0's topology.  The mass
    # balance layer reads `g.num_nodes`, so this must equal B·N.
    gB = GNNGraphs.batch([g0 for _ in 1:B])

    # Precompute the block-diagonal adjacency for B *before* moving to device so
    # every step takes the correct single-SpMM path (the reshape fallback does
    # not preserve per-member independence and is intentionally avoided).
    model_b = precompute_batched(model, B)

    model_d   = dev_fn(model_b)
    gB_d      = dev_fn(gB)
    static_d  = dev_fn(Array{Float32}(static))
    forcing_d = dev_fn(Array{Float32}(forcing))

    # Initial state stacked in block order [member1 (N cols) | member2 | … ].
    # reshape of an (n_state, N, B) array collapses (N, B) column-major → N fast,
    # B slow, which is exactly block order.
    state = if isnothing(states0)
        repeat(dev_fn(Array{Float32}(base_state)), 1, B)   # identical IC per member
    else
        reshape(dev_fn(Array{Float32}(states0)), n_state, N * B)
    end

    states_d = similar(state, n_state, N * B, T)

    sync_dev() = device == :gpu ? CUDA.synchronize() : nothing

    # Per-step forcing slice → block order (n_forcing, N·B).  The slice is
    # materialised (not a view) because reshaping a non-contiguous view of a
    # 4-D array is unsupported on the GPU.
    nf = size(forcing_d, 1)
    fslice(t) = reshape(forcing_d[:, :, t, :], nf, N * B)

    # Warm-up (discard) so timings exclude first-call compilation.
    model_d(gB_d, state, fslice(1), static_d, fslice(min(2, T_max)))
    sync_dev()

    # Nanosecond counter + sync-after-call, matching
    # scripts/benchmark_inference_vs_train.jl (see note in `rollout`): avoids the
    # ~1 ms Windows `time()` quantisation on few-ms GPU steps.
    t_start = time_ns()
    step_times = Vector{Float64}(undef, T)

    for t in 1:T
        t_step = time_ns()
        f_t               = fslice(t)
        f_next            = fslice(min(t + 1, T_max))
        state             = model_d(gB_d, state, f_t, static_d, f_next)
        states_d[:, :, t] = state
        sync_dev()
        step_times[t] = (time_ns() - t_step) / 1e9
    end

    t_total  = (time_ns() - t_start) / 1e9
    med_step = median(step_times)
    @info @sprintf("rollout_ensemble: B=%d members  %d steps  total=%.3f s  median/step=%.4f s  min=%.4f s  max=%.4f s",
                   B, T, t_total, med_step, minimum(step_times), maximum(step_times))

    # Un-stack: (n_state, N·B, T) → (n_state, N, B, T) → (n_state, N, T, B).
    states_cpu = Array{Float32}(Flux.cpu(states_d))
    result = permutedims(reshape(states_cpu, n_state, N, B, T), (1, 2, 4, 3))
    return return_timing ?
        (states = result, step_times = step_times, total_time = t_total,
         n_steps = T, n_members = B) :
        result
end


"""
    evaluate_trajectory(model, split, norm_stats, domain, static; device = :cpu)
        -> (pred_states, true_states)

Evaluate the model on an entire split of `make_horizon_dataset` by performing
a single autoregressive rollout over the reconstructed consecutive timeseries.

Steps:
1. Flatten the overlapping windows back into a consecutive `GNNGraph` timeseries
   of `T` unique timesteps.
2. Build the forcing array `(n_forcing, n_nodes, T-1)` from graphs t = 1 … T-1.
3. Call `rollout` using the state at t = 1 as the initial condition, producing
   predicted states at t = 2 … T.
4. Undo z-score normalization on both the predicted and ground-truth state
   arrays using `norm_stats` and `DOMAIN_VARS[domain]["state"]`.

Arguments:
- `model`      : a `WflowGNN`.
- `split`      : one of the splits returned by `make_horizon_dataset` — a
                 `Vector{Vector{GNNGraph}}` of consecutive overlapping windows.
- `norm_stats` : normalisation statistics as returned by `build_wflow_graph`,
                 mapping variable names to `(mean, std)` named tuples.
- `domain`     : routing domain string (key of `DOMAIN_VARS`).
- `static`     : `AbstractMatrix{Float32}` of shape `(n_static, n_nodes)` with
                 time-invariant node features (fifth return value of `build_wflow_graph`).
- `device`     : `:cpu` or `:gpu`. Passed to `rollout`; model and data are
                 moved to this device regardless of their current location.

Returns `(pred_states, true_states)`, each an `Array{Float32,3}` of shape
`(n_state, n_nodes, T-1)` in physical (un-normalised) units on the CPU.
"""
function evaluate_trajectory(model, split, norm_stats, domain::String,
                             static::AbstractMatrix{Float32};
                             device::Symbol = :cpu,
                             postscale::Dict{String,Vector{Float32}} = Dict{String,Vector{Float32}}())
    isempty(split) && throw(ArgumentError("split must not be empty"))

    # --- 1. Flatten overlapping windows to a consecutive timeseries ----------
    # Windows are [t : t+nhorizon-1]; take the first window in full, then
    # only the last (new) graph from each subsequent window.
    graphs = vcat(split[1], [w[end] for w in split[2:end]])
    T = length(graphs)
    T >= 2 || throw(ArgumentError("split must contain at least 2 unique timesteps"))

    g0        = graphs[1]
    n_nodes   = g0.num_nodes
    n_state   = size(g0.ndata.state,   1)
    n_forcing = size(g0.ndata.forcing, 1)

    # --- 2. Forcing array (n_forcing × n_nodes × T-1) -----------------------
    forcing = Array{Float32}(undef, n_forcing, n_nodes, T - 1)
    for t in 1:(T - 1)
        forcing[:, :, t] = graphs[t].ndata.forcing
    end

    # --- 3. Autoregressive rollout (always returns CPU array) ---------------
    pred_states = rollout(model, g0, static, forcing; device)

    # --- 4. Ground-truth states at t = 2 … T --------------------------------
    true_states = Array{Float32}(undef, n_state, n_nodes, T - 1)
    for t in 1:(T - 1)
        true_states[:, :, t] = graphs[t + 1].ndata.state
    end

    # --- 5. Undo z-score normalisation on state variables -------------------
    state_vars = DOMAIN_VARS[domain]["state"]
    for (vi, vname) in enumerate(state_vars)
        μ = Float32(norm_stats[vname].mean)
        σ = Float32(norm_stats[vname].std)
        pred_states[vi, :, :] .= pred_states[vi, :, :] .* σ .+ μ
        true_states[vi, :, :] .= true_states[vi, :, :] .* σ .+ μ
        # Undo any per-node preprocessing applied before z-score normalisation
        if haskey(postscale, vname)
            scale = postscale[vname]   # length n_nodes
            pred_states[vi, :, :] .*= scale
            true_states[vi, :, :] .*= scale
        end
    end

    return pred_states, true_states
end

# ---------------------------------------------------------------------------
# Fixed-horizon validation metric (constant-length rollout from many anchors)
# ---------------------------------------------------------------------------

"""
    FixedHorizonEval

Precomputed, batched inputs for a **constant-length** autoregressive validation
rollout used as an epoch-to-epoch comparable metric (unlike the curriculum
`val_rollout`, whose horizon grows across phases and so is not comparable). Built
once by [`build_fixed_horizon_eval`](@ref) and consumed each epoch by
[`fixed_horizon_metrics`](@ref).

`B` evenly-spaced anchor start points in the validation timeseries are each
rolled out `horizon` steps. The anchors are stacked as `B` disjoint copies of the
graph topology (a block-diagonal batch) so every rollout step is a single batched
forward pass (the model's reshape-fallback routing keeps the anchors independent).

Fields:
- `gB`          : batched `GNNGraph` (`B` disjoint copies of the topology).
- `static`      : `(n_static, N)` per-node constants for a single graph (the
                  forward pass tiles them across the batch).
- `forcing`     : `(n_forcing, N·B, horizon)` per-step forcing in block order.
- `states0`     : `(n_state, N·B)` initial state in block order.
- `anchor_starts`: length-`B` start indices in the flattened validation
                  timeseries (1-based graph index).
- `start_q_phys`: length-`B` anchor start-state discharge summary (basin-mean
                  physical q at the anchor start step).
- `start_q_percentile`: length-`B` empirical percentile of `start_q_phys`
                  within the flattened validation-timeseries start-flow
                  distribution (range `[0, 1]`).
- `true_q_phys` : `(N, horizon, B)` ground-truth discharge in physical units.
- `true_peak`   : global `max|true_q_phys|` (reference for the peak ratio).
- `qi`          : row index of discharge in the state.
- `q_mu`, `q_sigma`, `q_postscale` : denormalisation constants for discharge.
- `horizon`, `N`, `B`.
"""
struct FixedHorizonEval
    gB          :: GNNGraph
    static      :: Matrix{Float32}
    forcing     :: Array{Float32, 3}
    states0     :: Matrix{Float32}
    anchor_starts :: Vector{Int}
    start_q_phys  :: Vector{Float32}
    start_q_percentile :: Vector{Float32}
    true_q_phys :: Array{Float32, 3}
    true_peak   :: Float32
    qi          :: Int
    q_mu        :: Float32
    q_sigma     :: Float32
    q_postscale :: Vector{Float32}
    horizon     :: Int
    N           :: Int
    B           :: Int
end

"""
    build_fixed_horizon_eval(split, static, norm_stats, domain, postscale;
                             horizon = 30, n_anchors = 32)
        -> FixedHorizonEval | nothing

Assemble a [`FixedHorizonEval`](@ref) from a validation `split` of
`make_horizon_dataset`. The overlapping windows are flattened to a consecutive
timeseries, then up to `n_anchors` evenly-spaced start points that each admit a
full `horizon`-step rollout are selected and batched.

Returns `nothing` (with a warning) when `horizon`/`n_anchors` are non-positive or
the split is too short to host a single anchor. If the split is shorter than
`horizon + 1` graphs the horizon is clamped down (with a warning).
"""
function build_fixed_horizon_eval(split, static::AbstractMatrix{Float32},
                                  norm_stats, domain::String,
                                  postscale::Dict{String,Vector{Float32}};
                                  horizon::Int = 30, n_anchors::Int = 32)
    (horizon > 0 && n_anchors > 0) || return nothing
    isempty(split) && return nothing

    # Flatten overlapping windows to a consecutive timeseries (as evaluate_trajectory).
    graphs = vcat(split[1], [w[end] for w in split[2:end]])
    T      = length(graphs)

    H = min(horizon, T - 1)
    if H < 1
        @warn "build_fixed_horizon_eval: validation split too short ($T graphs) for a fixed-horizon metric; disabling."
        return nothing
    end
    H < horizon &&
        @warn "build_fixed_horizon_eval: validation split holds only $T graphs; clamping fixed horizon $horizon → $H."

    # Anchor start s (1-based) needs graphs[s … s+H] available, so s ∈ 1:(T-H).
    n_avail = T - H
    B       = min(n_anchors, n_avail)
    starts  = unique(round.(Int, range(1, n_avail; length = B)))
    B       = length(starts)

    g0        = graphs[1]
    N         = g0.num_nodes
    n_state   = size(g0.ndata.state,   1)
    n_forcing = size(g0.ndata.forcing, 1)

    forcing     = Array{Float32}(undef, n_forcing, N * B, H)
    states0     = Array{Float32}(undef, n_state,   N * B)
    true_q_norm = Array{Float32}(undef, N, H, B)

    qi    = 1                                   # discharge is state row 1 (mass-balance convention)
    qname = DOMAIN_VARS[domain]["state"][qi]
    μq    = Float32(norm_stats[qname].mean)
    σq    = Float32(norm_stats[qname].std)
    qpost = get(postscale, qname, ones(Float32, N))

    # Start-state flow summary distribution over the full flattened validation
    # timeseries (basin-mean physical q at each graph step).
    q_start_dist = Vector{Float32}(undef, T)
    for t in 1:T
        qn = graphs[t].ndata.state[qi, :]
        qp = (qn .* σq .+ μq) .* qpost
        q_start_dist[t] = Float32(mean(qp))
    end

    # Anchor metadata for regime-conditioned diagnostics.
    start_q_phys = Vector{Float32}(undef, B)
    start_q_pct  = Vector{Float32}(undef, B)

    for (a, s) in enumerate(starts)
        cols = ((a - 1) * N + 1):(a * N)
        states0[:, cols] = graphs[s].ndata.state
        start_q_phys[a] = q_start_dist[s]
        start_q_pct[a]  = Float32(count(<=(q_start_dist[s]), q_start_dist) / length(q_start_dist))
        for k in 1:H
            forcing[:, cols, k]  = graphs[s + k - 1].ndata.forcing
            true_q_norm[:, k, a] = graphs[s + k].ndata.state[qi, :]
        end
    end

    true_q_phys = (true_q_norm .* σq .+ μq) .* reshape(qpost, N, 1, 1)
    true_peak   = Float32(maximum(abs, true_q_phys))
    gB          = GNNGraphs.batch([g0 for _ in 1:B])

    return FixedHorizonEval(gB, Matrix{Float32}(static), forcing, states0,
                            starts, start_q_phys, start_q_pct,
                            true_q_phys, true_peak, qi, μq, σq, qpost, H, N, B)
end

"""
    fixed_horizon_metrics(model, fh; device = :cpu) -> NamedTuple

Run the constant-length autoregressive rollout for all anchors of `fh` (batched
into one forward pass per step).

Returns:
- `rmse_q`: global discharge RMSE in physical units (pooled over all nodes,
  horizons and anchors).
- `peak_ratio`: global peak-amplification ratio `max|q_pred| / max|q_truth|`.
- `rmse_q_anchor`: length-`B` per-anchor RMSE (reduced over node × horizon).
- `peak_ratio_anchor`: length-`B` per-anchor peak ratio with a per-anchor truth
  peak denominator.

No gradients are taken; `model` may live on either device.
"""
function fixed_horizon_metrics(model::WflowGNN, fh::FixedHorizonEval; device::Symbol = :cpu)
    dev     = device == :gpu ? Flux.gpu : identity
    gB      = dev(fh.gB)
    static  = dev(fh.static)
    forcing = dev(fh.forcing)
    state   = dev(fh.states0)
    H       = fh.horizon
    qi      = fh.qi

    q_norm = Array{Float32}(undef, fh.N * fh.B, H)   # CPU accumulator of the q row
    for k in 1:H
        f_t    = forcing[:, :, k]
        f_next = forcing[:, :, min(k + 1, H)]
        state  = model(gB, state, f_t, static, f_next)
        q_norm[:, k] = Array(Flux.cpu(state[qi:qi, :]))[1, :]
    end

    # (N·B, H) block order → (N, B, H) → (N, H, B)
    pred_q_norm = permutedims(reshape(q_norm, fh.N, fh.B, H), (1, 3, 2))
    pred_q_phys = (pred_q_norm .* fh.q_sigma .+ fh.q_mu) .* reshape(fh.q_postscale, fh.N, 1, 1)

    err = pred_q_phys .- fh.true_q_phys
    rmse_q = sqrt(mean(abs2, err))
    peak_ratio = maximum(abs, pred_q_phys) / max(fh.true_peak, eps(Float32))

    rmse_q_anchor = Vector{Float32}(undef, fh.B)
    peak_ratio_anchor = Vector{Float32}(undef, fh.B)
    for a in 1:fh.B
        rmse_q_anchor[a] = Float32(sqrt(mean(abs2, @view(err[:, :, a]))))
        true_peak_a = maximum(abs, @view(fh.true_q_phys[:, :, a]))
        peak_ratio_anchor[a] = Float32(maximum(abs, @view(pred_q_phys[:, :, a])) /
                                       max(true_peak_a, eps(Float32)))
    end

    return (; rmse_q = Float32(rmse_q),
              peak_ratio = Float32(peak_ratio),
              rmse_q_anchor,
              peak_ratio_anchor)
end

"""
    rollout_mb_diagnostics(model, split) -> NamedTuple

Run a single autoregressive rollout over `split` and record physical mass-balance
terms at every step.  Also runs the mass balance with ground-truth q/h inputs to
verify the equation independently of the model.

Returns a NamedTuple with matrices of shape `(n_nodes, T)`:
- `pred_q`      [m³/s]: predicted discharge (denorm + postscale)
- `pred_h`      [m]:    water depth from MB applied to predicted q
- `true_q`      [m³/s]: ground-truth discharge
- `true_h`      [m]:    ground-truth water depth
- `upstream_q`  [m³/s]: sum of upstream q (using predicted q as input)
- `inwater`     [m³/s]: lateral inflow at each step
- `net_flux`    [m³/s]: upstream_q + inwater - q_out  (using predicted q)
- `h_raw`       [m]:    h before the ≥0 floor (using predicted q)
- `mb_verify_h` [m]:    h from MB fed true q/h — verifies the equation itself
"""
function rollout_mb_diagnostics(model::WflowGNN, split, static::AbstractMatrix{Float32})
    isnothing(model.mass_balance) &&
        throw(ArgumentError("rollout_mb_diagnostics requires a MassBalanceLayer"))
    mb = model.mass_balance

    # Flatten windows to a consecutive timeseries (same logic as evaluate_trajectory)
    graphs = vcat(split[1], [w[end] for w in split[2:end]])
    T      = length(graphs) - 1
    T >= 1 || throw(ArgumentError("split must contain at least 2 unique timesteps"))

    g0      = graphs[1]
    n_nodes = g0.num_nodes

    # Output matrices (n_nodes × T)
    pred_q      = Matrix{Float32}(undef, n_nodes, T)
    pred_h      = Matrix{Float32}(undef, n_nodes, T)
    true_q      = Matrix{Float32}(undef, n_nodes, T)
    true_h      = Matrix{Float32}(undef, n_nodes, T)
    upstream_q  = Matrix{Float32}(undef, n_nodes, T)
    inwater     = Matrix{Float32}(undef, n_nodes, T)
    net_flux    = Matrix{Float32}(undef, n_nodes, T)
    h_raw       = Matrix{Float32}(undef, n_nodes, T)
    mb_verify_h = Matrix{Float32}(undef, n_nodes, T)

    state_pred = g0.ndata.state  # normalised, starts from true initial condition

    for t in 1:T
        forcing_t    = graphs[t].ndata.forcing
        forcing_next = graphs[t + 1].ndata.forcing   # graphs has T+1 elements
        target_state = graphs[t + 1].ndata.state

        # One autoregressive step (fully-implicit: pass forcing_next)
        state_pred = model(g0, state_pred, forcing_t, static, forcing_next)

        # --- Diagnostics: MB with predicted q --------------------------------
        d = mb_diagnostics(mb, g0, state_pred, forcing_t, forcing_next, state_pred[1:1, :])
        pred_q[:,     t] = d.q_phys_new
        pred_h[:,     t] = d.h_phys_new
        upstream_q[:, t] = d.upstream_q
        inwater[:,    t] = d.inwater_phys
        net_flux[:,   t] = d.net_flux
        h_raw[:,      t] = d.h_phys_raw

        # --- Ground truth (physical units, denorm + postscale) ---------------
        # Undo z-score AND the variable postscale so these match every other
        # series here (all sourced from mb_diagnostics, which applies postscale).
        true_q[:, t] = (vec(target_state[1:1, :]) .* mb.σ_q .+ mb.μ_q) .*
                       Array(mb.postscale_q)
        true_h[:, t] = (vec(target_state[2:2, :]) .* mb.σ_h .+ mb.μ_h) .*
                       Array(mb.postscale_h)

        # --- Verification: MB fed true q, from true previous state -----------
        d_v = mb_diagnostics(mb, g0, graphs[t].ndata.state, forcing_t, forcing_next,
                             target_state[1:1, :])
        mb_verify_h[:, t] = d_v.h_phys_new
    end

    return (pred_q      = pred_q,
            pred_h      = pred_h,
            true_q      = true_q,
            true_h      = true_h,
            upstream_q  = upstream_q,
            inwater     = inwater,
            net_flux    = net_flux,
            h_raw       = h_raw,
            mb_verify_h = mb_verify_h)
end
