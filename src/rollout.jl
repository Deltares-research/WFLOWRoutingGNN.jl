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
