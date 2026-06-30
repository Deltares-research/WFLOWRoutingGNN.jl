"""
    rollout(model, g0, forcing; device = :cpu, timesteps = nothing) -> Array{Float32, 3}

Perform an autoregressive rollout over `timesteps` steps (default: all steps
in `forcing`).

Arguments:
- `model`     : a `WflowGNN`.
- `g0`        : initial `GNNGraph` with `ndata.state` (n_state × n_nodes) giving
                the state at t = 0, `ndata.static` (n_static × n_nodes) for the
                time-invariant features, and the graph topology.
- `forcing`   : `AbstractArray` of shape `(n_forcing, n_nodes, T)` with the
                forcing inputs for timesteps t = 1 … T.
- `device`    : `:cpu` or `:gpu`. The model, graph, and forcing are moved to this
                device before the rollout; results are always returned on CPU.
- `timesteps` : number of autoregressive steps to perform. Must be ≤ `T`.
                `nothing` (default) means all `T` steps.

Returns an `Array{Float32, 3}` of shape `(n_state, n_nodes, timesteps)` on the CPU.
"""
function rollout(model, g0::GNNGraph, forcing::AbstractArray{<:Real, 3};
                 device::Symbol = :cpu,
                 timesteps::Union{Int, Nothing} = nothing)
    device in (:cpu, :gpu) || throw(ArgumentError("device must be :cpu or :gpu"))
    dev_fn = device == :gpu ? Flux.gpu : Flux.cpu

    model_d   = dev_fn(model)
    g0_d      = dev_fn(g0)
    forcing_d = dev_fn(Array{Float32}(forcing))

    T_max = size(forcing_d, 3)
    T     = isnothing(timesteps) ? T_max :
            (1 ≤ timesteps ≤ T_max ? timesteps :
             throw(ArgumentError("timesteps ($timesteps) must be between 1 and $T_max")))

    static = g0_d.ndata.static
    state  = g0_d.ndata.state

    n_state = size(state, 1)
    n_nodes = size(state, 2)

    states_d = similar(state, n_state, n_nodes, T)

    for t in 1:T
        state             = model_d(g0_d, state, forcing_d[:, :, t], static)
        states_d[:, :, t] = state
    end

    return Array{Float32}(Flux.cpu(states_d))
end

"""
    evaluate_trajectory(model, split, norm_stats, domain; device = :cpu)
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
- `device`     : `:cpu` or `:gpu`. Passed to `rollout`; model and data are
                 moved to this device regardless of their current location.

Returns `(pred_states, true_states)`, each an `Array{Float32,3}` of shape
`(n_state, n_nodes, T-1)` in physical (un-normalised) units on the CPU.
"""
function evaluate_trajectory(model, split, norm_stats, domain::String;
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
    pred_states = rollout(model, g0, forcing; device)

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
