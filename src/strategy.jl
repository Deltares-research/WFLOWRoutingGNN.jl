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
    current_steps  :: Int
end

"""
    TrainingStrategy(steps, durations, noise_scale = 0) -> TrainingStrategy

Construct a `TrainingStrategy`. `current_steps` is initialised to `steps[1]`.
"""
function TrainingStrategy(steps, durations, noise_scale = 0; h_loss_weight = 1f0)
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
    steps_v = convert(Vector{Int}, steps)
    TrainingStrategy(steps_v,
                     convert(Vector{Int}, durations),
                     Float32(noise_scale),
                     Float32(h_loss_weight),
                     steps_v[1])
end

function Base.show(io::IO, s::TrainingStrategy)
    println(io, "TrainingStrategy:")
    println(io, "  steps          : ", s.steps)
    println(io, "  durations      : ", s.durations)
    println(io, "  noise_scale    : ", s.noise_scale)
    println(io, "  h_loss_weight  : ", s.h_loss_weight)
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
        Float32(get(d, "noise_scale", 0.0)),
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

"""
    loss_function(model, batch, strategy) -> Float32

Multi-step rollout loss on a collated batch (`Vector{GNNGraph}` of length
`strategy.current_steps + 1`, each element a batched `GNNGraph`).

At each step `t`:
1. Optionally add Gaussian noise (std `strategy.noise_scale`) to state/forcing.
2. Forward the model to get `pred_state`.
3. Accumulate MSE against `batch[t+1].ndata.state`.
4. Carry `pred_state` forward; ground-truth forcing from `batch[t+1]` is used next.

Returns mean MSE across all steps.
"""
function loss_function(model    ::WflowGNN,
                       batch    ::Vector{<:GNNGraph},
                       strategy ::TrainingStrategy)
    nsteps      = strategy.current_steps
    noise_scale = strategy.noise_scale
    length(batch) >= nsteps + 1 ||
        throw(ArgumentError("batch length ($(length(batch))) must be >= current_steps+1 ($(nsteps+1))"))

    # Extract all arrays from GNNGraph.ndata outside the diff path.
    # GNNGraph.ndata uses a Dict internally; Zygote cannot accumulate Dict
    # tangents, so we mark these reads as non-differentiable constants.
    g_topo, state, static, forcings, forcings_next, targets = Flux.ignore_derivatives() do
        g    = batch[1]
        st   = g.ndata.state
        sl   = g.ndata.static
        fs   = [batch[t].ndata.forcing     for t in 1:nsteps]
        fsn  = [batch[t + 1].ndata.forcing for t in 1:nsteps]  # fully-implicit: iw[t+1]
        tgts = [batch[t + 1].ndata.state   for t in 1:nsteps]
        g, st, sl, fs, fsn, tgts
    end

    loss = 0f0
    for t in 1:nsteps
        forcing      = forcings[t]
        forcing_next = forcings_next[t]
        if noise_scale > 0f0
            state   = state   .+ noise_scale .* randn(Float32, size(state))
            forcing = forcing .+ noise_scale .* randn(Float32, size(forcing))
        end
        pred_state = model(g_topo, state, forcing, static, forcing_next)
        loss += Flux.mse(pred_state[1:1, :], targets[t][1:1, :]) +
                strategy.h_loss_weight * Flux.mse(pred_state[2:2, :], targets[t][2:2, :])
        state      = pred_state
    end
    return loss / nsteps
end

"""
    one_step_loss(model, batch) -> Float32

1-step-ahead MSE on a collated batch. Uses `batch[1]` as input and
`batch[2].ndata.state` as target.
"""
function one_step_loss(model::WflowGNN, batch::Vector{<:GNNGraph})
    g, state, forcing, forcing_next, static, target = Flux.ignore_derivatives() do
        g = batch[1]
        g, g.ndata.state, g.ndata.forcing, batch[2].ndata.forcing, g.ndata.static, batch[2].ndata.state
    end
    pred = model(g, state, forcing, static, forcing_next)
    Flux.mse(pred[1:1, :], target[1:1, :]) +
        Flux.mse(pred[2:2, :], target[2:2, :])
end

"""
    loss_components(model, batch) -> (q_mse, h_mse)

Compute unweighted 1-step-ahead MSE separately for q (row 1) and h (row 2) of
the state. Returns `(nothing, nothing)` when the model has no mass balance.

Runs entirely under `Flux.ignore_derivatives` — for diagnostic reporting only.
"""
function loss_components(model::WflowGNN, batch::Vector{<:GNNGraph})
    isnothing(model.mass_balance) && return (nothing, nothing)
    Flux.ignore_derivatives() do
        g      = batch[1]
        pred   = model(g, g.ndata.state, g.ndata.forcing, g.ndata.static, batch[2].ndata.forcing)
        target = batch[2].ndata.state
        Flux.mse(pred[1:1, :], target[1:1, :]),
        Flux.mse(pred[2:2, :], target[2:2, :])
    end
end

