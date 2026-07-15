import TOML
import Dates
using Flux
using GraphNeuralNetworks
using MLUtils
using ParameterSchedulers
using ProgressMeter

"""
    TrainSettings

Configuration for a training run.

Fields:
- `epochs`         : total number of training epochs.
- `batch_size`     : number of windows per mini-batch.
- `lr_start`       : initial learning rate.
- `lr_final`       : target learning rate after decay.
- `lr_steps`       : number of decay steps (epochs between each LR drop).
- `strategy`       : training curriculum (rollout steps and noise schedule).
- `device`         : compute device; `:cpu` or `:gpu`. If `:gpu` is requested but
                     CUDA is unavailable, falls back to `:cpu` with a warning.
- `val_daterange`  : optional `(start::DateTime, stop::DateTime)` pair. When set,
                     an additional autoregressive rollout is run over the validation
                     data for the timesteps that fall within this date range and a
                     movie is saved as `validation_daterange.mp4`.
"""
struct TrainSettings
    epochs        :: Int
    batch_size    :: Int
    lr_start      :: Float32
    lr_final      :: Float32
    lr_steps      :: Int
    strategy      :: TrainingStrategy
    device        :: Symbol
    val_daterange :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}}
end

"""
    TrainSettings(; epochs, batch_size, lr_start, lr_final, lr_steps, strategy,
                    device = :cpu, val_daterange = nothing) -> TrainSettings
"""
function TrainSettings(;
        epochs        :: Int,
        batch_size    :: Int,
        lr_start      :: Real,
        lr_final      :: Real,
        lr_steps      :: Int,
        strategy      :: TrainingStrategy,
        device        :: Symbol = :cpu,
        val_daterange :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}} = nothing)

    epochs     > 0 || throw(ArgumentError("epochs must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    lr_steps   > 0 || throw(ArgumentError("lr_steps must be positive"))
    lr_start   > 0 || throw(ArgumentError("lr_start must be positive"))
    lr_final   > 0 || throw(ArgumentError("lr_final must be positive"))
    lr_final  <= lr_start || throw(ArgumentError("lr_final must be <= lr_start"))
    device in (:cpu, :gpu) || throw(ArgumentError("device must be :cpu or :gpu"))

    if device == :gpu
        try
            # Flux.gpu returns the same CPU array when CUDA is unavailable.
            if typeof(Flux.gpu(zeros(Float32, 1))) <: Array
                @warn "CUDA not available; falling back to :cpu"
                device = :cpu
            end
        catch
            @warn "Could not initialise GPU; falling back to :cpu"
            device = :cpu
        end
    end

    TrainSettings(epochs, batch_size,
                  Float32(lr_start), Float32(lr_final),
                  lr_steps, strategy, device, val_daterange)
end

function Base.show(io::IO, s::TrainSettings)
    println(io, "TrainSettings:")
    println(io, "  epochs        : ", s.epochs)
    println(io, "  batch_size    : ", s.batch_size)
    println(io, "  lr_start      : ", s.lr_start)
    println(io, "  lr_final      : ", s.lr_final)
    println(io, "  lr_steps      : ", s.lr_steps)
    println(io, "  device        : ", s.device)
    println(io, "  val_daterange : ", isnothing(s.val_daterange) ? "nothing" :
                                     string(s.val_daterange[1], " – ", s.val_daterange[2]))
    println(io, "  strategy      :")
    print(  io, "    ", s.strategy)
end

"""
    save_train_settings(path, settings)

Write `settings` to a TOML file at `path`.
"""
function save_train_settings(path::String, s::TrainSettings)
    dict = Dict(
        "epochs"     => s.epochs,
        "batch_size" => s.batch_size,
        "lr_start"   => Float64(s.lr_start),
        "lr_final"   => Float64(s.lr_final),
        "lr_steps"   => s.lr_steps,
        "device"     => String(s.device),
        "strategy"   => Dict(
            "steps"       => s.strategy.steps,
            "durations"   => s.strategy.durations,
            "noise_scale" => Float64(s.strategy.noise_scale),
        ),
    )
    if !isnothing(s.val_daterange)
        dict["val_daterange"] = [string(s.val_daterange[1]), string(s.val_daterange[2])]
    end
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_train_settings(path) -> TrainSettings

Read a `TrainSettings` from the TOML file at `path`.
"""
function load_train_settings(path::String)
    d  = TOML.parsefile(path)
    sd = d["strategy"]
    strategy = TrainingStrategy(
        convert(Vector{Int}, sd["steps"]),
        convert(Vector{Int}, sd["durations"]),
        Float32(get(sd, "noise_scale", 0.0)),
    )
    return TrainSettings(
        epochs        = d["epochs"],
        batch_size    = d["batch_size"],
        lr_start      = Float32(d["lr_start"]),
        lr_final      = Float32(d["lr_final"]),
        lr_steps      = d["lr_steps"],
        strategy      = strategy,
        device        = Symbol(get(d, "device", "cpu")),
        val_daterange = if haskey(d, "val_daterange")
            r = d["val_daterange"]
            (Dates.DateTime(r[1]), Dates.DateTime(r[2]))
        else
            nothing
        end,
    )
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

"""
    train_model!(model, train_loader, val_loader, ts)
        -> (train_rollout, val_rollout, train_1step, val_1step)

Train `model` in-place and return four `Vector{Float32}` arrays with
per-epoch losses:
1. `train_rollout` - multi-step rollout loss on the training set.
2. `val_rollout`   - multi-step rollout loss on the validation set.
3. `train_1step`   - 1-step-ahead MSE on the training set.
4. `val_1step`     - 1-step-ahead MSE on the validation set.

`model` must already reside on the target compute device before this call
(move it with `Flux.gpu` / `Flux.cpu` at the call site). The data loaders are
moved to the same device internally based on `ts.device`.

The `Step` LR schedule decays by a constant factor every `ts.lr_steps` epochs,
reaching approximately `ts.lr_final` after `ts.epochs` epochs.
"""
function train_model!(model,
                      train_loader,
                      val_loader,
                      ts::TrainSettings)

    strategy = ts.strategy

    # Move loaders to the target device. The model is already on device.
    dev_fn         = ts.device == :gpu ? Flux.gpu : identity
    train_loader_d = dev_fn(train_loader)
    val_loader_d   = dev_fn(val_loader)

    # Derive per-step decay so that after floor(epochs/lr_steps) drops
    # the LR reaches lr_final.
    ndecays  = max(1, floor(Int, ts.epochs / ts.lr_steps))
    decay    = (ts.lr_final / ts.lr_start)^(1f0 / ndecays)
    schedule = Step(ts.lr_start, decay, ts.lr_steps)

    opt_state = Flux.setup(Adam(ts.lr_start), model)

    train_rollout = Float32[]
    val_rollout   = Float32[]
    train_1step   = Float32[]
    val_1step     = Float32[]
    train_q_1step = Float32[]
    val_q_1step   = Float32[]
    train_h_1step = Float32[]
    val_h_1step   = Float32[]

    has_components = !isnothing(model.mass_balance)

    prog = Progress(ts.epochs; desc = "Training ", showspeed = true)

    for epoch in 1:ts.epochs

        update_steps!(strategy, epoch)
        Flux.adjust!(opt_state, schedule(epoch - 1))

        # Training pass
        ep_train_rollout = 0f0
        ep_train_1step   = 0f0
        ep_train_q_1step = 0f0
        ep_train_h_1step = 0f0
        n_batches        = 0

        for batch in train_loader_d
            train_loss, grads = Flux.withgradient(m -> loss_function(m, batch, strategy), model)
            Flux.update!(opt_state, model, grads[1])
            ep_train_rollout += train_loss
            ep_train_1step   += one_step_loss(model, batch, strategy.h_loss_weight)
            if has_components
                qc, hc = loss_components(model, batch)
                ep_train_q_1step += qc
                ep_train_h_1step += hc
            end
            n_batches        += 1
        end
        ep_train_rollout /= n_batches
        ep_train_1step   /= n_batches
        ep_train_q_1step /= n_batches
        ep_train_h_1step /= n_batches

        # Validation pass
        ep_val_rollout = mean(loss_function(model, b, strategy) for b in val_loader_d)
        ep_val_1step   = mean(one_step_loss(model, b, strategy.h_loss_weight) for b in val_loader_d)
        if has_components
            val_comps      = [loss_components(model, b) for b in val_loader_d]
            ep_val_q_1step = mean(c[1] for c in val_comps)
            ep_val_h_1step = mean(c[2] for c in val_comps)
        else
            ep_val_q_1step = NaN32
            ep_val_h_1step = NaN32
        end

        push!(train_rollout, ep_train_rollout)
        push!(val_rollout,   ep_val_rollout)
        push!(train_1step,   ep_train_1step)
        push!(val_1step,     ep_val_1step)
        push!(train_q_1step, has_components ? ep_train_q_1step : NaN32)
        push!(val_q_1step,   ep_val_q_1step)
        push!(train_h_1step, has_components ? ep_train_h_1step : NaN32)
        push!(val_h_1step,   ep_val_h_1step)

        base_vals = [
            (:epoch,         "$epoch / $(ts.epochs)"),
            (:steps,         strategy.current_steps),
            (:lr,            round(schedule(epoch - 1), sigdigits = 3)),
            (:train_rollout, round(ep_train_rollout, sigdigits = 4)),
            (:val_rollout,   round(ep_val_rollout,   sigdigits = 4)),
            (:train_1step,   round(ep_train_1step,   sigdigits = 4)),
            (:val_1step,     round(ep_val_1step,     sigdigits = 4)),
        ]
        comp_vals = has_components ? [
            (:train_q_1step, round(ep_train_q_1step, sigdigits = 4)),
            (:val_q_1step,   round(ep_val_q_1step,   sigdigits = 4)),
            (:train_h_1step, round(ep_train_h_1step, sigdigits = 4)),
            (:val_h_1step,   round(ep_val_h_1step,   sigdigits = 4)),
        ] : []
        next!(prog; showvalues = vcat(base_vals, comp_vals))
    end

    return (train_rollout = train_rollout,
            val_rollout   = val_rollout,
            train_1step   = train_1step,
            val_1step     = val_1step,
            train_q_1step = train_q_1step,
            val_q_1step   = val_q_1step,
            train_h_1step = train_h_1step,
            val_h_1step   = val_h_1step)
end

