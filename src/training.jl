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
- `lr_start`       : peak learning rate of the first curriculum phase.
- `lr_final`       : floor learning rate that each phase decays toward.
- `lr_steps`       : retained for backward compatibility (unused by the
                     curriculum-aligned schedule; see `train_model!`).
- `lr_warmup_epochs` : number of epochs of linear warmup at the start of each
                     curriculum phase (warm restart). Ramps from `lr_final`
                     up to the phase peak.
- `lr_peak_decay`  : geometric factor by which the per-phase peak learning rate
                     shrinks at each successive curriculum phase
                     (`peak_p = lr_start * lr_peak_decay^(p-1)`, floored at
                     `lr_final`). Must be in `(0, 1]`.
- `strategy`       : training curriculum (rollout steps and noise schedule).
- `device`         : compute device; `:cpu` or `:gpu`. If `:gpu` is requested but
                     CUDA is unavailable, falls back to `:cpu` with a warning.
- `val_daterange`  : optional `(start::DateTime, stop::DateTime)` pair. When set,
                     an additional autoregressive rollout is run over the validation
                     data for the timesteps that fall within this date range and a
                     movie is saved as `validation_daterange.mp4`.
"""
struct TrainSettings
    epochs           :: Int
    batch_size       :: Int
    lr_start         :: Float32
    lr_final         :: Float32
    lr_steps         :: Int
    lr_warmup_epochs :: Int
    lr_peak_decay    :: Float32
    strategy         :: TrainingStrategy
    device           :: Symbol
    val_daterange    :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}}
end

"""
    TrainSettings(; epochs, batch_size, lr_start, lr_final, lr_steps, strategy,
                    lr_warmup_epochs = 1, lr_peak_decay = 0.7,
                    device = :cpu, val_daterange = nothing) -> TrainSettings
"""
function TrainSettings(;
        epochs           :: Int,
        batch_size       :: Int,
        lr_start         :: Real,
        lr_final         :: Real,
        lr_steps         :: Int,
        strategy         :: TrainingStrategy,
        lr_warmup_epochs :: Int = 1,
        lr_peak_decay    :: Real = 0.7,
        device           :: Symbol = :cpu,
        val_daterange    :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}} = nothing)

    epochs     > 0 || throw(ArgumentError("epochs must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    lr_steps   > 0 || throw(ArgumentError("lr_steps must be positive"))
    lr_start   > 0 || throw(ArgumentError("lr_start must be positive"))
    lr_final   > 0 || throw(ArgumentError("lr_final must be positive"))
    lr_final  <= lr_start || throw(ArgumentError("lr_final must be <= lr_start"))
    lr_warmup_epochs >= 0 || throw(ArgumentError("lr_warmup_epochs must be non-negative"))
    0 < lr_peak_decay <= 1 || throw(ArgumentError("lr_peak_decay must be in (0, 1]"))
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
                  lr_steps, lr_warmup_epochs, Float32(lr_peak_decay),
                  strategy, device, val_daterange)
end

function Base.show(io::IO, s::TrainSettings)
    println(io, "TrainSettings:")
    println(io, "  epochs        : ", s.epochs)
    println(io, "  batch_size    : ", s.batch_size)
    println(io, "  lr_start         : ", s.lr_start)
    println(io, "  lr_final         : ", s.lr_final)
    println(io, "  lr_steps         : ", s.lr_steps)
    println(io, "  lr_warmup_epochs : ", s.lr_warmup_epochs)
    println(io, "  lr_peak_decay    : ", s.lr_peak_decay)
    println(io, "  device           : ", s.device)
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
        "lr_warmup_epochs" => s.lr_warmup_epochs,
        "lr_peak_decay"    => Float64(s.lr_peak_decay),
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
        lr_warmup_epochs = get(d, "lr_warmup_epochs", 1),
        lr_peak_decay    = Float32(get(d, "lr_peak_decay", 0.7)),
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
# Learning-rate schedule (curriculum-aligned warm restarts)
# ---------------------------------------------------------------------------

"""
    curriculum_lr(ts, epoch) -> Float32

Learning rate for a 1-based `epoch`, aligned to the rollout curriculum in
`ts.strategy`. Each curriculum phase is treated as a warm restart:

1. **Cosine decay within each phase** from the phase peak down to `ts.lr_final`,
   reset at every phase boundary.
2. **Linear warmup** of `ts.lr_warmup_epochs` epochs at the start of each phase,
   ramping from `ts.lr_final` up to the phase peak. This tames the first-few-batch
   instability of a newly-deepened BPTT graph.
3. **Shrinking peaks**: the per-phase peak decreases geometrically,
   `peak_p = lr_start * lr_peak_decay^(p-1)` (floored at `lr_final`), so later,
   harder phases take gentler steps.
4. **Phase boundaries follow `ts.strategy.durations`**, staying in lock-step with
   `update_steps!`. Epochs beyond the last scheduled phase hold at `ts.lr_final`.
"""
function curriculum_lr(ts::TrainSettings, epoch::Int)
    durations = ts.strategy.durations
    total     = sum(durations)

    # Beyond the scheduled phases the curriculum repeats the final horizon;
    # hold the LR at its fully-decayed floor.
    epoch > total && return ts.lr_final

    cum = 0
    for (p, dur) in enumerate(durations)
        if epoch <= cum + dur
            return _phase_lr(ts, p, epoch - cum, dur)
        end
        cum += dur
    end
    return ts.lr_final  # unreachable (epoch <= total guaranteed above)
end

# LR within phase `p` (1-based) at `local_epoch` (1-based) of length `phase_len`.
function _phase_lr(ts::TrainSettings, p::Int, local_epoch::Int, phase_len::Int)
    peak   = max(ts.lr_final, ts.lr_start * ts.lr_peak_decay^(p - 1))
    warmup = min(ts.lr_warmup_epochs, max(0, phase_len - 1))
    le     = local_epoch - 1  # 0-based position within the phase

    if warmup > 0 && le < warmup
        # Linear warmup: lr_final -> peak across `warmup` epochs.
        frac = (le + 1) / (warmup + 1)
        return Float32(ts.lr_final + (peak - ts.lr_final) * frac)
    end

    # Cosine decay: peak -> lr_final across the remaining epochs.
    decay_len = phase_len - warmup            # >= 1
    pos       = le - warmup                   # 0-based within decay region
    t         = decay_len <= 1 ? 1.0 : clamp(pos / (decay_len - 1), 0.0, 1.0)
    cos_f     = 0.5 * (1 + cos(pi * t))
    return Float32(ts.lr_final + (peak - ts.lr_final) * cos_f)
end

# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------

# Global L2 norm of an explicit (structural) gradient tree, as returned by
# `Flux.withgradient`. Traverses NamedTuple/Tuple/array structure and sums the
# squared entries of every numeric array leaf. `nothing` (non-differentiable
# fields) and scalar leaves contribute zero. Used purely as a training-health
# diagnostic: a rising norm at curriculum-phase restarts signals the phase peak
# LR is too high for the newly-deepened BPTT graph.
_gn_sq(x::AbstractArray{<:Number}) = sum(abs2, x)
_gn_sq(x::AbstractArray)           = isempty(x) ? 0.0 : sum(_gn_sq, x)
_gn_sq(x::NamedTuple)              = isempty(x) ? 0.0 : sum(_gn_sq, values(x))
_gn_sq(x::Tuple)                   = isempty(x) ? 0.0 : sum(_gn_sq, x)
_gn_sq(::Nothing)                  = 0.0
_gn_sq(::Number)                   = 0.0
_gn_sq(::Any)                      = 0.0

_grad_l2norm(grads) = sqrt(_gn_sq(grads))

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

The learning rate follows `curriculum_lr`: a warm-restart schedule aligned to
the rollout curriculum phases (`ts.strategy.durations`), with per-phase warmup,
cosine decay, and geometrically shrinking peaks.
"""
function train_model!(model,
                      train_loader,
                      val_loader,
                      ts::TrainSettings,
                      static_cpu::AbstractMatrix{Float32})

    strategy = ts.strategy

    # Move loaders and static features to the target device.
    # The model is already on device.
    dev_fn         = ts.device == :gpu ? Flux.gpu : identity
    train_loader_d = dev_fn(train_loader)
    val_loader_d   = dev_fn(val_loader)
    static_d       = dev_fn(static_cpu)

    opt_state = Flux.setup(Adam(ts.lr_start), model)

    train_rollout = Float32[]
    val_rollout   = Float32[]
    train_1step   = Float32[]
    val_1step     = Float32[]
    train_q_1step = Float32[]
    val_q_1step   = Float32[]
    train_h_1step = Float32[]
    val_h_1step   = Float32[]
    grad_norm     = Float32[]

    has_components = !isnothing(model.mass_balance)

    prog = Progress(ts.epochs; desc = "Training ", showspeed = true)

    for epoch in 1:ts.epochs

        update_steps!(strategy, epoch)
        lr = curriculum_lr(ts, epoch)
        Flux.adjust!(opt_state, lr)

        # Training pass
        ep_train_rollout = 0f0
        ep_train_1step   = 0f0
        ep_train_q_1step = 0f0
        ep_train_h_1step = 0f0
        ep_grad_norm     = 0.0
        n_batches        = 0

        for batch in train_loader_d
            train_loss, grads = Flux.withgradient(m -> loss_function(m, batch, strategy, static_d), model)
            Flux.update!(opt_state, model, grads[1])
            ep_grad_norm     += _grad_l2norm(grads[1])
            ep_train_rollout += train_loss
            ep_train_1step   += one_step_loss(model, batch, static_d, strategy.h_loss_weight)
            if has_components
                qc, hc = loss_components(model, batch, static_d)
                ep_train_q_1step += qc
                ep_train_h_1step += hc
            end
            n_batches        += 1
        end
        ep_train_rollout /= n_batches
        ep_train_1step   /= n_batches
        ep_train_q_1step /= n_batches
        ep_train_h_1step /= n_batches
        ep_grad_norm     /= n_batches

        # Validation pass
        ep_val_rollout = mean(loss_function(model, b, strategy, static_d) for b in val_loader_d)
        ep_val_1step   = mean(one_step_loss(model, b, static_d, strategy.h_loss_weight) for b in val_loader_d)
        if has_components
            val_comps      = [loss_components(model, b, static_d) for b in val_loader_d]
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
        push!(grad_norm,     Float32(ep_grad_norm))

        base_vals = [
            (:epoch,         "$epoch / $(ts.epochs)"),
            (:steps,         strategy.current_steps),
            (:lr,            round(lr, sigdigits = 3)),
            (:grad_norm,     round(ep_grad_norm, sigdigits = 3)),
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
            val_h_1step   = val_h_1step,
            grad_norm     = grad_norm)
end

