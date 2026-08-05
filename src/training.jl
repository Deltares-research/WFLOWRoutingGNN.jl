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
- `grad_clip`      : global gradient-L2-norm clip applied before each optimiser
                     step (`Optimisers.ClipNorm`). `0` disables clipping. This is
                     a safety net for curriculum-phase restarts, where a newly
                     deepened BPTT graph can produce a transient gradient spike.
- `h_loss_scale`   : how the water-depth (`h`) term of the loss is normalised
                     (river domain only). `:absolute` (default) weights the
                     z-scored h-MSE by `σ_h/(dt·σ_q)` so the two loss magnitudes
                     match. `:increment` measures h-error on the mass-balance
                     increment scale (`dt·σ_q`) instead of the absolute-depth
                     scale (`σ_h`), i.e. weight `(σ_h/(dt·σ_q))²`, which makes
                     `∂h_norm/∂q_norm ≈ O(1)` and removes the stiff
                     gradient amplification of the hard mass-balance decoder.
- `phase_backoff_factor` : runtime safety net for curriculum warm restarts. At a
                     phase boundary the deepened rollout can destabilise even a
                     well-tuned LR (something an untrained offline range test
                     cannot predict). When an epoch is detected as unstable
                     (non-finite gradient, skipped batches, or a grad-norm spike
                     far above the phase's running median), the current phase's
                     LR is multiplied by this factor for the remaining epochs of
                     the phase, then reset at the next boundary. Must be in
                     `(0, 1]`; `1.0` disables the backoff.
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
    grad_clip        :: Float32
    h_loss_scale     :: Symbol
    phase_backoff_factor :: Float32
    strategy         :: TrainingStrategy
    device           :: Symbol
    val_daterange    :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}}
end

"""
    TrainSettings(; epochs, batch_size, lr_start, lr_final, lr_steps, strategy,
                    lr_warmup_epochs = 1, lr_peak_decay = 0.7, grad_clip = 1.0,
                    h_loss_scale = :absolute, phase_backoff_factor = 0.5,
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
        grad_clip        :: Real = 1.0,
        h_loss_scale     :: Symbol = :absolute,
        phase_backoff_factor :: Real = 0.5,
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
    grad_clip >= 0 || throw(ArgumentError("grad_clip must be non-negative (0 disables clipping)"))
    h_loss_scale in (:absolute, :increment) ||
        throw(ArgumentError("h_loss_scale must be :absolute or :increment"))
    0 < phase_backoff_factor <= 1 ||
        throw(ArgumentError("phase_backoff_factor must be in (0, 1] (1 disables backoff)"))
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
                  lr_steps, lr_warmup_epochs, Float32(lr_peak_decay), Float32(grad_clip),
                  h_loss_scale, Float32(phase_backoff_factor), strategy, device, val_daterange)
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
    println(io, "  grad_clip        : ", s.grad_clip)
    println(io, "  h_loss_scale     : ", s.h_loss_scale)
    println(io, "  phase_backoff_factor : ", s.phase_backoff_factor)
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
        "grad_clip"        => Float64(s.grad_clip),
        "h_loss_scale"     => String(s.h_loss_scale),
        "phase_backoff_factor" => Float64(s.phase_backoff_factor),
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
        grad_clip        = Float32(get(d, "grad_clip", 1.0)),
        h_loss_scale     = Symbol(get(d, "h_loss_scale", "absolute")),
        phase_backoff_factor = Float32(get(d, "phase_backoff_factor", 0.5)),
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

    # Optimiser: optionally clip the global gradient L2 norm before each Adam
    # step. `grad_clip <= 0` disables clipping. `throw = false` leaves a
    # non-finite gradient untouched (those steps are filtered out in the loop).
    rule = ts.grad_clip > 0 ?
        Flux.Optimisers.OptimiserChain(
            Flux.Optimisers.ClipNorm(Float32(ts.grad_clip); throw = false),
            Adam(ts.lr_start)) :
        Adam(ts.lr_start)
    opt_state = Flux.setup(rule, model)

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

    # Adaptive per-phase LR backoff state. `lr_scale` multiplies the scheduled
    # curriculum LR; it resets to 1 at each curriculum-phase boundary (fresh warm
    # restart) and is shrunk by `ts.phase_backoff_factor` whenever an epoch is
    # flagged unstable. `phase_gnorms` holds this phase's finite epoch grad norms
    # so a spike can be judged relative to the phase's own running median.
    lr_scale     = 1.0
    prev_steps   = strategy.current_steps
    phase_gnorms = Float64[]
    backoff_on   = ts.phase_backoff_factor < 1
    const_spike_factor = 10.0   # grad norm this far above the phase median = spike
    const_scale_floor  = 1f-3   # never shrink the phase LR below this fraction

    prog = Progress(ts.epochs; desc = "Training ", showspeed = true)

    for epoch in 1:ts.epochs

        update_steps!(strategy, epoch)
        # New curriculum phase: reset the warm-restart backoff state.
        if strategy.current_steps != prev_steps
            prev_steps = strategy.current_steps
            lr_scale   = 1.0
            empty!(phase_gnorms)
        end
        lr = curriculum_lr(ts, epoch) * lr_scale
        Flux.adjust!(opt_state, lr)

        # Training pass
        ep_train_rollout = 0f0
        ep_train_1step   = 0f0
        ep_train_q_1step = 0f0
        ep_train_h_1step = 0f0
        ep_grad_norm     = 0.0
        n_batches        = 0
        n_skipped        = 0

        for batch in train_loader_d
            train_loss, grads = Flux.withgradient(m -> loss_function(m, batch, strategy, static_d), model)
            gn = _grad_l2norm(grads[1])
            # Skip updates from a non-finite loss/gradient (e.g. a curriculum-phase
            # restart spike) so a single bad step cannot poison the weights.
            # ClipNorm above tames the merely-large-but-finite steps.
            if !isfinite(train_loss) || !isfinite(gn)
                n_skipped += 1
                continue
            end
            Flux.update!(opt_state, model, grads[1])
            ep_grad_norm     += gn
            ep_train_rollout += train_loss
            ep_train_1step   += one_step_loss(model, batch, static_d, strategy.h_loss_weight)
            if has_components
                qc, hc = loss_components(model, batch, static_d)
                ep_train_q_1step += qc
                ep_train_h_1step += hc
            end
            n_batches        += 1
        end
        if n_skipped > 0
            @warn "Epoch $epoch: skipped $n_skipped non-finite update(s) (loss/grad)."
        end
        denom = max(n_batches, 1)
        ep_train_rollout /= denom
        ep_train_1step   /= denom
        ep_train_q_1step /= denom
        ep_train_h_1step /= denom
        ep_grad_norm     /= denom

        # Adaptive backoff: if this epoch looks unstable (non-finite grad, any
        # skipped batch, or a grad-norm spike well above the phase median),
        # shrink the LR for the rest of the phase. This reacts to a destabilised
        # warm restart that an offline range test cannot foresee, and composes
        # with the ClipNorm + non-finite skip above.
        if backoff_on
            spike = !isempty(phase_gnorms) &&
                    ep_grad_norm > const_spike_factor * median(phase_gnorms)
            if !isfinite(ep_grad_norm) || n_skipped > 0 || spike
                new_scale = max(lr_scale * ts.phase_backoff_factor, const_scale_floor)
                if new_scale < lr_scale
                    @warn @sprintf("Epoch %d (steps=%d): unstable epoch; backing off phase LR ×%.3g (scale → %.3g).",
                                   epoch, strategy.current_steps, ts.phase_backoff_factor, new_scale)
                    lr_scale = new_scale
                end
            end
        end
        isfinite(ep_grad_norm) && push!(phase_gnorms, ep_grad_norm)

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

