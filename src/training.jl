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
- `eval_horizon`   : number of steps for the **fixed-horizon** validation metric
                     evaluated each epoch (see [`build_fixed_horizon_eval`](@ref)).
                     Unlike `val_rollout` (measured at the growing curriculum
                     horizon, hence not comparable across phases), this is a
                     constant-length autoregressive rollout, so it is directly
                     comparable epoch-to-epoch and drives early stopping. `0`
                     disables the fixed-horizon eval entirely.
- `eval_anchors`   : number of evenly-spaced start points in the validation split
                     used as anchors for the fixed-horizon rollout (batched into a
                     single ensemble forward pass per step).
- `upstream_points`: number of percentile-ladder nodes to plot for validation
                     upstream/downstream timeseries diagnostics. `1` reproduces
                     the previous outlet-only behaviour.
- `early_stopping` : when `true`, stop training once the fixed-horizon validation
                     discharge RMSE has not improved for `early_stopping_patience`
                     epochs, and restore the best-metric weights. Requires
                     `eval_horizon > 0`.
- `early_stopping_patience` : number of epochs without fixed-horizon RMSE
                     improvement tolerated before early stopping triggers.
- `checkpoint_every` : save a model checkpoint every this many epochs (into
                     `<run_dir>/checkpoints/epoch_NNNN/`). `0` disables periodic
                     checkpointing.
- `checkpoint_full_eval` : when `true`, run the same full end-of-training
                     evaluation (trajectory rollout, plots, NetCDF, movie) for
                     each periodic checkpoint, written alongside its weights.
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
    eval_horizon     :: Int
    eval_anchors     :: Int
    upstream_points  :: Int
    early_stopping   :: Bool
    early_stopping_patience :: Int
    checkpoint_every :: Int
    checkpoint_full_eval :: Bool
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
        val_daterange    :: Union{Nothing, Tuple{Dates.DateTime, Dates.DateTime}} = nothing,
        eval_horizon     :: Int = 30,
        eval_anchors     :: Int = 32,
        upstream_points  :: Int = 5,
        early_stopping   :: Bool = false,
        early_stopping_patience :: Int = 20,
        checkpoint_every :: Int = 0,
        checkpoint_full_eval :: Bool = false)

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
    eval_horizon >= 0 || throw(ArgumentError("eval_horizon must be non-negative (0 disables the fixed-horizon eval)"))
    eval_horizon == 0 || eval_anchors > 0 ||
        throw(ArgumentError("eval_anchors must be positive when eval_horizon > 0"))
    upstream_points > 0 || throw(ArgumentError("upstream_points must be positive"))
    early_stopping_patience > 0 ||
        throw(ArgumentError("early_stopping_patience must be positive"))
    checkpoint_every >= 0 || throw(ArgumentError("checkpoint_every must be non-negative (0 disables checkpointing)"))
    (!early_stopping || eval_horizon > 0) ||
        throw(ArgumentError("early_stopping requires eval_horizon > 0"))

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
                  h_loss_scale, Float32(phase_backoff_factor), strategy, device, val_daterange,
                  eval_horizon, eval_anchors, upstream_points,
                  early_stopping, early_stopping_patience,
                  checkpoint_every, checkpoint_full_eval)
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
    println(io, "  eval_horizon     : ", s.eval_horizon)
    println(io, "  eval_anchors     : ", s.eval_anchors)
    println(io, "  upstream_points  : ", s.upstream_points)
    println(io, "  early_stopping   : ", s.early_stopping)
    println(io, "  early_stopping_patience : ", s.early_stopping_patience)
    println(io, "  checkpoint_every : ", s.checkpoint_every)
    println(io, "  checkpoint_full_eval : ", s.checkpoint_full_eval)
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
        "eval_horizon"     => s.eval_horizon,
        "eval_anchors"     => s.eval_anchors,
        "upstream_points"  => s.upstream_points,
        "early_stopping"   => s.early_stopping,
        "early_stopping_patience" => s.early_stopping_patience,
        "checkpoint_every" => s.checkpoint_every,
        "checkpoint_full_eval" => s.checkpoint_full_eval,
        "strategy"   => Dict(
            "steps"       => s.strategy.steps,
            "durations"   => s.strategy.durations,
            "noise_scale" => Float64(s.strategy.noise_scale),
            "loss_type"   => String(s.strategy.loss_type),
            "peak_delta"  => Float64(s.strategy.peak_delta),
            "peak_lambda" => Float64(s.strategy.peak_lambda),
            "peak_gamma"  => Float64(s.strategy.peak_gamma),
            "peak_w_max"  => Float64(s.strategy.peak_w_max),
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
        Float32(get(sd, "noise_scale", 0.0));
        loss_type = Symbol(get(sd, "loss_type", "mse")),
        peak_delta = Float32(get(sd, "peak_delta", 1.0)),
        peak_lambda = Float32(get(sd, "peak_lambda", 0.0)),
        peak_gamma = Float32(get(sd, "peak_gamma", 1.0)),
        peak_w_max = Float32(get(sd, "peak_w_max", 4.0)),
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
        eval_horizon     = get(d, "eval_horizon", 30),
        eval_anchors     = get(d, "eval_anchors", 32),
        upstream_points  = get(d, "upstream_points", 5),
        early_stopping   = get(d, "early_stopping", false),
        early_stopping_patience = get(d, "early_stopping_patience", 20),
        checkpoint_every = get(d, "checkpoint_every", 0),
        checkpoint_full_eval = get(d, "checkpoint_full_eval", false),
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
# Tier-1 peak-loss diagnostics (docs/notes/peak_accuracy_todo.md §2b Tier 1)
# ---------------------------------------------------------------------------

"""
    peak_epoch_diagnostics(model, batch, strategy, static; peak_stats = nothing)
        -> NamedTuple

Tier-1 loss-tuning diagnostics for the `:huber` peak-weighted loss, computed
on a single teacher-forced one-step prediction (`batch[1] → batch[2]`, as in
[`loss_components`](@ref)):

- `c_peak`               : fraction of the total (q + `h_loss_weight`·h)
                           weighted Huber loss mass coming from cells above
                           the peak threshold `u_i`.
- `rmse_high`/`mae_high` : RMSE/MAE restricted to cells above `u_i` (pooled
                           over q and h).
- `w_mean`/`w_max`/`w_min` : peak-weight (`w_i,t`) summary stats, pooled over
                           q and h (see [`peak_loss_summary`](@ref)).
- `q_grad_norm`/`h_grad_norm` : parameter-gradient L2 norms of the q-only and
                           h-only loss terms — for `h_loss_weight`
                           gradient-norm balancing (notes §7).
- `peak_grad_frac`       : fraction of the total parameter-gradient L2 norm
                           attributable to the peak-masked subset of the loss
                           (notes §4.9), `NaN32` when no peak cell exists.

Runs 3 extra forward/backward passes (total, q-only, h-only), plus a 4th when
any peak cell exists — intended to be called **once per epoch** on a single
fixed batch, not every training step. Returns all-`NaN32` fields when
`strategy.loss_type != :huber`.
"""
function peak_epoch_diagnostics(model::WflowGNN, batch::Vector{<:GNNGraph},
                                strategy::TrainingStrategy, static::AbstractMatrix;
                                peak_stats = nothing)
    nan_result = (; c_peak = NaN32, rmse_high = NaN32, mae_high = NaN32,
                    w_mean = NaN32, w_max = NaN32, w_min = NaN32,
                    q_grad_norm = NaN32, h_grad_norm = NaN32, peak_grad_frac = NaN32)
    (strategy.loss_type == :huber && length(batch) >= 2) || return nan_result

    g, state, forcing, forcing_next, target = Flux.ignore_derivatives() do
        gg = batch[1]
        gg, gg.ndata.state, gg.ndata.forcing, batch[2].ndata.forcing, batch[2].ndata.state
    end
    q_target = target[1:1, :]
    h_target = target[2:2, :]

    q_u, q_s, h_u, h_s = Flux.ignore_derivatives() do
        _resolve_peak_thresholds(peak_stats, q_target, h_target)
    end

    delta  = strategy.peak_delta
    lambda = strategy.peak_lambda
    gamma  = strategy.peak_gamma
    wmax   = strategy.peak_w_max

    # --- Forward-only: weights/mask, C_peak, RMSE_high/MAE_high, weight stats.
    c_peak, rmse_high, mae_high, w_mean, w_max_v, w_min_v, q_mask, h_mask, q_w, h_w =
        Flux.ignore_derivatives() do
            pred0   = model(g, state, forcing, static, forcing_next)
            q_pred0 = pred0[1:1, :]
            h_pred0 = pred0[2:2, :]

            qw, qmask = peak_weight_matrix(q_target, q_u, q_s; lambda, gamma, w_max = wmax)
            hw, hmask = peak_weight_matrix(h_target, h_u, h_s; lambda, gamma, w_max = wmax)

            q_elem = _huber_element.(q_pred0 .- q_target, delta) .* qw
            h_elem = _huber_element.(h_pred0 .- h_target, delta) .* hw

            total_mass = sum(q_elem) + strategy.h_loss_weight * sum(h_elem)
            peak_mass  = sum(q_elem[qmask]) + strategy.h_loss_weight * sum(h_elem[hmask])
            cpk        = total_mass > 0f0 ? Float32(peak_mass / total_mass) : NaN32

            sq_sum = 0f0
            ab_sum = 0f0
            n_high = 0
            if any(qmask)
                q_diff = q_pred0[qmask] .- q_target[qmask]
                sq_sum += sum(abs2, q_diff)
                ab_sum += sum(abs, q_diff)
                n_high += Int(sum(qmask))
            end
            if any(hmask)
                h_diff = h_pred0[hmask] .- h_target[hmask]
                sq_sum += sum(abs2, h_diff)
                ab_sum += sum(abs, h_diff)
                n_high += Int(sum(hmask))
            end
            rmse_h = n_high == 0 ? NaN32 : Float32(sqrt(sq_sum / n_high))
            mae_h  = n_high == 0 ? NaN32 : Float32(ab_sum / n_high)

            wm  = Float32((mean(qw) + mean(hw)) / 2)
            wmx = Float32(max(maximum(qw), maximum(hw)))
            wmn = Float32(min(minimum(qw), minimum(hw)))

            (cpk, rmse_h, mae_h, wm, wmx, wmn, qmask, hmask, qw, hw)
        end

    # --- Backward passes: total, q-only, h-only, peak-masked gradient norms.
    _, total_grads = Flux.withgradient(model) do m
        pred   = m(g, state, forcing, static, forcing_next)
        q_loss = peak_weighted_huber_loss(pred[1:1, :], q_target, q_u, q_s; delta, lambda, gamma, w_max = wmax)
        h_loss = peak_weighted_huber_loss(pred[2:2, :], h_target, h_u, h_s; delta, lambda, gamma, w_max = wmax)
        q_loss + strategy.h_loss_weight * h_loss
    end
    total_norm = _grad_l2norm(total_grads[1])

    _, q_grads = Flux.withgradient(model) do m
        pred = m(g, state, forcing, static, forcing_next)
        peak_weighted_huber_loss(pred[1:1, :], q_target, q_u, q_s; delta, lambda, gamma, w_max = wmax)
    end
    q_grad_norm = Float32(_grad_l2norm(q_grads[1]))

    _, h_grads = Flux.withgradient(model) do m
        pred = m(g, state, forcing, static, forcing_next)
        peak_weighted_huber_loss(pred[2:2, :], h_target, h_u, h_s; delta, lambda, gamma, w_max = wmax)
    end
    h_grad_norm = Float32(_grad_l2norm(h_grads[1]))

    peak_grad_frac = NaN32
    if any(q_mask) || any(h_mask)
        _, peak_grads = Flux.withgradient(model) do m
            pred   = m(g, state, forcing, static, forcing_next)
            q_pred = pred[1:1, :]
            h_pred = pred[2:2, :]
            q_res  = _huber_element.(q_pred .- q_target, delta) .* q_w
            h_res  = _huber_element.(h_pred .- h_target, delta) .* h_w
            qm = any(q_mask) ? sum(q_res[q_mask]) : 0f0
            hm = any(h_mask) ? sum(h_res[h_mask]) : 0f0
            qm + strategy.h_loss_weight * hm
        end
        peak_norm = Float32(_grad_l2norm(peak_grads[1]))
        peak_grad_frac = total_norm > 0f0 ? Float32(peak_norm / total_norm) : NaN32
    end

    return (; c_peak, rmse_high, mae_high, w_mean, w_max = w_max_v, w_min = w_min_v,
            q_grad_norm, h_grad_norm, peak_grad_frac)
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

"""
    train_model!(model, train_loader, val_loader, ts, static;
                 fixed_eval = nothing, checkpoint_callback = nothing)
        -> NamedTuple of per-epoch Vector{Float32}

Train `model` in-place and return a `NamedTuple` of per-epoch history arrays:
1. `train_rollout` - multi-step rollout loss on the training set.
2. `val_rollout`   - multi-step rollout loss on the validation set.
3. `train_1step`   - 1-step-ahead MSE on the training set.
4. `val_1step`     - 1-step-ahead MSE on the validation set.

For mass-balance models it also returns the per-component 1-step MSE
(`train_q_1step`, `val_q_1step`, `train_h_1step`, `val_h_1step`), the q→h error
amplification diagnostic (`train_amp`, `val_amp`; see [`mb_amplification`](@ref))
and its analytic reference gain (`mb_gain`), plus `grad_norm`, the per-epoch
applied learning rate (`lr`) and curriculum rollout length (`steps`).
Non-mass-balance models fill the mass-balance columns with `NaN32`.

When a `fixed_eval::FixedHorizonEval` is supplied (see
[`build_fixed_horizon_eval`](@ref)) the returned NamedTuple additionally holds
`val_fixed_rmse` and `val_peak_ratio` — the constant-length rollout discharge
RMSE and peak-amplification ratio, computed each epoch (filled with `NaN32` when
no `fixed_eval` is given). To guard against single-anchor max masking, it also
stores two per-epoch aggregates from the per-anchor arrays:
`val_peak_ratio_frac_gt2` (fraction of anchors with `peak_ratio > 2`) and
`val_fixed_rmse_highflow` (mean anchor RMSE over anchors with
`start_q_percentile ≥ 0.8`; `NaN32` if none).
It also reports `stopped_epoch` (the last completed
epoch, `< ts.epochs` when early stopping triggered) and `best_epoch` (the
fixed-horizon RMSE minimiser), plus the training-stability counters
`n_nonfinite_skips` (total non-finite update skips), `n_backoffs` (adaptive LR
backoff events) and `stopped_early` (whether early stopping fired). When
`ts.early_stopping` is set, training halts
once the fixed-horizon RMSE has not improved for `ts.early_stopping_patience`
epochs and the best-metric weights are restored into `model`.

When `strategy.loss_type == :huber`, the returned NamedTuple also holds the
Tier-1 loss-tuning diagnostics from [`peak_epoch_diagnostics`](@ref), computed
once per epoch on the last training batch: `peak_c_peak`, `peak_rmse_high`,
`peak_mae_high`, `peak_w_mean`, `peak_w_max`, `peak_w_min`, `peak_q_grad_norm`,
`peak_h_grad_norm`, `peak_grad_frac` (all `NaN32` otherwise).

`checkpoint_callback`, when supplied, is invoked as `checkpoint_callback(model,
epoch)` every `ts.checkpoint_every` epochs (the caller handles the I/O and any
per-checkpoint evaluation).

`peak_stats`, used only when `ts.strategy.loss_type == :huber`, is the
per-node peak threshold/scale `NamedTuple` returned by
[`peak_node_stats`](@ref) (computed on the training split); it is forwarded to
[`loss_function`](@ref) for both the train and validation loss, and moved to
the training device once up-front. When `nothing` (the default),
`loss_function`'s coarse per-batch fallback threshold is used instead.

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
                      static_cpu::AbstractMatrix{Float32};
                      fixed_eval = nothing,
                      checkpoint_callback = nothing,
                      peak_stats = nothing)

    strategy = ts.strategy

    # Move static features to the target device up-front. Each batch is copied
    # to the device by a `DeviceIterator` (Flux/MLDataDevices) wrapped around the
    # CPU loaders below.
    #
    # We build the `DeviceIterator` MANUALLY rather than calling `Flux.gpu(loader)`.
    # `gpu(loader)` routes through `adapt_structure`, which for a `parallel = true`
    # loader performs the host→device copy INSIDE the worker threads
    # (`eachobsparallel`: `put!(ch, dev(obs))`). Issuing CUDA ops off the owning
    # thread corrupts the stream and surfaces later as a deferred
    # CUDA_ERROR_INVALID_VALUE at the first device→host sync (e.g. the `nrm2`
    # readback inside `ClipNorm`). Constructing `DeviceIterator(dev_fn, loader)`
    # directly keeps the parallel CPU batch prep on worker threads while doing the
    # transfer AND the eager `unsafe_free!` of the previous batch on the main
    # thread — stream-safe and memory-efficient (CuIterator semantics).
    dev_fn         = ts.device == :gpu ? Flux.gpu : identity
    wrap_loader    = ts.device == :gpu ? (ld -> Flux.DeviceIterator(dev_fn, ld)) : identity
    train_loader_d = wrap_loader(train_loader)
    val_loader_d   = wrap_loader(val_loader)
    static_d       = dev_fn(static_cpu)
    peak_stats_d   = peak_stats === nothing ? nothing :
        (q = (u = dev_fn(peak_stats.q.u), s = dev_fn(peak_stats.q.s)),
         h = (u = dev_fn(peak_stats.h.u), s = dev_fn(peak_stats.h.s)))

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
    train_amp     = Float32[]   # q→h error amplification through the mass balance
    val_amp       = Float32[]
    mb_gain       = Float32[]   # analytic self-gain θ·dt·σ_q/σ_h (reference)
    grad_norm     = Float32[]
    lr_hist       = Float32[]   # scheduled LR actually applied each epoch
    steps_hist    = Int[]       # curriculum rollout length (current_steps) each epoch
    n_skip_total  = 0           # cumulative non-finite update skips across all epochs
    n_backoffs    = 0           # cumulative adaptive LR backoff events
    val_fixed_rmse  = Float32[] # fixed-horizon discharge RMSE (physical units)
    val_peak_ratio  = Float32[] # fixed-horizon max|q_pred|/max|q_truth|
    val_peak_ratio_frac_gt2 = Float32[] # fraction anchors with peak_ratio > 2
    val_fixed_rmse_highflow = Float32[] # mean anchor RMSE for start_q_percentile >= 0.8

    # Reduction-masking guard constants for fixed-horizon per-anchor diagnostics.
    const_peak_ratio_threshold = 2.0f0
    const_highflow_percentile_threshold = 0.8f0

    # Tier-1 peak-loss diagnostics (huber loss_type only; NaN32 otherwise) —
    # see peak_epoch_diagnostics.
    peak_c_peak        = Float32[]
    peak_rmse_high      = Float32[]
    peak_mae_high       = Float32[]
    peak_w_mean         = Float32[]
    peak_w_max          = Float32[]
    peak_w_min          = Float32[]
    peak_q_grad_norm    = Float32[]
    peak_h_grad_norm    = Float32[]
    peak_grad_frac      = Float32[]
    do_peak_diagnostics = strategy.loss_type == :huber

    has_components = !isnothing(model.mass_balance)

    # Fixed-horizon validation metric + early stopping / best-weight tracking.
    do_fixed_eval = !isnothing(fixed_eval)
    if ts.early_stopping && !do_fixed_eval
        @warn "early_stopping requested but no fixed-horizon eval set was supplied; early stopping disabled."
    end
    early_stop_on = ts.early_stopping && do_fixed_eval
    best_metric   = Inf32
    best_epoch    = 0
    best_state    = nothing
    since_improve = 0
    stopped_epoch = ts.epochs

    # Adaptive per-phase LR backoff state. `lr_scale` multiplies the scheduled
    # curriculum LR; it resets to 1 at each curriculum-phase boundary (fresh warm
    # restart) and is shrunk by `ts.phase_backoff_factor` whenever an epoch is
    # flagged unstable. `phase_gnorms` holds this phase's finite epoch grad norms
    # so a spike can be judged relative to the phase's own running median.
    # Keep `lr_scale` Float32: `Flux.adjust!` rebuilds each optimiser rule with
    # the eta's type, so a Float64 eta would try to store an `Adam{Float64}` leaf
    # into the `Adam{Float32}` optimiser tree and fail to convert.
    lr_scale     = 1.0f0
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
            lr_scale   = 1.0f0
            empty!(phase_gnorms)
        end
        lr = curriculum_lr(ts, epoch) * lr_scale
        Flux.adjust!(opt_state, Float32(lr))

        # Training pass
        ep_train_rollout = 0f0
        ep_train_1step   = 0f0
        ep_train_q_1step = 0f0
        ep_train_h_1step = 0f0
        ep_train_amp     = 0f0
        ep_mb_gain       = NaN32
        ep_grad_norm     = 0.0
        n_batches        = 0
        n_skipped        = 0
        last_train_batch = nothing

        for batch in train_loader_d
            train_loss, grads = Flux.withgradient(m -> loss_function(m, batch, strategy, static_d; peak_stats = peak_stats_d), model)
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
                amp, gain = mb_amplification(model, batch, static_d)
                ep_train_amp += amp
                ep_mb_gain    = gain
            end
            last_train_batch  = batch
            n_batches        += 1
        end
        if n_skipped > 0
            @warn "Epoch $epoch: skipped $n_skipped non-finite update(s) (loss/grad)."
        end
        n_skip_total += n_skipped
        denom = max(n_batches, 1)
        ep_train_rollout /= denom
        ep_train_1step   /= denom
        ep_train_q_1step /= denom
        ep_train_h_1step /= denom
        ep_train_amp     /= denom
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
                    n_backoffs += 1
                end
            end
        end
        isfinite(ep_grad_norm) && push!(phase_gnorms, ep_grad_norm)

        # Validation pass. The `DeviceIterator` copies each batch to the device
        # (and frees the previous one) on the main thread, single pass.
        ep_val_rollout = 0f0
        ep_val_1step   = 0f0
        val_q_sum      = 0f0
        val_h_sum      = 0f0
        val_amp_sum    = 0f0
        n_val          = 0
        for b in val_loader_d
            ep_val_rollout += loss_function(model, b, strategy, static_d; peak_stats = peak_stats_d)
            ep_val_1step   += one_step_loss(model, b, static_d, strategy.h_loss_weight)
            if has_components
                qc, hc      = loss_components(model, b, static_d)
                val_q_sum  += qc
                val_h_sum  += hc
                val_amp_sum += mb_amplification(model, b, static_d)[1]
            end
            n_val += 1
        end
        val_denom      = max(n_val, 1)
        ep_val_rollout /= val_denom
        ep_val_1step   /= val_denom
        if has_components
            ep_val_q_1step = val_q_sum   / val_denom
            ep_val_h_1step = val_h_sum   / val_denom
            ep_val_amp     = val_amp_sum / val_denom
        else
            ep_val_q_1step = NaN32
            ep_val_h_1step = NaN32
            ep_val_amp     = NaN32
        end

        push!(train_rollout, ep_train_rollout)
        push!(val_rollout,   ep_val_rollout)
        push!(train_1step,   ep_train_1step)
        push!(val_1step,     ep_val_1step)
        push!(train_q_1step, has_components ? ep_train_q_1step : NaN32)
        push!(val_q_1step,   ep_val_q_1step)
        push!(train_h_1step, has_components ? ep_train_h_1step : NaN32)
        push!(val_h_1step,   ep_val_h_1step)
        push!(train_amp,     has_components ? ep_train_amp : NaN32)
        push!(val_amp,       ep_val_amp)
        push!(mb_gain,       has_components ? ep_mb_gain : NaN32)
        push!(grad_norm,     Float32(ep_grad_norm))
        push!(lr_hist,       Float32(lr))
        push!(steps_hist,    strategy.current_steps)

        # Fixed-horizon validation metric (constant-length rollout from anchors),
        # comparable epoch-to-epoch and used for early stopping / best selection.
        if do_fixed_eval
            fh_metrics = fixed_horizon_metrics(model, fixed_eval; device = ts.device)
            ep_fixed_rmse = fh_metrics.rmse_q
            ep_peak_ratio = fh_metrics.peak_ratio
            ep_peak_ratio_frac = Float32(mean(fh_metrics.peak_ratio_anchor .> const_peak_ratio_threshold))
            highflow_mask = fixed_eval.start_q_percentile .>= const_highflow_percentile_threshold
            ep_fixed_rmse_highflow = any(highflow_mask) ?
                Float32(mean(fh_metrics.rmse_q_anchor[highflow_mask])) : NaN32
        else
            ep_fixed_rmse, ep_peak_ratio = NaN32, NaN32
            ep_peak_ratio_frac = NaN32
            ep_fixed_rmse_highflow = NaN32
        end
        push!(val_fixed_rmse, ep_fixed_rmse)
        push!(val_peak_ratio, ep_peak_ratio)
        push!(val_peak_ratio_frac_gt2, ep_peak_ratio_frac)
        push!(val_fixed_rmse_highflow, ep_fixed_rmse_highflow)

        # Tier-1 peak-loss diagnostics (huber loss_type only), computed once per
        # epoch on the last training batch (a fresh forward/backward, not part of
        # the optimisation step above).
        if do_peak_diagnostics && !isnothing(last_train_batch)
            pdiag = peak_epoch_diagnostics(model, last_train_batch, strategy, static_d;
                                           peak_stats = peak_stats_d)
        else
            pdiag = (c_peak = NaN32, rmse_high = NaN32, mae_high = NaN32,
                     w_mean = NaN32, w_max = NaN32, w_min = NaN32,
                     q_grad_norm = NaN32, h_grad_norm = NaN32, peak_grad_frac = NaN32)
        end
        push!(peak_c_peak,     pdiag.c_peak)
        push!(peak_rmse_high,  pdiag.rmse_high)
        push!(peak_mae_high,   pdiag.mae_high)
        push!(peak_w_mean,     pdiag.w_mean)
        push!(peak_w_max,      pdiag.w_max)
        push!(peak_w_min,      pdiag.w_min)
        push!(peak_q_grad_norm, pdiag.q_grad_norm)
        push!(peak_h_grad_norm, pdiag.h_grad_norm)
        push!(peak_grad_frac,   pdiag.peak_grad_frac)

        # Track the best fixed-horizon RMSE and (when early stopping) keep a copy
        # of the best weights so the final model is the best epoch, not the last.
        if do_fixed_eval && isfinite(ep_fixed_rmse) && ep_fixed_rmse < best_metric
            best_metric   = ep_fixed_rmse
            best_epoch    = epoch
            since_improve = 0
            early_stop_on && (best_state = deepcopy(Flux.state(model)))
        elseif do_fixed_eval
            since_improve += 1
        end

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
            (Symbol("q→h_amp"), round(ep_val_amp,    sigdigits = 3)),
        ] : []
        fixed_vals = do_fixed_eval ? [
            (:val_fixed_rmse, round(ep_fixed_rmse, sigdigits = 4)),
            (:val_peak_ratio, round(ep_peak_ratio, sigdigits = 3)),
            (:peak_ratio_frac_gt2, round(ep_peak_ratio_frac, sigdigits = 3)),
            (:val_rmse_highflow, round(ep_fixed_rmse_highflow, sigdigits = 4)),
        ] : []
        peak_vals = do_peak_diagnostics ? [
            (:C_peak,     round(pdiag.c_peak, sigdigits = 3)),
            (:RMSE_high,  round(pdiag.rmse_high, sigdigits = 4)),
            (:peak_grad_frac, round(pdiag.peak_grad_frac, sigdigits = 3)),
        ] : []
        next!(prog; showvalues = vcat(base_vals, comp_vals, fixed_vals, peak_vals))

        # Periodic checkpoint hook (I/O + optional full eval handled by caller).
        if !isnothing(checkpoint_callback) && ts.checkpoint_every > 0 &&
                epoch % ts.checkpoint_every == 0
            checkpoint_callback(model, epoch)
        end

        # Early stopping on the fixed-horizon RMSE.
        if early_stop_on && since_improve >= ts.early_stopping_patience
            stopped_epoch = epoch
            @info @sprintf("Early stopping at epoch %d: no fixed-horizon RMSE improvement for %d epochs (best %.5g at epoch %d).",
                           epoch, ts.early_stopping_patience, best_metric, best_epoch)
            break
        end
    end

    # Restore the best-metric weights so the returned/saved model is the best epoch.
    if early_stop_on && !isnothing(best_state)
        @info @sprintf("Restoring best fixed-horizon weights from epoch %d (RMSE %.5g).",
                       best_epoch, best_metric)
        Flux.loadmodel!(model, best_state)
    end

    return (train_rollout = train_rollout,
            val_rollout   = val_rollout,
            train_1step   = train_1step,
            val_1step     = val_1step,
            train_q_1step = train_q_1step,
            val_q_1step   = val_q_1step,
            train_h_1step = train_h_1step,
            val_h_1step   = val_h_1step,
            train_amp     = train_amp,
            val_amp       = val_amp,
            mb_gain       = mb_gain,
            grad_norm     = grad_norm,
            lr            = lr_hist,
            steps         = steps_hist,
            val_fixed_rmse = val_fixed_rmse,
            val_peak_ratio = val_peak_ratio,
            val_peak_ratio_frac_gt2 = val_peak_ratio_frac_gt2,
            val_fixed_rmse_highflow = val_fixed_rmse_highflow,
            peak_c_peak       = peak_c_peak,
            peak_rmse_high    = peak_rmse_high,
            peak_mae_high     = peak_mae_high,
            peak_w_mean       = peak_w_mean,
            peak_w_max        = peak_w_max,
            peak_w_min        = peak_w_min,
            peak_q_grad_norm  = peak_q_grad_norm,
            peak_h_grad_norm  = peak_h_grad_norm,
            peak_grad_frac    = peak_grad_frac,
            n_nonfinite_skips = n_skip_total,
            n_backoffs     = n_backoffs,
            stopped_early  = stopped_epoch < ts.epochs,
            stopped_epoch  = stopped_epoch,
            best_epoch     = best_epoch)
end

