import Random

# ---------------------------------------------------------------------------
# Tuning toolkit: batch-size estimator, LR range test, gradient-growth probe and
# the one-shot `autotune`, promoted from `scripts/lr_range_test.jl` and
# `scripts/autotune_train.jl` (which are now thin CLIs over this file).
#
# Everything here works on the *untrained* model and a single-horizon probe
# strategy, so it is a cheap proxy for tuning the curriculum LR schedule without
# a full training run.
#
# On batch size vs LR: larger batches average out gradient noise, so the stable
# LR rises with batch size. Because each config trains at ONE fixed batch size,
# the range test is run at that batch size. The dominant knob for the warm-restart
# *peaks* is the rollout HORIZON: BPTT multiplies gradient magnitudes through the
# unrolled steps, so the stable LR falls as the horizon grows.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Setup builders (build the graph time series once; cheap per-config rebuilds)
# ---------------------------------------------------------------------------

"""
    load_tuning_data(ds, ms) -> (; graphs, norm_stats, postscale, static_arr, output_file)

Build the graph time series for `ds`/`ms` once, so that [`make_model_loader`](@ref)
can cheaply rebuild models and loaders for different `(horizon, batch_size)`.
"""
function load_tuning_data(ds::DataSettings, ms::ModelSettings)
    staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
    output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")
    schema = load_schema(ds.wflow_schema)
    graphs, norm_stats, grid, postscale, static_arr =
        build_wflow_graph(staticmaps_file, output_file, ms.domain;
                         schema,
                         stats_frac = ds.train_frac,
                         include_log_upstream_area = ms.include_log_upstream_area,
                         sigma_floor_h = ms.mb_sigma_h_floor,
                         sigma_floor_q = ms.mb_sigma_q_floor,
                         sigma_floor_inwater = ms.mb_sigma_inwater_floor,
                         normalization = ms.normalization)
    return (; graphs, norm_stats, postscale, static_arr, output_file)
end

"""
    probe_training_strategy(base, horizon) -> TrainingStrategy

Single-horizon probe strategy used by the LR autotune/range tests. It preserves the
configured loss setup so probe gradients match the eventual training run's loss
scale (especially for `:huber`).
"""
function probe_training_strategy(base::TrainingStrategy, horizon::Int)
    return TrainingStrategy([horizon], [1], base.noise_scale;
                            h_loss_weight = base.h_loss_weight,
                            loss_type = base.loss_type,
                            peak_delta = base.peak_delta,
                            peak_lambda = base.peak_lambda,
                            peak_gamma = base.peak_gamma,
                            peak_w_max = base.peak_w_max)
end

"""
    make_model_loader(gd, ds, ms, ts; horizon, batch_size, seed = nothing, shuffle = true)
        -> (; model, loader, static_d, strategy, batch_size)

Model + single-horizon training loader on `ts.device` for a given
`(horizon, batch_size)`, reusing the pre-built graph data `gd` from
[`load_tuning_data`](@ref). `batch_size` is clamped to the number of training
windows; the effective value is returned.
"""
function make_model_loader(gd, ds::DataSettings, ms::ModelSettings, ts::TrainSettings;
                           horizon::Int, batch_size::Int,
                           seed::Union{Nothing,Int} = nothing,
                           shuffle::Bool = true)
    seed !== nothing && Random.seed!(seed)
    nhorizon = horizon + 1
    dataset  = make_horizon_dataset(gd.graphs, nhorizon; at = (ds.train_frac, ds.val_frac))
    bs = min(batch_size, length(dataset.train))
    loader = MLUtils.DataLoader(dataset.train;
                                batchsize = bs, shuffle = shuffle, collate = true, parallel = true)

    # Fixed-horizon strategy: current_steps = horizon (single phase, no schedule),
    # while preserving the configured loss setup.
    strategy = probe_training_strategy(ts.strategy, horizon)

    model = build_gnn_model(ms, gd.graphs, gd.norm_stats, gd.postscale,
                            gd.output_file, bs; strategy = strategy,
                            h_loss_scale = ts.h_loss_scale)

    dev_fn = ts.device == :gpu ? Flux.gpu : identity
    return (; model    = dev_fn(model),
              loader   = dev_fn(loader),
              static_d = dev_fn(gd.static_arr),
              strategy,
              batch_size = bs)
end

# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

gpu_active() = CUDA.functional()

function gpu_mem_gib()
    gpu_active() || return (used = NaN, total = NaN)
    CUDA.synchronize()
    mi = CUDA.MemoryInfo()
    return (used = (mi.total_bytes - mi.free_bytes) / 2^30,
            total = mi.total_bytes / 2^30)
end

function free_gpu!()
    GC.gc()
    gpu_active() && CUDA.reclaim()
    return nothing
end

is_oom(e) = (isdefined(CUDA, :OutOfGPUMemoryError) && e isa CUDA.OutOfGPUMemoryError) ||
            e isa OutOfMemoryError

# ---------------------------------------------------------------------------
# LR range test
# ---------------------------------------------------------------------------

"""
    lr_range_test(setup; num_steps, lr_min, lr_max, diverge_factor = 4.0, beta = 0.98)
        -> (lrs, losses, gnorms)

Leslie-Smith LR range test: exponentially ramps the LR from `lr_min` to `lr_max`
over `num_steps` mini-batches at the fixed horizon of `setup`
(from [`make_model_loader`](@ref)), recording the bias-corrected EMA of the loss and
the gradient L2 norm. Stops early when the smoothed loss exceeds `diverge_factor`
times its best value (or becomes non-finite). The max usable LR is roughly the
point of steepest loss descent, about one order of magnitude below divergence;
see [`recommend_lr`](@ref).
"""
function lr_range_test(setup; num_steps::Int, lr_min::Float64, lr_max::Float64,
                       diverge_factor::Float64 = 4.0, beta::Float64 = 0.98)
    model    = setup.model
    strategy = setup.strategy
    static_d = setup.static_d

    opt_state = Flux.setup(Adam(Float32(lr_min)), model)
    gamma = (lr_max / lr_min)^(1 / (num_steps - 1))

    lrs    = Float64[]
    losses = Float64[]   # bias-corrected EMA of the raw loss
    gnorms = Float64[]

    lr       = lr_min
    avg_loss = 0.0
    best     = Inf
    step     = 0

    for batch in Iterators.cycle(setup.loader)
        step += 1
        step > num_steps && break

        Flux.adjust!(opt_state, Float32(lr))
        l, grads = Flux.withgradient(m -> loss_function(m, batch, strategy, static_d), model)
        Flux.update!(opt_state, model, grads[1])

        gn = _grad_l2norm(grads[1])
        avg_loss = beta * avg_loss + (1 - beta) * Float64(l)
        smooth   = avg_loss / (1 - beta^step)

        push!(lrs, lr); push!(losses, smooth); push!(gnorms, gn)

        if isnan(smooth) || isinf(smooth) || !isfinite(gn) || (step > 1 && smooth > diverge_factor * best)
            @info @sprintf("Diverged at step %d (lr=%.3e); stopping range test.", step, lr)
            break
        end
        smooth < best && (best = smooth)
        lr *= gamma
    end

    return lrs, losses, gnorms
end

# ---------------------------------------------------------------------------
# Gradient-growth probe (for setting lr_peak_decay across curriculum phases)
# ---------------------------------------------------------------------------

"""
    gradient_growth_probe(setup; num_steps = 12, warmup = 2, lr = 1e-5) -> Float64

Median gradient L2 norm over `num_steps` steps at a FIXED small LR and the fixed
rollout horizon of `setup`. Unlike the LR range test, which RAMPS the LR and, at
deep horizons on an UNTRAINED model, drives the free-running rollout into
artificial blow-up, this holds a small constant LR and merely records the gradient
norm.

The ABSOLUTE value is still inflated by the untrained rollout, so it is NOT a
usable LR. It is used only for a RELATIVE comparison across horizons: the ratio
`g(shallow)/g(deep)` sets `lr_peak_decay`, and because a deeper horizon
over-reports its gradient growth, the derived peak-shrink errs on the conservative
(smaller deep-phase peak) side. `NaN` when no finite gradient was recorded.
"""
function gradient_growth_probe(setup; num_steps::Int = 12, warmup::Int = 2,
                               lr::Float64 = 1e-5)
    model    = setup.model
    strategy = setup.strategy
    static_d = setup.static_d
    opt_state = Flux.setup(Adam(Float32(lr)), model)

    gnorms = Float64[]
    step = 0
    for batch in Iterators.cycle(setup.loader)
        step += 1
        step > warmup + num_steps && break
        _l, grads = Flux.withgradient(m -> loss_function(m, batch, strategy, static_d), model)
        gn = _grad_l2norm(grads[1])
        isfinite(gn) && Flux.update!(opt_state, model, grads[1])
        step > warmup && isfinite(gn) && push!(gnorms, gn)
    end
    return isempty(gnorms) ? NaN : median(gnorms)
end

"""
    recommend_lr(lrs, losses, gnorms = nothing; kwargs...)
        -> (; steep, min_over_10, gnorm_cap, gnorm_ref, safe, near_lr_floor, min10_near_floor)

Suggested LRs from a recorded range-test curve:

- `steep`       : classic "point of steepest descent" (minimum of `d(loss)/d(log lr)`),
                  computed robustly: the EMA-biased head and the diverging tail are
                  trimmed and the slope is smoothed over a small window.
- `min_over_10` : fast.ai fallback (min-loss LR / 10). Optimistic when the loss
                  minimum sits right under the divergence cliff.
- `gnorm_cap`   : the LR at which the (smoothed) gradient norm first blows past a
                  multiple of its early plateau together with a rising loss, backed
                  off by a safety factor. The earliest, most reliable divergence
                  signal; not masked by the EMA-lagged loss curve.
- `gnorm_ref`   : median gradient norm over the healthy (pre-blow-up) region, a scale
                  for a `ClipNorm` threshold.
- `safe`        : second-smallest finite of `(steep, min_over_10, gnorm_cap)`, the
                  value to use for `lr_start`. The range test runs on an UNTRAINED
                  model, so the peak LR must survive the harshest regime it will see.
"""
function recommend_lr(lrs, losses, gnorms = nothing;
                      skip_start::Int = 5, skip_end::Int = 5, smooth_win::Int = 3,
                      gnorm_factor::Float64 = 3.0, gnorm_backoff::Float64 = 3.0,
                      gnorm_consecutive::Int = 3, gnorm_smooth_beta::Float64 = 0.9,
                      loss_rise_lookback::Int = 3, loss_rise_factor::Float64 = 1.02,
                      lr_floor_factor::Float64 = 10.0)
    n = length(lrs)
    lr_floor = n >= 1 ? minimum(lrs) : NaN
    if n < 8
        m10 = n >= 1 ? lrs[argmin(losses)] / 10 : NaN
        min10_near = isfinite(lr_floor) && isfinite(m10) && m10 <= lr_floor_factor * lr_floor
        return (steep = NaN, min_over_10 = m10, gnorm_cap = Inf,
                gnorm_ref = NaN, safe = m10,
            near_lr_floor = min10_near, min10_near_floor = min10_near)
    end

    loglr = log10.(lrs)
    slope = diff(losses) ./ diff(loglr)           # d(loss)/d(log10 lr), length n-1

    # Moving-average smoothing of the slope.
    sm = similar(slope)
    for i in eachindex(slope)
        a = max(firstindex(slope), i - smooth_win ÷ 2)
        b = min(lastindex(slope),  i + smooth_win ÷ 2)
        sm[i] = mean(@view slope[a:b])
    end

    # Search only the interior: drop the warmup head and the divergence tail.
    lo = clamp(skip_start, firstindex(sm), lastindex(sm))
    hi = clamp(length(sm) - skip_end, lo, lastindex(sm))
    seg = @view sm[lo:hi]
    i   = lo + argmin(seg) - 1                     # steepest descent within interior
    steep = sqrt(lrs[i] * lrs[i + 1])              # geometric midpoint of the segment
    imin  = argmin(losses)
    min_over_10 = lrs[imin] / 10

    # Gradient-norm blow-up detector: the earliest reliable divergence signal.
    gnorm_cap = Inf
    gnorm_ref = NaN
    if gnorms !== nothing && length(gnorms) == n
        # Plateau reference = median grad norm over the trustworthy head
        # (after the EMA warmup, before any blow-up).
        head_hi   = clamp(skip_start + max(1, (n - skip_start) ÷ 3), skip_start + 1, n)
        ref_slice = filter(isfinite, collect(@view gnorms[(skip_start + 1):head_hi]))
        gnorm_ref = isempty(ref_slice) ? NaN : median(ref_slice)
        if isfinite(gnorm_ref) && gnorm_ref > 0
            thresh = gnorm_factor * gnorm_ref
            # Smooth grad norms and require a sustained breach that coincides
            # with rising loss, so one early transient spike cannot force the LR
            # to the search floor.
            g_ema = zeros(Float64, n)
            first_finite = findfirst(isfinite, gnorms)
            if first_finite !== nothing
                g_ema[first_finite] = Float64(gnorms[first_finite])
                for k in (first_finite + 1):n
                    gk = isfinite(gnorms[k]) ? Float64(gnorms[k]) : Inf
                    g_ema[k] = gnorm_smooth_beta * g_ema[k - 1] + (1 - gnorm_smooth_beta) * gk
                end

                warm_lo = max(skip_start + 1, first_finite + loss_rise_lookback)
                streak = 0
                for k in warm_lo:n
                    prev_lo = max(1, k - loss_rise_lookback)
                    recent_min = minimum(@view losses[prev_lo:k-1])
                    rising = losses[k] > loss_rise_factor * recent_min
                    breached = !isfinite(g_ema[k]) || g_ema[k] > thresh
                    if breached && rising
                        streak += 1
                        if streak >= gnorm_consecutive
                            gnorm_cap = lrs[k] / gnorm_backoff
                            break
                        end
                    else
                        streak = 0
                    end
                end
            end
        end
    end

    # Robust aggregation: ignore one overly conservative outlier by taking the
    # second-smallest finite estimate when possible.
    candidates = sort(filter(isfinite, Float64[steep, min_over_10, gnorm_cap]))
    safe = isempty(candidates) ? NaN : candidates[min(2, length(candidates))]
    near_lr_floor = isfinite(lr_floor) && isfinite(safe) && safe <= lr_floor_factor * lr_floor
        min10_near = isfinite(lr_floor) && isfinite(min_over_10) && min_over_10 <= lr_floor_factor * lr_floor
    return (steep = steep, min_over_10 = min_over_10, gnorm_cap = gnorm_cap,
            gnorm_ref = gnorm_ref, safe = safe, near_lr_floor = near_lr_floor,
            min10_near_floor = min10_near)
end

"""
    aggregate_lr_recommendations(recs) -> NamedTuple

Aggregate multiple `recommend_lr` outputs robustly for init-sensitive curves.
Returns median `safe` and median finite `gnorm_ref` over valid recommendations.
Recommendations are considered valid only when `safe` is finite and not flagged
as `near_lr_floor`.
"""
function aggregate_lr_recommendations(recs::AbstractVector)
    valids = [r for r in recs if isfinite(r.safe) && !getproperty(r, :near_lr_floor)]
    if isempty(valids)
        # If every conservative pick is near-floor, fall back to a robust
        # loss-curve estimate (`min_over_10`) across inits.
        min10_valid = [r for r in recs if hasproperty(r, :min_over_10) &&
                                     isfinite(r.min_over_10) &&
                                     !(hasproperty(r, :min10_near_floor) && r.min10_near_floor)]
        if isempty(min10_valid)
            return (; safe = NaN, gnorm_ref = NaN, n_valid = 0, n_total = length(recs), used_fallback = false)
        end
        m10s = [r.min_over_10 for r in min10_valid]
        refs = [r.gnorm_ref for r in min10_valid if isfinite(r.gnorm_ref)]
        return (; safe = median(m10s),
                gnorm_ref = isempty(refs) ? NaN : median(refs),
                n_valid = 0, n_total = length(recs), used_fallback = true)
    end
    safes = [r.safe for r in valids]
    refs  = [r.gnorm_ref for r in valids if isfinite(r.gnorm_ref)]
    return (; safe = median(safes),
            gnorm_ref = isempty(refs) ? NaN : median(refs),
            n_valid = length(valids), n_total = length(recs), used_fallback = false)
end

"Print about `rows` evenly spaced points of a range-test curve."
function print_lr_curve(lrs, losses, gnorms; rows::Int = 20)
    n = length(lrs)
    n == 0 && return
    idx = unique(round.(Int, range(1, n; length = min(rows, n))))
    println("\n    step        lr        smooth_loss     grad_norm")
    println("  ---------------------------------------------------------")
    for j in idx
        @printf("  %6d   %.3e     %.6g      %.4g\n", j, lrs[j], losses[j], gnorms[j])
    end
end

"Write a range-test curve to `path` as CSV (`step,lr,smooth_loss,grad_norm`)."
function write_lr_curve_csv(path, lrs, losses, gnorms)
    open(path, "w") do io
        println(io, "step,lr,smooth_loss,grad_norm")
        for j in eachindex(lrs)
            @printf(io, "%d,%.8e,%.8e,%.8e\n", j, lrs[j], losses[j], gnorms[j])
        end
    end
    @info "Wrote LR-range-test curve to $path"
    return path
end

# ---------------------------------------------------------------------------
# Batch-size estimator
# ---------------------------------------------------------------------------

"""
    estimate_batch_size(gd, ds, ms, ts; horizon, candidates, mem_fraction = 0.9) -> Int

For a fixed rollout `horizon`, probe an increasing sequence of batch sizes,
run one real forward+backward step at each and report peak GPU memory until a
candidate runs out of memory (or crosses `mem_fraction` of the device memory).
Returns the largest batch that fit (`0` when none did). On the CPU the memory
columns are `NaN` and the largest candidate that runs is returned.
"""
function estimate_batch_size(gd, ds::DataSettings, ms::ModelSettings, ts::TrainSettings;
                             horizon::Int, candidates::Vector{Int},
                             mem_fraction::Float64 = 0.9)
    println("\n  Batch-size probe at horizon = $horizon " *
            (gpu_active() ? "(GPU)" : "(CPU - memory figures are N/A)"))
    println("  ---------------------------------------------------------")
    @printf("  %8s   %12s   %10s   %10s\n", "batch", "peak_mem_GiB", "frac", "step_s")

    best_fit = 0
    for B in candidates
        setup = nothing
        try
            free_gpu!()
            setup = make_model_loader(gd, ds, ms, ts; horizon = horizon, batch_size = B)
            batch = first(setup.loader)
            opt   = Flux.setup(Adam(1f-3), setup.model)

            # Warm-up step (compilation + pool growth), then a timed step.
            _l, g = Flux.withgradient(m -> loss_function(m, batch, setup.strategy, setup.static_d), setup.model)
            Flux.update!(opt, setup.model, g[1])
            gpu_active() && CUDA.synchronize()

            t = @elapsed begin
                _l2, g2 = Flux.withgradient(m -> loss_function(m, batch, setup.strategy, setup.static_d), setup.model)
                Flux.update!(opt, setup.model, g2[1])
                gpu_active() && CUDA.synchronize()
            end

            mem  = gpu_mem_gib()
            frac = mem.used / mem.total
            @printf("  %8d   %12.3f   %10.2f   %10.4f\n", setup.batch_size, mem.used, frac, t)

            best_fit = setup.batch_size

            if gpu_active() && frac > mem_fraction
                @info @sprintf("Batch %d exceeds %.0f%% of GPU memory; stopping probe.",
                               setup.batch_size, 100 * mem_fraction)
                break
            end
            # If the loader clamped the batch (few windows), larger candidates add nothing.
            setup.batch_size < B && break
        catch e
            if is_oom(e)
                @info "Batch $B: out of GPU memory; stopping probe."
                break
            else
                rethrow()
            end
        finally
            setup = nothing
            free_gpu!()
        end
    end

    println("  ---------------------------------------------------------")
    if best_fit > 0
        rec = gpu_active() ? max(1, best_fit ÷ 2) : best_fit
        @info "Largest batch that fit at horizon $horizon: $best_fit" *
              (gpu_active() ? "  (recommended with headroom: $rec)" : "")
        return best_fit
    else
        @warn "No batch size fit (even B=1 failed at horizon $horizon)."
        return 0
    end
end

# ---------------------------------------------------------------------------
# Autotune
# ---------------------------------------------------------------------------

# Format a value for TOML: ints stay ints, everything else is a Float64 literal.
_toml_val(x::Integer) = string(x)
_toml_val(x::Real)    = string(Float64(x))

"""
    update_train_table(text, updates) -> String

Update (or insert) `key = value` pairs inside the `[train]` table of the config
`text`, preserving all comments elsewhere. `updates` maps key => already formatted
TOML value string.
"""
function update_train_table(text::AbstractString, updates::AbstractDict)
    lines = collect(split(text, '\n'))

    ti = findfirst(l -> occursin(r"^\s*\[train\]\s*$", l), lines)
    ti === nothing && error("No [train] table found in config")

    # The [train] table ends at the next table header (e.g. [train.strategy]).
    te = findnext(l -> occursin(r"^\s*\[", l), lines, ti + 1)
    te = te === nothing ? length(lines) + 1 : te

    remaining = Set(keys(updates))
    for i in (ti + 1):(te - 1)
        for k in collect(remaining)
            if occursin(Regex("^\\s*" * k * "\\s*="), lines[i])
                m = match(r"(#.*)$", lines[i])                # preserve trailing comment
                comment = m === nothing ? "" : "  " * m.captures[1]
                lines[i] = "$k = $(updates[k])$comment"
                delete!(remaining, k)
            end
        end
    end

    if !isempty(remaining)
        ins = te - 1                                          # insert after last table line
        while ins > ti && strip(lines[ins]) == ""
            ins -= 1
        end
        newlines = String["$k = $(updates[k])" for k in keys(updates) if k in remaining]
        lines = vcat(lines[1:ins], newlines, lines[(ins + 1):end])
    end

    return join(lines, '\n')
end

"""
    autotune(config; kwargs...) -> tuned_config_path

Derive the curriculum-schedule knobs for the run config TOML `config` and write a
tuned config:

1. Largest batch size that fits at the DEEPEST curriculum horizon
   ([`estimate_batch_size`](@ref); halved on a GPU for headroom).
2. Robust peak `lr_start` and a `grad_clip` from LR range tests on the UNTRAINED
   model at the shallow horizon with `range_inits` seeds
   ([`lr_range_test`](@ref), [`recommend_lr`](@ref), [`aggregate_lr_recommendations`](@ref)).
   The deep horizon is not used for the absolute LR: an untrained free-running rollout
   diverges on its own there, so `lr_peak_decay` is derived from the RELATIVE
   gradient growth ([`gradient_growth_probe`](@ref)) instead.
3. `lr_final = lr_start / final_divisor`.
4. The `[train]` keys `batch_size`, `lr_start`, `lr_final`, `lr_peak_decay`,
   `grad_clip` are updated text-wise ([`update_train_table`](@ref)), so comments
   survive. Writes `<config>_autotuned.toml`, or overwrites `config` (keeping a
   `.bak` copy) with `in_place = true`.

Keyword arguments (defaults in parentheses): `num_steps` (100), `lr_min` (1e-7),
`lr_max` (1.0), `mem_fraction` (0.9), `final_divisor` (100), `clip_mult` (5),
`clip_min` (0.1), `clip_max` (10), `seed` (42), `range_inits` (3),
`probe_steps` (12), `decay_min` (0.3), `decay_max` (0.9), `candidates`
(`[1,2,4,8,16,32,64]`), `range_horizon` (1), `in_place` (false), and the overrides
`batch_size`, `lr_start`, `grad_clip`, `lr_peak_decay` (`nothing` = estimate).
"""
function autotune(config::AbstractString;
                  num_steps::Int = 100,
                  lr_min::Real = 1e-7,
                  lr_max::Real = 1.0,
                  mem_fraction::Real = 0.9,
                  final_divisor::Real = 100.0,
                  clip_mult::Real = 5.0,
                  clip_min::Real = 0.1,
                  clip_max::Real = 10.0,
                  seed::Int = 42,
                  range_inits::Int = 3,
                  probe_steps::Int = 12,
                  decay_min::Real = 0.3,
                  decay_max::Real = 0.9,
                  candidates::Vector{Int} = [1, 2, 4, 8, 16, 32, 64],
                  range_horizon::Int = 1,
                  in_place::Bool = false,
                  batch_size::Union{Nothing,Int} = nothing,
                  lr_start::Union{Nothing,Real} = nothing,
                  grad_clip::Union{Nothing,Real} = nothing,
                  lr_peak_decay::Union{Nothing,Real} = nothing)
    isfile(config) || error("Config not found: $config")
    lr_min, lr_max = Float64(lr_min), Float64(lr_max)

    ds, ms, ts = parse_run_config(config)
    deep_horizon = maximum(ts.strategy.steps)

    # The LR range test runs on an UNTRAINED model. At a DEEP horizon an untrained
    # free-running rollout diverges on its own, so its ramped-LR curve reflects
    # rollout blow-up rather than the structural stiffness that survives real
    # training — an unreliable basis for an absolute LR. We therefore measure the
    # absolute `lr_start` ONLY at the shallow horizon (trustworthy, no rollout
    # compounding) and handle the deep-horizon sensitivity RELATIVELY via
    # `lr_peak_decay` (gradient-growth probe below) plus the runtime
    # `phase_backoff_factor`.
    shallow_horizon = ts.strategy.steps[1]
    eff_range_horizon = min(range_horizon, shallow_horizon)
    if range_horizon != eff_range_horizon
        @warn @sprintf("Capping range-horizon %d -> %d to avoid untrained deep-rollout instability in LR range test.",
                       range_horizon, eff_range_horizon)
    end

    @info "Auto-tune config: $config"
    @info "Device=$(ts.device)  curriculum steps=$(ts.strategy.steps)  deepest horizon=$deep_horizon"
    @info "Probe seed=$seed  range_inits=$range_inits"

    @info "Building graph time series (once)"
    gd = load_tuning_data(ds, ms)

    # --- 1. Batch size ------------------------------------------------------
    local bs::Int
    if batch_size !== nothing
        bs = batch_size
        @info "Using user-supplied batch_size=$bs (skipping probe)."
    else
        fit = estimate_batch_size(gd, ds, ms, ts; horizon = deep_horizon,
                                  candidates = candidates, mem_fraction = Float64(mem_fraction))
        fit > 0 || error("Batch-size probe found no batch that fits (even B=1).")
        bs = gpu_active() ? max(1, fit ÷ 2) : fit
        @info "Selected batch_size=$bs (largest fit=$fit)."
    end

    # --- 2. Peak learning rate + gradient clip (at the shallow horizon) -----
    local lr0::Float64
    local clip::Float64
    if lr_start !== nothing
        lr0  = Float64(lr_start)
        clip = Float64(something(grad_clip, 0.0))
        @info "Using user-supplied lr_start=$lr0 (skipping range test)."
    else
        @info @sprintf("LR range test: %d steps  lr %.1e -> %.1e  horizon=%d  batch=%d  inits=%d",
                       num_steps, lr_min, lr_max, eff_range_horizon, bs, range_inits)

        recs = NamedTuple[]
        for i in 1:max(1, range_inits)
            seed_i = seed + i - 1
            setup = make_model_loader(gd, ds, ms, ts;
                                      horizon = eff_range_horizon,
                                      batch_size = bs,
                                      seed = seed_i,
                                      shuffle = false)
            @info "Range test init $i effective loss: loss_type=$(setup.strategy.loss_type) " *
                "peak_delta=$(setup.strategy.peak_delta) peak_lambda=$(setup.strategy.peak_lambda) " *
                "peak_gamma=$(setup.strategy.peak_gamma) peak_w_max=$(setup.strategy.peak_w_max)"
            lrs, losses, gnorms = lr_range_test(setup; num_steps = num_steps,
                                                lr_min = lr_min, lr_max = lr_max)
            print_lr_curve(lrs, losses, gnorms)
            csv = joinpath(dirname(abspath(config)),
                           @sprintf("lr_range_test_h%d_b%d_s%d_i%d.csv",
                                    eff_range_horizon, setup.batch_size, seed_i, i))
            write_lr_curve_csv(csv, lrs, losses, gnorms)

            rec = recommend_lr(lrs, losses, gnorms)
            push!(recs, rec)
            @info @sprintf(
                "  init %d (seed=%d): safe=%.3e  steep=%.3e  min/10=%.3e  gnorm_cap=%.3e  gnorm_ref=%.3e  near_floor=%s",
                i, seed_i, rec.safe, rec.steep, rec.min_over_10, rec.gnorm_cap, rec.gnorm_ref,
                string(rec.near_lr_floor))
            free_gpu!()
        end

        agg = aggregate_lr_recommendations(recs)
        isfinite(agg.safe) || error("LR range test produced no robust finite recommendation across inits.")
        agg.safe <= 10.0 * lr_min && error(@sprintf(
            "Robust LR recommendation %.3e is within one decade of lr_min %.3e; range test likely failed across inits. Re-run with explicit lr_start or inspect curves.",
            agg.safe, lr_min))
        agg.used_fallback && @warn "All conservative safe picks were near-floor; using median min_over_10 fallback across inits."

        lr0 = agg.safe
        # Clip the global grad norm to a few× the healthy (pre-blow-up) norm so a
        # transient spike at a curriculum restart is tamed without throttling
        # normal updates. Clamp to a sane range so spikes cannot inflate the
        # recommendation into a no-op clip threshold. `grad_clip` overrides.
        clip = isfinite(agg.gnorm_ref) ?
               clamp(Float64(clip_mult) * agg.gnorm_ref, Float64(clip_min), Float64(clip_max)) :
               Float64(clip_max)
        grad_clip !== nothing && (clip = Float64(grad_clip))
        @info @sprintf("Range test (robust median over %d/%d valid inits) → lr_start=%.3e (horizon %d)  grad_clip=%.3e",
                       agg.n_valid, agg.n_total, lr0, eff_range_horizon, clip)
        free_gpu!()
    end

    # --- 3. Curriculum peak decay (gradient-growth probe) ------------------
    # BPTT inflates gradients as the rollout horizon grows, so each warm-restart
    # phase needs a smaller peak LR. The RELATIVE growth of the gradient norm from
    # the shallow to the deep phase, measured at a fixed tiny LR so the rollout is
    # not driven unstable, gives the geometric per-phase shrink factor. A deeper
    # phase over-reports growth on an untrained model, so this biases the peak
    # smaller (the safe direction). `lr_peak_decay` overrides the probe.
    local peak_decay::Float64
    n_phases = length(ts.strategy.steps)
    if lr_peak_decay !== nothing
        peak_decay = Float64(lr_peak_decay)
        @info @sprintf("Using user-supplied lr_peak_decay=%.3g (skipping probe).", peak_decay)
    elseif n_phases <= 1
        peak_decay = 1.0
        @info "Single curriculum phase → lr_peak_decay=1.0 (no decay)."
    else
        shallow_h = ts.strategy.steps[1]
        deep_h    = ts.strategy.steps[end]
        s_shallow = make_model_loader(gd, ds, ms, ts;
                          horizon = shallow_h, batch_size = bs,
                          seed = seed + 10_000, shuffle = false)
        g_shallow = gradient_growth_probe(s_shallow; num_steps = probe_steps)
        free_gpu!()
        s_deep = make_model_loader(gd, ds, ms, ts;
                       horizon = deep_h, batch_size = bs,
                       seed = seed + 20_000, shuffle = false)
        g_deep = gradient_growth_probe(s_deep; num_steps = probe_steps)
        free_gpu!()
        if isfinite(g_shallow) && isfinite(g_deep) && g_deep > 0 && g_shallow > 0
            ratio      = min(g_shallow / g_deep, 1.0)   # deep grads are larger → ratio ≤ 1
            peak_decay = clamp(ratio ^ (1 / (n_phases - 1)), Float64(decay_min), Float64(decay_max))
            @info @sprintf(
                "Gradient-growth probe: g(h=%d)=%.3e  g(h=%d)=%.3e  ratio=%.3g → lr_peak_decay=%.3g",
                shallow_h, g_shallow, deep_h, g_deep, ratio, peak_decay)
        else
            peak_decay = Float64(decay_min)
            @warn @sprintf("Gradient-growth probe non-finite (g_shallow=%.3e g_deep=%.3e); using lr_peak_decay=%.3g",
                           g_shallow, g_deep, peak_decay)
        end
    end

    # --- 4. Remaining curriculum knobs -------------------------------------
    lr_final = lr0 / Float64(final_divisor)
    @info @sprintf("Curriculum knobs: lr_peak_decay=%.3g  lr_final=lr_start/%.3g=%.3e",
                   peak_decay, Float64(final_divisor), lr_final)

    updates = Dict(
        "batch_size"    => _toml_val(bs),
        "lr_start"      => _toml_val(lr0),
        "lr_final"      => _toml_val(lr_final),
        "lr_peak_decay" => _toml_val(peak_decay),
        "grad_clip"     => _toml_val(clip),
    )

    # --- 5. Write the tuned config -----------------------------------------
    text = read(config, String)
    tuned_text = update_train_table(text, updates)

    if in_place
        cp(config, config * ".bak"; force = true)
        write(config, tuned_text)
        @info "Updated config in place (backup: $(config).bak)"
        return config
    else
        base, ext = splitext(config)
        out = base * "_autotuned" * ext
        write(out, tuned_text)
        @info "Wrote tuned config to $out"
        return out
    end
end
