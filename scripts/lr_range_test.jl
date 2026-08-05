# =============================================================================
# LR range test + batch-size estimator for WflowRoutingGNN
# =============================================================================
#
# A cheap proxy for tuning the curriculum LR schedule *without* a full training
# run. Two modes:
#
#   range  (default) : Leslie-Smith LR range test. Exponentially ramps the LR
#                      over a few hundred mini-batches at a FIXED rollout
#                      horizon, recording the (smoothed) loss and gradient norm.
#                      The max usable LR is roughly the point of steepest loss
#                      descent (about one order of magnitude below divergence).
#
#   batch            : Batch-size estimator. For a fixed max horizon, probes an
#                      increasing sequence of batch sizes, runs one real
#                      forward+backward step at each, and reports peak GPU memory
#                      until it OOMs (or crosses a memory-fraction ceiling). The
#                      largest batch that fits is the recommendation.
#
#   both             : run `batch` then `range` (at the estimated batch size).
#
# -----------------------------------------------------------------------------
# On batch size vs LR (why we do NOT sweep batch size in the range test):
#
#   Larger batches average out gradient noise, so the stable/optimal LR rises
#   with batch size (roughly the sqrt-scaling rule for Adam). Because you train
#   each config at ONE fixed batch size, run the range test at THAT batch size
#   and you get a directly-usable number - no batch sweep needed. Only sweep
#   batch size if you intend to transfer an LR across batch sizes.
#
#   The dominant knob for THIS project's warm-restart *peaks* is the rollout
#   HORIZON, not the batch size: BPTT multiplies gradient magnitudes through the
#   unrolled steps, so the stable LR falls as the horizon grows. Hence run the
#   range test at (at least) the shallowest and the deepest curriculum horizons
#   and let `lr_peak_decay` interpolate between the two peaks.
#
# -----------------------------------------------------------------------------
# Usage:
#
#   julia --project=. scripts/lr_range_test.jl <config.toml> [options]
#
# Options (all optional):
#   --mode MODE          range | batch | both            (default: range)
#   --horizon N          rollout horizon (steps ahead)   (default: max strategy step)
#   --batch-size B       batch size for the range test    (default: config batch_size)
#   --num-steps K        LR-ramp iterations               (default: 100)
#   --lr-min X           smallest LR probed               (default: 1e-7)
#   --lr-max X           largest LR probed                (default: 1.0)
#   --candidates a,b,c   batch sizes to probe (batch mode)(default: 1,2,4,8,16,32,64)
#   --mem-fraction F     stop batch probe above this frac (default: 0.9)
#   --out PATH           CSV output path (range mode)     (default: next to config)
#
# Examples:
#   julia --project=. scripts/lr_range_test.jl experiments/test_sava_v081/config.toml
#   julia --project=. scripts/lr_range_test.jl experiments/test_sava_v081/config.toml --mode both --horizon 10
# =============================================================================

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_gnn_model, load_schema, _grad_l2norm
using Flux
using MLUtils: DataLoader
using GraphNeuralNetworks
using SparseArrays
using Statistics
using Printf
using CUDA

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function parse_cli(argv)
    length(argv) >= 1 || error("Usage: lr_range_test.jl <config.toml> [options]")
    config = argv[1]
    opts = Dict{String,String}()
    i = 2
    while i <= length(argv)
        tok = argv[i]
        startswith(tok, "--") || error("Unexpected argument: $tok")
        if occursin('=', tok)
            k, v = split(tok[3:end], '='; limit = 2)
            opts[String(k)] = String(v)
            i += 1
        else
            k = tok[3:end]
            i + 1 <= length(argv) || error("Missing value for --$k")
            opts[String(k)] = String(argv[i + 1])
            i += 2
        end
    end
    return config, opts
end

optget(opts, k, default) = haskey(opts, k) ? opts[k] : default

# ---------------------------------------------------------------------------
# Setup builders (build the graph time series once; cheap per-config rebuilds)
# ---------------------------------------------------------------------------

function load_graphs(ds, ms)
    staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
    output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")
    schema = load_schema(ds.wflow_schema)
    graphs, norm_stats, grid, postscale, static_arr =
        build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)
    return (; graphs, norm_stats, postscale, static_arr, output_file)
end

# Build a model + single-horizon loader on the target device for a given
# (horizon, batch_size). Reuses the pre-built graph data `gd`.
function make_model_loader(gd, ds, ms, ts; horizon::Int, batch_size::Int)
    nhorizon = horizon + 1
    dataset  = make_horizon_dataset(gd.graphs, nhorizon; at = (ds.train_frac, ds.val_frac))
    bs = min(batch_size, length(dataset.train))
    loader = DataLoader(dataset.train;
                        batchsize = bs, shuffle = true, collate = true, parallel = true)

    # Fixed-horizon strategy: current_steps = horizon (single phase, no schedule).
    strategy = TrainingStrategy([horizon], [1], ts.strategy.noise_scale;
                                h_loss_weight = ts.strategy.h_loss_weight)

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

# Median gradient L2 norm over a handful of steps at a FIXED small LR and a fixed
# rollout `horizon`. Unlike the LR range test — which RAMPS the LR and, at deep
# horizons on an UNTRAINED model, drives the free-running rollout into artificial
# blow-up — this holds a small constant LR and merely records the gradient norm.
#
# The ABSOLUTE value is still inflated by the untrained rollout, so it is NOT a
# usable LR. It is used only for a RELATIVE comparison across horizons: the ratio
# g(shallow)/g(deep) sets `lr_peak_decay`, and because a deeper horizon over-
# reports its gradient growth, the derived peak-shrink errs on the conservative
# (smaller deep-phase peak) side — the safe direction.
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

# Suggested LRs from the recorded curve.
#
# Returns three estimates plus a single conservative pick:
#   * `steep`       - classic "point of steepest descent" (minimum of
#                     d(loss)/d(log lr)), computed robustly: the EMA-biased head
#                     and the diverging tail are trimmed and the slope is smoothed
#                     over a small window so a single noisy segment cannot win.
#   * `min_over_10` - fast.ai fallback (min-loss LR / 10). Optimistic when the
#                     loss minimum sits right under the divergence cliff (its
#                     apparent position is pushed HIGH by the EMA lag, so this is
#                     the estimate biased high - not `steep`).
#   * `gnorm_cap`   - the LR at which the gradient norm first blows past a
#                     multiple of its early-training plateau, backed off by a
#                     safety factor. Gradient-norm blow-up is the earliest,
#                     most reliable divergence signal and is not masked by the
#                     EMA-lagged loss curve.
#   * `gnorm_ref`   - median gradient norm over the healthy (pre-blow-up) region;
#                     a scale for setting a ClipNorm threshold.
#   * `safe = min(steep, min_over_10, gnorm_cap)` - the value to use for
#                     `lr_start`. The range test runs on an UNTRAINED model, so
#                     the peak LR must survive the harshest regime it will ever
#                     see, with no chance to re-tune once training has begun.
function recommend_lr(lrs, losses, gnorms = nothing;
                      skip_start::Int = 5, skip_end::Int = 5, smooth_win::Int = 3,
                      gnorm_factor::Float64 = 3.0, gnorm_backoff::Float64 = 3.0)
    n = length(lrs)
    if n < 8
        m10 = n >= 1 ? lrs[argmin(losses)] / 10 : NaN
        return (steep = NaN, min_over_10 = m10, gnorm_cap = Inf,
                gnorm_ref = NaN, safe = m10)
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
            for k in (skip_start + 1):n
                if !isfinite(gnorms[k]) || gnorms[k] > thresh
                    gnorm_cap = lrs[k] / gnorm_backoff
                    break
                end
            end
        end
    end

    safe = minimum(filter(isfinite, Float64[steep, min_over_10, gnorm_cap]))
    return (steep = steep, min_over_10 = min_over_10, gnorm_cap = gnorm_cap,
            gnorm_ref = gnorm_ref, safe = safe)
end

function print_curve(lrs, losses, gnorms; rows::Int = 20)
    n = length(lrs)
    n == 0 && return
    idx = unique(round.(Int, range(1, n; length = min(rows, n))))
    println("\n    step        lr        smooth_loss     grad_norm")
    println("  ---------------------------------------------------------")
    for j in idx
        @printf("  %6d   %.3e     %.6g      %.4g\n", j, lrs[j], losses[j], gnorms[j])
    end
end

function write_csv(path, lrs, losses, gnorms)
    open(path, "w") do io
        println(io, "step,lr,smooth_loss,grad_norm")
        for j in eachindex(lrs)
            @printf(io, "%d,%.8e,%.8e,%.8e\n", j, lrs[j], losses[j], gnorms[j])
        end
    end
    @info "Wrote LR-range-test curve to $path"
end

# ---------------------------------------------------------------------------
# Batch-size estimator
# ---------------------------------------------------------------------------

function estimate_batch_size(gd, ds, ms, ts; horizon::Int, candidates::Vector{Int},
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
# Main
# ---------------------------------------------------------------------------

function main()
    config, opts = parse_cli(ARGS)
    ds, ms, ts   = parse_run_config(config)

    mode         = optget(opts, "mode", "range")
    max_step     = maximum(ts.strategy.steps)
    horizon      = parse(Int, optget(opts, "horizon", string(max_step)))
    batch_size   = parse(Int, optget(opts, "batch-size", string(ts.batch_size)))
    num_steps    = parse(Int, optget(opts, "num-steps", "100"))
    lr_min       = parse(Float64, optget(opts, "lr-min", "1e-7"))
    lr_max       = parse(Float64, optget(opts, "lr-max", "1.0"))
    mem_fraction = parse(Float64, optget(opts, "mem-fraction", "0.9"))
    candidates   = [parse(Int, strip(s)) for s in split(optget(opts, "candidates", "1,2,4,8,16,32,64"), ',')]
    out_opt      = get(opts, "out", nothing)  # explicit --out overrides the auto name

    @info "Config: $config"
    @info "Mode=$mode  horizon=$horizon  device=$(ts.device)"

    @info "Building graph time series (once)"
    gd = load_graphs(ds, ms)

    if mode in ("batch", "both")
        fit = estimate_batch_size(gd, ds, ms, ts; horizon = horizon,
                                  candidates = candidates, mem_fraction = mem_fraction)
        if mode == "both" && fit > 0
            batch_size = gpu_active() ? max(1, fit ÷ 2) : fit
            @info "Using batch_size=$batch_size for the range test."
        end
    end

    if mode in ("range", "both")
        @info @sprintf("LR range test: %d steps  lr %.1e -> %.1e  horizon=%d  batch=%d",
                       num_steps, lr_min, lr_max, horizon, batch_size)
        setup = make_model_loader(gd, ds, ms, ts; horizon = horizon, batch_size = batch_size)
        lrs, losses, gnorms = lr_range_test(setup; num_steps = num_steps,
                                            lr_min = lr_min, lr_max = lr_max)
        print_curve(lrs, losses, gnorms)
        # Name the CSV after the batch size actually used (loader may clamp it).
        out_path = out_opt !== nothing ? out_opt :
                   joinpath(dirname(abspath(config)),
                            @sprintf("lr_range_test_h%d_b%d.csv", horizon, setup.batch_size))
        write_csv(out_path, lrs, losses, gnorms)

        rec = recommend_lr(lrs, losses, gnorms)
        println()
        @info @sprintf("Recommended lr_start (safe)     : %.3e", rec.safe)
        @info @sprintf("  steepest descent              : %.3e", rec.steep)
        @info @sprintf("  min-loss LR / 10              : %.3e", rec.min_over_10)
        @info @sprintf("  grad-norm blow-up cap         : %.3e", rec.gnorm_cap)
        @info @sprintf("  healthy grad-norm reference   : %.3e", rec.gnorm_ref)
        println("""

  How to use this for the curriculum schedule:
    * Set  lr_start        near the steepest-descent LR from the SHALLOWEST horizon.
    * Re-run at the DEEPEST horizon to get its stable LR, then choose
      lr_peak_decay so that  lr_start * lr_peak_decay^(num_phases-1)  lands
      at (or just below) that deep-horizon LR.
    * Set  lr_final        a little below the deepest stable LR (schedule floor).
""")
        free_gpu!()
    end
end

# Only auto-run when this file is executed directly (`julia scripts/lr_range_test.jl ...`).
# When `include`d from another script (e.g. autotune_train.jl) the helper functions
# above are reused without triggering a range test.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
