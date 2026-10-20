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
using WflowRoutingGNN: gpu_active, free_gpu!
using Printf

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
# Main (the tuning toolkit itself lives in src/tune.jl)
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
    gd = load_tuning_data(ds, ms)

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
        @info "Range test effective loss: loss_type=$(setup.strategy.loss_type) " *
              "peak_delta=$(setup.strategy.peak_delta) peak_lambda=$(setup.strategy.peak_lambda) " *
              "peak_gamma=$(setup.strategy.peak_gamma) peak_w_max=$(setup.strategy.peak_w_max)"
        lrs, losses, gnorms = lr_range_test(setup; num_steps = num_steps,
                                            lr_min = lr_min, lr_max = lr_max)
        print_lr_curve(lrs, losses, gnorms)
        # Name the CSV after the batch size actually used (loader may clamp it).
        out_path = out_opt !== nothing ? out_opt :
                   joinpath(dirname(abspath(config)),
                            @sprintf("lr_range_test_h%d_b%d.csv", horizon, setup.batch_size))
        write_lr_curve_csv(out_path, lrs, losses, gnorms)

        rec = recommend_lr(lrs, losses, gnorms)
        rec.near_lr_floor && @warn @sprintf(
            "Recommended LR %.3e is within %.1fx of lr_min %.3e (likely fragile range test).",
            rec.safe, 10.0, minimum(lrs))
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
# When `include`d from another script (e.g. autotune_train.jl) the CLI helpers
# above are reused without triggering a range test.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
