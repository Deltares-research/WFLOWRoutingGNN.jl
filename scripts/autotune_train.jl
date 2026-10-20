# =============================================================================
# Auto-tune + train for WflowRoutingGNN
# =============================================================================
#
# One-shot pipeline that, given a run config TOML:
#
#   1. Estimates the largest batch size that fits in memory at the DEEPEST
#      curriculum horizon (the memory-heaviest phase).
#   2. Runs a Leslie-Smith LR range test on the UNTRAINED model at BOTH the
#      shallowest (horizon 1, most trustworthy) and the deepest curriculum
#      horizon (where BPTT makes gradients most explosive), and picks the most
#      conservative, gradient-norm-aware peak learning rate `lr_start`. It also
#      derives a `grad_clip` (global grad-norm clip) from the healthy grad norms.
#   3. Derives the remaining curriculum-schedule knobs with the "option 1"
#      heuristic:
#         lr_peak_decay = 0.8            (gentle geometric peak shrink)
#         lr_final      = lr_start / 100 (schedule floor)
#   4. Writes an updated config (batch_size, lr_start, lr_peak_decay, lr_final)
#      and launches the full training run on it.
#
# All heavy lifting is the `autotune` function of src/tune.jl (batch estimator,
# range test, gradient-growth probe, config writer); this script is the CLI.
# training goes through `run_wflow_gnn_from_toml` (src/run.jl).
#
# -----------------------------------------------------------------------------
# Usage:
#
#   julia --project=. scripts/autotune_train.jl <config.toml> [options]
#
# Options (all optional):
#   --in-place              overwrite the input config (a .bak copy is kept)
#                           instead of writing <config>_autotuned.toml
#   --num-steps K           LR-ramp iterations             (default: 100)
#   --lr-min X              smallest LR probed             (default: 1e-7)
#   --lr-max X              largest LR probed              (default: 1.0)
#   --range-horizon N       horizon for the LR range test  (default: 1)
#                           lr_start is measured at horizon 1 only; deep-horizon
#                           sensitivity is handled via lr_peak_decay (probe) and
#                           the runtime phase_backoff_factor.
#   --batch-size B          skip the batch probe, use B
#   --candidates a,b,c      batch sizes to probe           (default: 1,2,4,8,16,32,64)
#   --mem-fraction F        stop batch probe above frac    (default: 0.9)
#   --lr-peak-decay X       override the gradient-growth probe with a fixed
#                           curriculum peak-shrink factor
#   --probe-steps K         gradient-growth probe steps    (default: 12)
#   --peak-decay-min X      floor for the probed peak decay (default: 0.3)
#   --peak-decay-max X      cap for the probed peak decay   (default: 0.9)
#   --lr-final-divisor X    lr_final = lr_start / X        (default: 100)
#   --lr-start X            skip the range test, use X as the peak LR
#   --grad-clip X           explicit global grad-norm clip written to config
#   --grad-clip-mult X      grad_clip = X * healthy grad norm (default: 5)
#   --grad-clip-min X       lower clamp for auto grad_clip      (default: 0.1)
#   --grad-clip-max X       upper clamp for auto grad_clip      (default: 10)
#   --seed N                base seed for probe reproducibility (default: 42)
#   --range-inits N         number of inits for LR range test    (default: 3)
#   --dry-run               tune + write config but do NOT train
#
# Example:
#   julia --project=. scripts/autotune_train.jl experiments/test_sava_v081/config.toml
# =============================================================================
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config
using Printf

# Thin CLI over `autotune` (src/tune.jl). CLI parsing helpers (`optget`) are shared
# with lr_range_test.jl; its guard keeps its `main()` from firing on include.
include(joinpath(@__DIR__, "lr_range_test.jl"))

# ---------------------------------------------------------------------------
# CLI parsing (flag-aware: some options are boolean switches, not value pairs)
# ---------------------------------------------------------------------------

const AUTOTUNE_FLAGS = Set(["in-place", "dry-run"])

function parse_autotune_cli(argv)
    length(argv) >= 1 || error("Usage: autotune_train.jl <config.toml> [options]")
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
            if k in AUTOTUNE_FLAGS
                opts[k] = "true"
                i += 1
            else
                i + 1 <= length(argv) || error("Missing value for --$k")
                opts[k] = String(argv[i + 1])
                i += 2
            end
        end
    end
    return config, opts
end

# Map CLI options onto `autotune` keyword arguments.
function autotune_kwargs(opts::AbstractDict)
    kw = Dict{Symbol,Any}(
        :num_steps     => parse(Int,     optget(opts, "num-steps", "100")),
        :lr_min        => parse(Float64, optget(opts, "lr-min", "1e-7")),
        :lr_max        => parse(Float64, optget(opts, "lr-max", "1.0")),
        :mem_fraction  => parse(Float64, optget(opts, "mem-fraction", "0.9")),
        :final_divisor => parse(Float64, optget(opts, "lr-final-divisor", "100")),
        :clip_mult     => parse(Float64, optget(opts, "grad-clip-mult", "5.0")),
        :clip_min      => parse(Float64, optget(opts, "grad-clip-min", "0.1")),
        :clip_max      => parse(Float64, optget(opts, "grad-clip-max", "10.0")),
        :seed          => parse(Int,     optget(opts, "seed", "42")),
        :range_inits   => parse(Int,     optget(opts, "range-inits", "3")),
        :probe_steps   => parse(Int,     optget(opts, "probe-steps", "12")),
        :decay_min     => parse(Float64, optget(opts, "peak-decay-min", "0.3")),
        :decay_max     => parse(Float64, optget(opts, "peak-decay-max", "0.9")),
        :candidates    => [parse(Int, strip(s)) for s in
                           split(optget(opts, "candidates", "1,2,4,8,16,32,64"), ',')],
        :in_place      => haskey(opts, "in-place"),
    )
    haskey(opts, "range-horizon")  && (kw[:range_horizon]  = parse(Int,     opts["range-horizon"]))
    haskey(opts, "batch-size")     && (kw[:batch_size]     = parse(Int,     opts["batch-size"]))
    haskey(opts, "lr-start")       && (kw[:lr_start]       = parse(Float64, opts["lr-start"]))
    haskey(opts, "grad-clip")      && (kw[:grad_clip]      = parse(Float64, opts["grad-clip"]))
    haskey(opts, "lr-peak-decay")  && (kw[:lr_peak_decay]  = parse(Float64, opts["lr-peak-decay"]))
    return kw
end

function autotune_main()
    config, opts = parse_autotune_cli(ARGS)
    tuned_config = autotune(config; autotune_kwargs(opts)...)

    if haskey(opts, "dry-run")
        @info "Dry run: skipping training. Train with:"
        @info "  julia --project=. scripts/train.jl $tuned_config"
        return
    end

    @info "Starting training on $tuned_config"
    run_wflow_gnn_from_toml(tuned_config)
    @info "Done."
end

if abspath(PROGRAM_FILE) == @__FILE__
    autotune_main()
end
