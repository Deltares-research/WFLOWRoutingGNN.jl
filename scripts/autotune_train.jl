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
# All heavy lifting reuses the helpers from lr_range_test.jl (graph building,
# model/loader construction, range test, batch estimator) and run.jl
# (parse_run_config, build_gnn_model, run_wflow_gnn_from_toml).
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
#   --dry-run               tune + write config but do NOT train
#
# Example:
#   julia --project=. scripts/autotune_train.jl experiments/test_sava_v081/config.toml
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

# Reuse every helper (graph build, model/loader, range test, batch estimator).
# The guard in lr_range_test.jl prevents its `main()` from firing on include.
include(joinpath(@__DIR__, "lr_range_test.jl"))

using Printf

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

# ---------------------------------------------------------------------------
# Config editing (text-based, so comments in the original file survive)
# ---------------------------------------------------------------------------

# Format a value for TOML: ints stay ints, everything else is a Float64 literal.
_toml_val(x::Integer) = string(x)
_toml_val(x::Real)    = string(Float64(x))

# Update (or insert) `key = value` pairs inside the [train] table of `text`,
# preserving all comments elsewhere. `updates` maps key => already-formatted
# TOML value string. Returns the modified text.
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

# ---------------------------------------------------------------------------
# Tuning driver
# ---------------------------------------------------------------------------

function autotune(config::AbstractString, opts::AbstractDict)
    isfile(config) || error("Config not found: $config")

    ds, ms, ts = parse_run_config(config)

    num_steps     = parse(Int,     optget(opts, "num-steps", "100"))
    lr_min        = parse(Float64, optget(opts, "lr-min", "1e-7"))
    lr_max        = parse(Float64, optget(opts, "lr-max", "1.0"))
    mem_fraction  = parse(Float64, optget(opts, "mem-fraction", "0.9"))
    final_divisor = parse(Float64, optget(opts, "lr-final-divisor", "100"))
    clip_mult     = parse(Float64, optget(opts, "grad-clip-mult", "5.0"))
    probe_steps   = parse(Int,     optget(opts, "probe-steps", "12"))
    decay_min     = parse(Float64, optget(opts, "peak-decay-min", "0.3"))
    decay_max     = parse(Float64, optget(opts, "peak-decay-max", "0.9"))
    candidates    = [parse(Int, strip(s)) for s in
                     split(optget(opts, "candidates", "1,2,4,8,16,32,64"), ',')]

    deep_horizon = maximum(ts.strategy.steps)

    # The LR range test runs on an UNTRAINED model. At a DEEP horizon an untrained
    # free-running rollout diverges on its own, so its ramped-LR curve reflects
    # rollout blow-up rather than the structural stiffness that survives real
    # training — an unreliable basis for an absolute LR. We therefore measure the
    # absolute `lr_start` ONLY at horizon 1 (trustworthy, no rollout compounding)
    # and handle the deep-horizon sensitivity RELATIVELY via `lr_peak_decay`
    # (gradient-growth probe below) plus the runtime `phase_backoff_factor`.
    # `--range-horizon N` overrides the range-test horizon with an explicit value.
    range_horizon = haskey(opts, "range-horizon") ?
        parse(Int, opts["range-horizon"]) : 1

    @info "Auto-tune config: $config"
    @info "Device=$(ts.device)  curriculum steps=$(ts.strategy.steps)  deepest horizon=$deep_horizon"

    @info "Building graph time series (once)"
    gd = load_graphs(ds, ms)

    # --- 1. Batch size ------------------------------------------------------
    local batch_size
    if haskey(opts, "batch-size")
        batch_size = parse(Int, opts["batch-size"])
        @info "Using user-supplied batch_size=$batch_size (skipping probe)."
    else
        fit = estimate_batch_size(gd, ds, ms, ts; horizon = deep_horizon,
                                  candidates = candidates, mem_fraction = mem_fraction)
        fit > 0 || error("Batch-size probe found no batch that fits (even B=1).")
        batch_size = gpu_active() ? max(1, fit ÷ 2) : fit
        @info "Selected batch_size=$batch_size (largest fit=$fit)."
    end

    # --- 2. Peak learning rate + gradient clip (at horizon 1) --------------
    local lr_start
    local grad_clip
    if haskey(opts, "lr-start")
        lr_start  = parse(Float64, opts["lr-start"])
        grad_clip = parse(Float64, optget(opts, "grad-clip", "0.0"))
        @info "Using user-supplied lr_start=$lr_start (skipping range test)."
    else
        @info @sprintf("LR range test: %d steps  lr %.1e -> %.1e  horizon=%d  batch=%d",
                       num_steps, lr_min, lr_max, range_horizon, batch_size)
        setup = make_model_loader(gd, ds, ms, ts; horizon = range_horizon,
                                  batch_size = batch_size)
        lrs, losses, gnorms = lr_range_test(setup; num_steps = num_steps,
                                            lr_min = lr_min, lr_max = lr_max)
        print_curve(lrs, losses, gnorms)
        csv = joinpath(dirname(abspath(config)),
                       @sprintf("lr_range_test_h%d_b%d.csv", range_horizon, setup.batch_size))
        write_csv(csv, lrs, losses, gnorms)

        rec = recommend_lr(lrs, losses, gnorms)
        @info @sprintf(
            "  horizon=%d → safe=%.3e  (steep=%.3e  min/10=%.3e  gnorm_cap=%.3e  gnorm_ref=%.3e)",
            range_horizon, rec.safe, rec.steep, rec.min_over_10, rec.gnorm_cap, rec.gnorm_ref)
        isfinite(rec.safe) || error("LR range test produced no finite recommendation.")
        lr_start = rec.safe
        # Clip the global grad norm to a few× the healthy (pre-blow-up) norm so a
        # transient spike at a curriculum restart is tamed without throttling
        # normal updates. Fall back to a conservative absolute cap if no healthy
        # reference was measured. `--grad-clip X` overrides.
        grad_clip = isfinite(rec.gnorm_ref) ? clip_mult * rec.gnorm_ref : 10.0
        haskey(opts, "grad-clip") && (grad_clip = parse(Float64, opts["grad-clip"]))
        @info @sprintf("Range test → lr_start=%.3e (horizon %d)  grad_clip=%.3e",
                       lr_start, range_horizon, grad_clip)
        free_gpu!()
    end

    # --- 3. Curriculum peak decay (gradient-growth probe) ------------------
    # BPTT inflates gradients as the rollout horizon grows, so each warm-restart
    # phase needs a smaller peak LR. We cannot get a trustworthy ABSOLUTE deep-
    # horizon LR offline (untrained rollout blow-up), but the RELATIVE growth of
    # the gradient norm from the shallow to the deep phase — measured at a fixed
    # tiny LR so the rollout is not driven unstable — gives the geometric per-
    # phase shrink factor. A deeper phase over-reports growth on an untrained
    # model, so this biases the peak smaller (the safe direction).
    # `--lr-peak-decay X` overrides the probe.
    local peak_decay
    n_phases = length(ts.strategy.steps)
    if haskey(opts, "lr-peak-decay")
        peak_decay = parse(Float64, opts["lr-peak-decay"])
        @info @sprintf("Using user-supplied lr_peak_decay=%.3g (skipping probe).", peak_decay)
    elseif n_phases <= 1
        peak_decay = 1.0
        @info "Single curriculum phase → lr_peak_decay=1.0 (no decay)."
    else
        shallow_h = ts.strategy.steps[1]
        deep_h    = ts.strategy.steps[end]
        s_shallow = make_model_loader(gd, ds, ms, ts; horizon = shallow_h,
                                      batch_size = batch_size)
        g_shallow = gradient_growth_probe(s_shallow; num_steps = probe_steps)
        free_gpu!()
        s_deep = make_model_loader(gd, ds, ms, ts; horizon = deep_h,
                                   batch_size = batch_size)
        g_deep = gradient_growth_probe(s_deep; num_steps = probe_steps)
        free_gpu!()
        if isfinite(g_shallow) && isfinite(g_deep) && g_deep > 0 && g_shallow > 0
            ratio      = min(g_shallow / g_deep, 1.0)   # deep grads are larger → ratio ≤ 1
            peak_decay = clamp(ratio ^ (1 / (n_phases - 1)), decay_min, decay_max)
            @info @sprintf(
                "Gradient-growth probe: g(h=%d)=%.3e  g(h=%d)=%.3e  ratio=%.3g → lr_peak_decay=%.3g",
                shallow_h, g_shallow, deep_h, g_deep, ratio, peak_decay)
        else
            peak_decay = decay_min
            @warn @sprintf("Gradient-growth probe non-finite (g_shallow=%.3e g_deep=%.3e); using lr_peak_decay=%.3g",
                           g_shallow, g_deep, peak_decay)
        end
    end

    # --- 4. Remaining curriculum knobs -------------------------------------
    lr_final = lr_start / final_divisor
    @info @sprintf("Curriculum knobs: lr_peak_decay=%.3g  lr_final=lr_start/%.3g=%.3e",
                   peak_decay, final_divisor, lr_final)

    updates = Dict(
        "batch_size"    => _toml_val(batch_size),
        "lr_start"      => _toml_val(lr_start),
        "lr_final"      => _toml_val(lr_final),
        "lr_peak_decay" => _toml_val(peak_decay),
        "grad_clip"     => _toml_val(grad_clip),
    )

    # --- 5. Write the tuned config -----------------------------------------
    text = read(config, String)
    tuned_text = update_train_table(text, updates)

    if haskey(opts, "in-place")
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function autotune_main()
    config, opts = parse_autotune_cli(ARGS)
    tuned_config = autotune(config, opts)

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
