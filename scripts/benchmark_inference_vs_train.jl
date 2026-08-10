"""
benchmark_inference_vs_train.jl

Reconcile the two per-step numbers that were being conflated:

  * a *rollout / inference* step  = one forward pass, and
  * a *training* step             = forward + backward (+ optimiser update).

For a trained experiment it loads the model and graph time series, then times —
on the SAME model, graph, static features and device, with a warm-up and a
`CUDA.synchronize()` after every timed call — three quantities:

    1. forward-only            model(g, state, forcing, static, forcing_next)
    2. forward + backward      Flux.withgradient(loss_function, model)
    3. full training step      (2) + ClipNorm + Adam Flux.update!

All three use exactly the same code paths as `rollout` (1) and `train_model!`
(2, 3), at the training `strategy.current_steps` horizon (1 for a 1-step model)
and `noise_scale = 0`.  This makes the forward:backward:update ratio explicit so
the honest inference cost of the surrogate is unambiguous.

Usage:
    julia --project=. scripts/benchmark_inference_vs_train.jl <experiment_dir> [options]

Options (all optional):
    --staticmaps PATH   Override staticmaps.nc path (default: from config).
    --output     PATH   Override output.nc path     (default: from config).
    --reps       N      Number of timed repetitions per quantity (default: 30).
    --warmup     N      Number of discarded warm-up calls          (default: 3).
    --index      K      Which timestep (graph index) to benchmark  (default: 1).
    --devices  LIST     Comma-separated devices: cpu,gpu
                        (default: cpu plus gpu when CUDA is functional).

Example:
    julia --project=. scripts/benchmark_inference_vs_train.jl experiments/test_sava_v081 \
        --reps 50 --devices cpu,gpu
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_wflow_graph, build_gnn_model,
                      load_schema, loss_function, TrainingStrategy
using Flux
using CUDA
using JLD2
using Statistics
using Printf

# ── CLI parsing ───────────────────────────────────────────────────────────────
function parse_args(args)
    isempty(args) && error("usage: benchmark_inference_vs_train.jl <experiment_dir> " *
                           "[--staticmaps ..] [--output ..] [--reps N] [--warmup N] " *
                           "[--index K] [--devices cpu,gpu]")
    exp_dir = args[1]
    opts = Dict{String,String}()
    i = 2
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument: $a")
        key = a[3:end]
        i + 1 <= length(args) || error("missing value for --$key")
        opts[key] = args[i+1]
        i += 2
    end
    return exp_dir, opts
end

const EXP_DIR, OPTS = parse_args(ARGS)

# ── Load config + trained model ───────────────────────────────────────────────
config_path = joinpath(EXP_DIR, "config.toml")
isfile(config_path) || error("config.toml not found in $EXP_DIR")
model_path  = joinpath(EXP_DIR, "model.jld2")
isfile(model_path)  || error("model.jld2 not found in $EXP_DIR (is this a trained run?)")

@info "Loading experiment config from $config_path"
ds, ms, ts = parse_run_config(config_path)

staticmaps_file = get(OPTS, "staticmaps", joinpath(ds.wflow_model_path, "staticmaps.nc"))
output_file     = get(OPTS, "output",     joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc"))
isfile(staticmaps_file) || error("staticmaps not found: $staticmaps_file")
isfile(output_file)     || error("output.nc not found: $output_file")

schema = load_schema(ds.wflow_schema)

@info "Building graph time series (domain=$(ms.domain), schema=$(ds.wflow_schema)) …"
graphs, norm_stats, _grid, postscale, static_arr =
    build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)

model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file, 1;
                        h_loss_scale = ts.h_loss_scale)
Flux.loadmodel!(model, JLD2.load(model_path, "model_state"))
@info "Restored trained weights from $model_path"

# ── Benchmark inputs ──────────────────────────────────────────────────────────
reps   = haskey(OPTS, "reps")   ? parse(Int, OPTS["reps"])   : 30
warmup = haskey(OPTS, "warmup") ? parse(Int, OPTS["warmup"]) : 3
idx    = haskey(OPTS, "index")  ? parse(Int, OPTS["index"])  : 1
(1 <= idx < length(graphs)) || error("--index must be in 1 … $(length(graphs) - 1)")

# One training step at the model's own horizon, matching train_model! exactly:
# batch = [g_t, g_{t+1}, …], strategy.current_steps steps ahead, noise off.
nsteps   = ts.strategy.current_steps
noise0   = ts.strategy.noise_scale
strategy = TrainingStrategy(ts.strategy.steps, ts.strategy.durations, 0f0;
                            h_loss_weight = ts.strategy.h_loss_weight)
strategy.current_steps = nsteps
(idx + nsteps <= length(graphs)) ||
    error("--index $idx + horizon $nsteps exceeds graph count $(length(graphs))")

batch_cpu = [graphs[idx + k] for k in 0:nsteps]      # nsteps+1 graphs

# Devices
requested = haskey(OPTS, "devices") ?
    Symbol.(strip.(split(OPTS["devices"], ","))) :
    (CUDA.functional() ? [:cpu, :gpu] : [:cpu])
devices = Symbol[]
for d in requested
    d in (:cpu, :gpu) || error("unknown device: $d (use cpu or gpu)")
    if d == :gpu && !CUDA.functional()
        @warn "GPU requested but CUDA is not functional; skipping GPU tests"
        continue
    end
    push!(devices, d)
end
isempty(devices) && error("no runnable devices")

@info @sprintf("graph: %d nodes  |  horizon (current_steps): %d  |  reps: %d  |  warmup: %d",
               graphs[idx].num_nodes, nsteps, reps, warmup)
if noise0 != 0f0
    @info @sprintf("note: training noise_scale=%.3g disabled for timing (deterministic inputs)", noise0)
end

# ── Timing helper ─────────────────────────────────────────────────────────────
# Runs `f` `warmup` times (discarded) then `reps` times (timed), calling `sync`
# after every call.  Returns per-call times in milliseconds.
function timeit(f, sync, warmup, reps)
    for _ in 1:warmup
        f(); sync()
    end
    ts_ms = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns()                      # nanosecond counter: ~2 ms GPU steps
        f(); sync()                         # would be lost to time()'s ~1 ms grid
        ts_ms[r] = (time_ns() - t0) / 1e6
    end
    return ts_ms
end

statline(v) = (min = minimum(v), median = median(v),
               mean = Statistics.mean(v),
               std = length(v) > 1 ? std(v) : 0.0, max = maximum(v))

results = NamedTuple[]

for dev in devices
    dev_fn = dev == :gpu ? Flux.gpu : identity
    sync   = dev == :gpu ? (() -> CUDA.synchronize()) : (() -> nothing)

    model_d  = dev_fn(model)
    static_d = dev_fn(Array{Float32}(static_arr))
    batch_d  = dev_fn.(batch_cpu)

    g0    = batch_d[1]
    state = g0.ndata.state
    f_now = g0.ndata.forcing
    f_nxt = batch_d[2].ndata.forcing

    # 3-D forcing buffer + 3-D output buffer so the rollout-step closure below
    # performs the same array slicing/copying as the real `rollout` loop body:
    # slice forcing[:,:,t] and forcing[:,:,t+1], and store state into states[:,:,t].
    n_state   = size(state, 1)
    n_nodes   = size(state, 2)
    n_forcing = size(f_now, 1)
    forcing3  = dev_fn(Array{Float32}(cat(Array(Flux.cpu(f_now)), Array(Flux.cpu(f_nxt)); dims = 3)))
    states3   = similar(state, n_state, n_nodes, 1)

    # Fresh optimiser identical to train_model! (ClipNorm ∘ Adam).
    rule = ts.grad_clip > 0 ?
        Flux.Optimisers.OptimiserChain(
            Flux.Optimisers.ClipNorm(Float32(ts.grad_clip); throw = false),
            Adam(ts.lr_start)) :
        Adam(ts.lr_start)
    opt_state = Flux.setup(rule, model_d)

    # 1. forward-only inference (pure model call, no bookkeeping)
    fwd() = model_d(g0, state, f_now, static_d, f_nxt)
    t_fwd = timeit(fwd, sync, warmup, reps)

    # 1b. rollout step: forward + forcing slices + state store (the real
    #     `rollout` loop body). Extra ops vs. `fwd` are two forcing slices and
    #     one store into the output buffer — each a separate GPU kernel launch.
    function roll()
        f_next = forcing3[:, :, 2]
        st     = model_d(g0, state, forcing3[:, :, 1], static_d, f_next)
        states3[:, :, 1] = st
        return st
    end
    t_roll = timeit(roll, sync, warmup, reps)

    # 2. forward + backward (gradient), same loss as training
    grad() = Flux.withgradient(m -> loss_function(m, batch_d, strategy, static_d), model_d)
    t_grad = timeit(grad, sync, warmup, reps)

    # 3. full training step: gradient + optimiser update
    function step()
        _, gs = Flux.withgradient(m -> loss_function(m, batch_d, strategy, static_d), model_d)
        Flux.update!(opt_state, model_d, gs[1])
    end
    t_step = timeit(step, sync, warmup, reps)

    push!(results, (device = dev, kind = "forward (pure model)",  t = t_fwd))
    push!(results, (device = dev, kind = "rollout step",          t = t_roll))
    push!(results, (device = dev, kind = "fwd+bwd (gradient)",    t = t_grad))
    push!(results, (device = dev, kind = "full train step",       t = t_step))
end

# ── Report ────────────────────────────────────────────────────────────────────
println()
println("="^92)
println("INFERENCE vs TRAINING STEP  |  model: $(EXP_DIR)  |  $(graphs[idx].num_nodes) nodes  |  horizon $(nsteps)")
println("="^92)
@printf("%-6s %-22s | %8s %8s %8s %8s %8s\n",
        "device", "operation", "min", "median", "mean", "std", "max")
println("  (times in ms per step)")
println("-"^92)
for r in results
    s = statline(r.t)
    @printf("%-6s %-22s | %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            string(r.device), r.kind, s.min, s.median, s.mean, s.std, s.max)
end
println("-"^92)

# Ratios per device: how much each step costs relative to one pure forward pass.
for dev in devices
    fi = findfirst(r -> r.device == dev && r.kind == "forward (pure model)", results)
    ri = findfirst(r -> r.device == dev && r.kind == "rollout step",         results)
    gi = findfirst(r -> r.device == dev && r.kind == "fwd+bwd (gradient)",   results)
    si = findfirst(r -> r.device == dev && r.kind == "full train step",      results)
    if !isnothing(fi) && !isnothing(ri) && !isnothing(gi) && !isnothing(si)
        f = median(results[fi].t); r = median(results[ri].t)
        g = median(results[gi].t); s = median(results[si].t)
        @printf("%-6s  rollout/fwd = %.2fx   |   gradient/fwd = %.2fx   |   full train step/fwd = %.2fx\n",
                string(dev), r / f, g / f, s / f)
    end
end
println("="^92)
