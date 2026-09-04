"""
diag_overhead.jl  (temporary performance-diagnostic script)

Quantify the per-batch cost of the training-loop *diagnostics* that run on every
batch in `train_model!` in addition to the real optimiser step:

    one_step_loss                       (unconditional, training.jl)
    loss_components + mb_amplification   (when the model has a mass balance)

Each is an extra forward pass whose result is only logged (epoch-granularity
metrics). This times the real train step vs. those diagnostics on the SAME
model/batch/device, with warm-up and `CUDA.synchronize()` around GPU calls, and
reports the fraction of each batch spent on logging-only work.

Usage:
    julia --project=. scripts/diag_overhead.jl <experiment_dir> [--devices cpu,gpu]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_wflow_graph, build_gnn_model,
                      load_schema, loss_function, one_step_loss, loss_components,
                      mb_amplification, TrainingStrategy
using Flux
using CUDA
using JLD2
using Statistics
using Printf
using Dates
import TOML

function parse_args(args)
    isempty(args) && error("usage: diag_overhead.jl <experiment_dir> [--devices cpu,gpu]")
    exp_dir = args[1]
    opts = Dict{String,String}()
    i = 2
    while i <= length(args)
        a = args[i]; startswith(a, "--") || error("unexpected arg: $a")
        opts[a[3:end]] = args[i+1]; i += 2
    end
    return exp_dir, opts
end

EXP_DIR, OPTS = parse_args(ARGS)

config_path = joinpath(EXP_DIR, "config.toml")
model_path  = joinpath(EXP_DIR, "model", "model.jld2")
isfile(model_path) || (model_path = joinpath(EXP_DIR, "model.jld2"))

ds, ms, ts = parse_run_config(config_path)
staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")
schema = load_schema(ds.wflow_schema)

graphs, norm_stats, _grid, postscale, static_arr =
    build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)

strategy = ts.strategy
nsteps   = strategy.current_steps
hw       = strategy.h_loss_weight

model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file, ts.batch_size;
                        strategy = strategy, h_loss_scale = ts.h_loss_scale)
Flux.loadmodel!(model, JLD2.load(model_path, "model_state"))

# One representative collated batch: batch_size windows of nsteps+1 graphs.
# Use consecutive graphs, batched the same way the training DataLoader collates.
using GraphNeuralNetworks: GNNGraphs
B = ts.batch_size
windows = [[graphs[s + k] for k in 0:nsteps] for s in 1:B]
# Collate: batch across windows → vector of nsteps+1 batched graphs.
batch_cpu = [GNNGraphs.batch([windows[w][t] for w in 1:B]) for t in 1:(nsteps + 1)]

requested = haskey(OPTS, "devices") ? Symbol.(strip.(split(OPTS["devices"], ","))) :
            (CUDA.functional() ? [:cpu, :gpu] : [:cpu])

# Simple median-of-reps timer (ms), sync after each call.
function timeit(f, sync; warmup = 3, reps = 30)
    for _ in 1:warmup; f(); sync(); end
    t = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns(); f(); sync(); t[r] = (time_ns() - t0) / 1e6
    end
    return median(t)
end

results = NamedTuple[]
for dev in requested
    (dev == :gpu && !CUDA.functional()) && (@warn "GPU not functional; skipping"; continue)
    dev_fn = dev == :gpu ? Flux.gpu : identity
    sync   = dev == :gpu ? (() -> CUDA.synchronize()) : (() -> nothing)

    model_d  = dev_fn(model)
    static_d = dev_fn(Array{Float32}(static_arr))
    batch_d  = dev_fn.(batch_cpu)

    rule = ts.grad_clip > 0 ?
        Flux.Optimisers.OptimiserChain(
            Flux.Optimisers.ClipNorm(Float32(ts.grad_clip); throw = false), Adam(ts.lr_start)) :
        Adam(ts.lr_start)
    opt_state = Flux.setup(rule, model_d)

    function train_step()
        _, gs = Flux.withgradient(m -> loss_function(m, batch_d, strategy, static_d), model_d)
        Flux.update!(opt_state, model_d, gs[1])
    end

    t_train = timeit(train_step, sync)
    t_1step = timeit(() -> one_step_loss(model_d, batch_d, static_d, hw), sync)
    has_mb  = model_d.mass_balance !== nothing
    t_comp  = has_mb ? timeit(() -> loss_components(model_d, batch_d, static_d), sync) : 0.0
    t_amp   = has_mb ? timeit(() -> mb_amplification(model_d, batch_d, static_d), sync) : 0.0

    diag = t_1step + t_comp + t_amp
    push!(results, (device = dev, t_train = t_train, t_1step = t_1step,
                    t_comp = t_comp, t_amp = t_amp, diag = diag,
                    frac = diag / (t_train + diag)))
end

# Persist to metrics/performance.toml (do not clobber other benchmark tables).
perf_out = joinpath(EXP_DIR, "metrics", "performance.toml")
root = isfile(perf_out) ? TOML.parsefile(perf_out) : Dict{String,Any}()
_r(x) = round(Float64(x); sigdigits = 6)
tbl = Dict{String,Any}("timestamp" => string(Dates.now()),
                       "batch_size" => B, "horizon_steps" => nsteps,
                       "n_nodes_per_graph" => graphs[1].num_nodes)
for r in results
    tbl["$(r.device)"] = Dict(
        "train_step_ms" => _r(r.t_train), "one_step_loss_ms" => _r(r.t_1step),
        "loss_components_ms" => _r(r.t_comp), "mb_amplification_ms" => _r(r.t_amp),
        "diagnostics_total_ms" => _r(r.diag),
        "diagnostics_fraction_of_batch" => _r(r.frac))
end
root["training_diagnostic_overhead"] = tbl
mkpath(dirname(perf_out))
open(perf_out, "w") do io; TOML.print(io, root); end

println("="^90)
println("TRAINING PER-BATCH DIAGNOSTIC OVERHEAD  |  $(EXP_DIR)  |  B=$B  horizon=$nsteps")
println("="^90)
@printf("%-6s | %10s %11s %11s %10s | %10s %8s\n",
        "device", "train_ms", "1step_ms", "comp_ms", "amp_ms", "diag_ms", "frac")
println("-"^90)
for r in results
    @printf("%-6s | %10.3f %11.3f %11.3f %10.3f | %10.3f %7.1f%%\n",
            string(r.device), r.t_train, r.t_1step, r.t_comp, r.t_amp, r.diag, 100 * r.frac)
end
println("="^90)
println("frac = diagnostics / (real train step + diagnostics): logging-only tax per batch")
