"""
benchmark_rollout.jl

Benchmark the inference rollout of a **trained** WflowRoutingGNN model.

For a given experiment (a trained-run directory containing `config.toml` and
`model.jld2`), it loads the trained weights, builds the graph time series, selects
a prediction window `[start, end]`, and then times the autoregressive rollout in
four configurations:

    {CPU, GPU} × {single rollout, ensemble rollout}

The ensemble uses the **same initial state** for every member and the **training
batch size** as the number of ensemble members (overridable).  For each test it
reports the total rollout time, the number of predicted timesteps, and summary
statistics of the *per-timestep, per-ensemble-member* wall time.

Usage:
    julia --project=. scripts/benchmark_rollout.jl <experiment_dir> [options]

Positional:
    <experiment_dir>   Trained-run directory with config.toml and model.jld2.

Options (all optional):
    --start   DATETIME  First timestamp of the prediction window (ISO, e.g.
                        2001-01-01 or 2001-01-01T00:00:00).  Default: first step.
    --end     DATETIME  Last timestamp of the prediction window.  Default: last.
    --staticmaps PATH   Override the staticmaps.nc path (default: from config).
    --output     PATH   Override the output.nc path (default: from config).
    --members    N      Ensemble size (default: training batch_size from config).
    --timesteps  N      Cap the number of rollout steps (default: whole window).
    --devices  LIST     Comma-separated devices to test: cpu,gpu
                        (default: cpu plus gpu when CUDA is functional).

Example:
    julia --project=. scripts/benchmark_rollout.jl experiments/test_sava_v081 \
        --start 1976-01-01 --end 1976-12-31 --members 16
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_wflow_graph, build_gnn_model,
                      load_schema, rollout, rollout_ensemble
using Flux
using CUDA
using JLD2
using NCDatasets
using Statistics
using Printf
using Dates

# ── CLI parsing ───────────────────────────────────────────────────────────────
function parse_args(args)
    isempty(args) && error("usage: benchmark_rollout.jl <experiment_dir> [--start ..] [--end ..] " *
                           "[--staticmaps ..] [--output ..] [--members N] [--timesteps N] [--devices cpu,gpu]")
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

# Build the model with the same settings used in training, then restore weights.
# batch_size=1 here; the ensemble path re-precomputes the block-diagonal for B.
model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file, 1;
                        h_loss_scale = ts.h_loss_scale)
model_state = JLD2.load(model_path, "model_state")
Flux.loadmodel!(model, model_state)
@info "Restored trained weights from $model_path"

# ── Select the prediction window [start, end] ─────────────────────────────────
raw_times = NCDataset(output_file, "r") do d
    d["time"][:]
end
times = try
    DateTime.(raw_times)
catch
    raw_times
end
L = length(times)
L == length(graphs) || @warn "time count ($L) != graph count ($(length(graphs)))"

start_dt = haskey(OPTS, "start") ? DateTime(OPTS["start"]) : times[1]
end_dt   = haskey(OPTS, "end")   ? DateTime(OPTS["end"])   : times[end]

t_start_idx = findfirst(t -> t >= start_dt, times)
t_end_idx   = findlast( t -> t <= end_dt,   times)
isnothing(t_start_idx) && error("no timestep >= $start_dt in $output_file")
isnothing(t_end_idx)   && error("no timestep <= $end_dt in $output_file")
t_end_idx > t_start_idx || error("empty window: start ($start_dt) must precede end ($end_dt)")

# Number of forward steps: predict states at t_start+1 … t_end.
T_window = t_end_idx - t_start_idx
T = haskey(OPTS, "timesteps") ? min(parse(Int, OPTS["timesteps"]), T_window) : T_window

g0        = graphs[t_start_idx]
N         = g0.num_nodes
n_forcing = size(g0.ndata.forcing, 1)

# Forcing for the window: forcing[:, :, k] drives the step producing state t_start+k.
forcing = Array{Float32}(undef, n_forcing, N, T)
for k in 1:T
    forcing[:, :, k] = graphs[t_start_idx + k - 1].ndata.forcing
end

B = haskey(OPTS, "members") ? parse(Int, OPTS["members"]) : ts.batch_size
B >= 1 || error("--members must be ≥ 1")

# Ensemble forcing: same forcing replicated across all B members (same IC too).
forcing_ens = repeat(reshape(forcing, n_forcing, N, T, 1), 1, 1, 1, B)

# ── Devices to test ───────────────────────────────────────────────────────────
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

@info @sprintf("Window: %s … %s  (%d steps)  |  graph: %d nodes  |  ensemble B=%d",
               string(times[t_start_idx]), string(times[t_start_idx + T]), T, N, B)

# ── Run the four benchmarks ───────────────────────────────────────────────────
# Stats of a per-member per-step time vector (seconds → reported in ms).
function summarize(step_times_per_member)
    ms_ = step_times_per_member .* 1e3
    (n      = length(ms_),
     min    = minimum(ms_),
     median = median(ms_),
     mean   = Statistics.mean(ms_),
     std    = length(ms_) > 1 ? std(ms_) : 0.0,
     max    = maximum(ms_))
end

results = NamedTuple[]

for dev in devices
    # ---- single rollout ------------------------------------------------------
    # Small warm-up call (compiles the rollout function itself), then measure.
    rollout(model, g0, static_arr, forcing;
            device = dev, timesteps = min(T, 3))
    r1 = rollout(model, g0, static_arr, forcing;
                 device = dev, timesteps = T, return_timing = true)
    s1 = summarize(r1.step_times)              # each step already per one member
    push!(results, (device = dev, mode = :single, members = 1,
                    total = r1.total_time, nsteps = r1.n_steps, stats = s1))

    # ---- ensemble rollout ----------------------------------------------------
    rollout_ensemble(model, g0, static_arr, forcing_ens;
                     device = dev, timesteps = min(T, 3))
    r2 = rollout_ensemble(model, g0, static_arr, forcing_ens;
                          device = dev, timesteps = T, return_timing = true)
    s2 = summarize(r2.step_times ./ B)         # per-member = whole-batch step / B
    push!(results, (device = dev, mode = :ensemble, members = B,
                    total = r2.total_time, nsteps = r2.n_steps, stats = s2))
end

# ── Report ────────────────────────────────────────────────────────────────────
println()
println("="^96)
println("ROLLOUT PERFORMANCE  |  model: $(EXP_DIR)  |  $(N) nodes  |  window $(T) steps")
println("="^96)
@printf("%-6s %-9s %8s %8s | %-42s\n", "device", "mode", "members", "total_s",
        "per-timestep per-member (ms)")
@printf("%-6s %-9s %8s %8s | %8s %8s %8s %8s %8s\n", "", "", "", "",
        "min", "median", "mean", "std", "max")
println("-"^96)
for r in results
    st = r.stats
    @printf("%-6s %-9s %8d %8.3f | %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            string(r.device), string(r.mode), r.members, r.total,
            st.min, st.median, st.mean, st.std, st.max)
end
println("-"^96)

# Ensemble efficiency vs single (per-member median step time), per device.
for dev in devices
    single = findfirst(r -> r.device == dev && r.mode == :single,   results)
    ens    = findfirst(r -> r.device == dev && r.mode == :ensemble, results)
    if !isnothing(single) && !isnothing(ens)
        sp = results[single].stats.median / results[ens].stats.median
        @printf("%-6s  ensemble per-member median speedup vs single: %.2fx  (B=%d)\n",
                string(dev), sp, results[ens].members)
    end
end
println("="^96)
