"""
profile_rollout.jl

Decompose the cost of a single autoregressive rollout step and compare it to
the "single message pass" figure from `benchmark_message_passing.jl`.

The goal is to explain the ~10x gap between "<1 ms per message pass" and the
~20-30 ms/step attributed to the rollout. It:

  1. Builds the real river model (SparseConv processor + hard MassBalance),
     exactly as training does, on a ~8k-node graph.
  2. Times each sub-component of one step (encoder, each processor layer,
     decoder, mass balance, and the host-side vcat/slice/store), with a warm-up
     pass and — on GPU — a `CUDA.synchronize()` after every timed call so the
     numbers reflect real device work rather than async kernel launches.
  3. Reproduces the number the `rollout` function itself prints, and separates
     the first (compilation) step from the steady-state steps, showing how the
     reported mean is inflated.

Usage:
    julia --project=. scripts/profile_rollout.jl [staticmaps.nc output.nc] [cpu|gpu] [T]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: build_wflow_graph, build_gnn_model, ModelSettings, rollout, load_schema
using GraphNeuralNetworks
using Flux
using CUDA
using Statistics
using Printf

# ── CLI ───────────────────────────────────────────────────────────────────────
const STATICMAPS = get(ARGS, 1,
    joinpath(@__DIR__, "..", "models", "sava_v081", "staticmaps.nc"))
const OUTPUT_NC  = get(ARGS, 2,
    joinpath(@__DIR__, "..", "models", "sava_v081", "run_historical", "output.nc"))
const DEV_ARG    = lowercase(get(ARGS, 3, ""))
const T_STEPS    = parse(Int, get(ARGS, 4, "200"))
const SCHEMA_ARG = get(ARGS, 5, "v0.8.1")   # v081 staticmaps use `wflow_ldd` etc.

use_gpu = DEV_ARG == "gpu" || (DEV_ARG == "" && CUDA.functional())
dev_fn  = use_gpu ? Flux.gpu : Flux.cpu
@info "Device: $(use_gpu ? "GPU" : "CPU")   T=$T_STEPS steps"

# Synchronise the device (no-op on CPU) so timers capture completed work.
sync!() = use_gpu ? CUDA.synchronize() : nothing

# Warm-up once (compile), then take the median wall-time over `reps` reps,
# synchronising after each call on GPU. `min` is also reported (least noisy).
function timeit(f; reps::Int = 50)
    f(); f(); sync!()                    # two warm-ups: compile all branches
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f()
        sync!()
        ts[i] = (time_ns() - t0) / 1e6   # ms
    end
    (median = median(ts), min = minimum(ts))
end

# ── 1. Build graph + model (mirrors training / rollout) ───────────────────────
@info "Building wflow graph …"
schema = load_schema(SCHEMA_ARG)
graphs, norm_stats, _grid, postscale, static_arr =
    build_wflow_graph(STATICMAPS, OUTPUT_NC, "river"; schema)

ms    = ModelSettings(domain = "river")                 # hidden=64, nlayers=3, mlp=1
model = build_gnn_model(ms, graphs, norm_stats, postscale, OUTPUT_NC, 1)

g0      = graphs[1]
n_nodes = g0.num_nodes
T_max   = min(T_STEPS, length(graphs) - 1)

# Forcing array (n_forcing, n_nodes, T) from consecutive graphs.
n_forcing = size(g0.ndata.forcing, 1)
forcing = Array{Float32}(undef, n_forcing, n_nodes, T_max)
for t in 1:T_max
    forcing[:, :, t] = graphs[t].ndata.forcing
end

@info @sprintf("Graph: %d nodes  %d edges  |  processor: %d SparseConv layers  hidden=%d",
               n_nodes, g0.num_edges, ms.nlayers, ms.hidden_dim)

# ── 2. Move to device ─────────────────────────────────────────────────────────
model_d   = dev_fn(model)
g0_d      = dev_fn(g0)
static_d  = dev_fn(Array{Float32}(static_arr))
forcing_d = dev_fn(forcing)
state0    = g0_d.ndata.state

# Processor layers (individually callable) and encoder/decoder/mass balance.
enc   = model_d.encoder
procs = model_d.processor.layers
dec   = model_d.decoder
mb    = model_d.mass_balance

# ── 3. Time each sub-component of ONE step ────────────────────────────────────
state       = state0
forcing_t   = forcing_d[:, :, 1]
forcing_nt  = forcing_d[:, :, min(2, T_max)]

# Precompute intermediate activations so each layer is timed on correct input.
x0  = vcat(state, forcing_t, static_d)
h0  = enc(x0)
h1  = procs[1](g0_d, h0)
h2  = procs[2](g0_d, h1)
h3  = procs[3](g0_d, h2)
Δ0  = dec(h3)
q0  = state[1:1, :] .+ Δ0

comp = Pair{String,Any}[
    "host: slice forcing_d[:,:,t]"      => () -> forcing_d[:, :, 1],
    "host: vcat(state,forcing,static)"  => () -> vcat(state, forcing_t, static_d),
    "encoder (Dense in→64)"             => () -> enc(x0),
    "processor layer 1 (SparseConv)"    => () -> procs[1](g0_d, h0),
    "processor layer 2 (SparseConv)"    => () -> procs[2](g0_d, h1),
    "processor layer 3 (SparseConv)"    => () -> procs[3](g0_d, h2),
    "decoder (Dense 64→1)"              => () -> dec(h3),
    "mass_balance layer"                => () -> mb(g0_d, state, forcing_t, forcing_nt, q0),
    "full model step  m(g,…)"           => () -> model_d(g0_d, state, forcing_t, static_d, forcing_nt),
]

println("\n" * "="^72)
println(@sprintf("PER-COMPONENT TIMING  (%s, %d nodes)", use_gpu ? "GPU" : "CPU", n_nodes))
println("="^72)
@printf("%-38s %12s %12s\n", "component", "median (ms)", "min (ms)")
println("-"^72)
sum_parts = 0.0
for (name, f) in comp
    r = timeit(f)
    startswith(name, "full model") || (global sum_parts += r.median)
    @printf("%-38s %12.4f %12.4f\n", name, r.median, r.min)
end
println("-"^72)
@printf("%-38s %12.4f\n", "Σ components (excl. full step)", sum_parts)

# ── 4. Reproduce the rollout's own per-step number ────────────────────────────
# The rollout loop times each step with `time()` and reports the MEAN, with no
# device sync and including the first (compilation) step. Replicate that exactly
# to show the artifact, then contrast with a warm, synced steady-state median.
println("\n" * "="^72)
println("ROLLOUT-LOOP MEASUREMENT (as the rollout function reports it)")
println("="^72)

# 4a. Cold loop exactly like rollout(): no sync, mean over all steps incl. step 1.
states_d = similar(state0, size(state0, 1), n_nodes, T_max)
raw_times = Vector{Float64}(undef, T_max)
st = state0
for t in 1:T_max
    t0 = time()
    fnt = forcing_d[:, :, min(t + 1, T_max)]
    global st = model_d(g0_d, st, forcing_d[:, :, t], static_d, fnt)
    states_d[:, :, t] = st
    raw_times[t] = time() - t0
end
sync!()
@printf("cold loop (no sync):   step 1 = %.3f ms   mean(all) = %.3f ms   median(2:end) = %.3f ms\n",
        raw_times[1] * 1e3, mean(raw_times) * 1e3, median(raw_times[2:end]) * 1e3)

# 4b. Warm loop WITH per-step sync: the honest steady-state cost.
st = state0
warm = Vector{Float64}(undef, T_max)
for t in 1:T_max
    t0 = time_ns()
    fnt = forcing_d[:, :, min(t + 1, T_max)]
    global st = model_d(g0_d, st, forcing_d[:, :, t], static_d, fnt)
    states_d[:, :, t] = st
    sync!()
    warm[t] = (time_ns() - t0) / 1e6
end
@printf("warm loop (synced):    median = %.3f ms   min = %.3f ms   max = %.3f ms\n",
        median(warm), minimum(warm), maximum(warm))

println("\nInterpretation:")
println(" • The rollout does 3 message passes + encoder + decoder + mass balance,")
println("   vs. the benchmark's single message pass — expect ≳3-5× on compute alone.")
println(" • The cold-loop MEAN folds in the first step's JIT compilation.")
println(" • Without CUDA.@sync the per-step time() deltas are meaningless on GPU;")
println("   the warm+synced median is the honest steady-state cost.")
