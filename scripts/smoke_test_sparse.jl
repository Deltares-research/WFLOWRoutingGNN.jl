using WflowRoutingGNN, SparseArrays, Flux, GraphNeuralNetworks, CUDA

root       = joinpath(@__DIR__, "..")
staticmaps = joinpath(root, "wflow_model", "wflow_test_full", "staticmaps.nc")
output     = joinpath(root, "wflow_model", "wflow_test_full", "run_default", "output.nc")
isfile(staticmaps) || (staticmaps = joinpath(root, "models", "sava_small", "staticmaps.nc"))
isfile(output)     || (output     = joinpath(root, "models", "sava_small", "run_default", "output.nc"))

graphs, norm_stats, grid, postscale = build_wflow_graph(staticmaps, output, "river")
g0 = graphs[1]
n  = g0.num_nodes
s, t = edge_index(g0)
A = sparse(vcat(t, collect(1:n)), vcat(s, collect(1:n)),
           ones(Float32, length(s) + n), n, n)

ms = ModelSettings(domain = "river")
dt = get_timestep(output)
mb = MassBalanceLayer(
    postscale["river_q"], postscale["river_h"],
    Float32(norm_stats["river_q"].mean),      Float32(norm_stats["river_q"].std),
    Float32(norm_stats["river_h"].mean),      Float32(norm_stats["river_h"].std),
    Float32(norm_stats["river_inwater"].mean), Float32(norm_stats["river_inwater"].std),
    Float32(dt),
)

model = WflowGNN(ms, mb, A)

B = 8
model_pre = precompute_batched(model, B)
println("CPU A_batched type:  ", typeof(model_pre.processor.layers[1].A_batched))
println("CPU A_batched size:  ", size(model_pre.processor.layers[1].A_batched),
        "  expected ($(B*n), $(B*n))")

n_state   = size(g0.ndata.state,   1)
n_forcing = size(g0.ndata.forcing, 1)
n_static  = size(g0.ndata.static,  1)

if CUDA.functional()
    model_gpu = Flux.gpu(model_pre)
    println("GPU A_batched type:  ", typeof(model_gpu.processor.layers[1].A_batched))
    model_cpu = Flux.cpu(model_gpu)
    println("RT  A_batched type:  ", typeof(model_cpu.processor.layers[1].A_batched))

    # --- Batched processor path (training path) ---
    using MLUtils
    gb  = MLUtils.batch(fill(g0 |> Flux.gpu, B))
    Xb  = randn(Float32, n_state + n_forcing + n_static, n * B) |> Flux.gpu
    h   = model_gpu.encoder(Xb)
    out = model_gpu.processor(gb, h)
    println("Batched GPU forward: ", size(out), "  expected ($(ms.hidden_dim), $(n*B))")

    # --- Single-graph path (rollout path) ---
    # Mirrors what rollout() does: model_d(g0_d, state, forcing, static, forcing_next)
    g0_gpu      = g0 |> Flux.gpu
    state_gpu   = g0_gpu.ndata.state
    forcing_gpu = g0_gpu.ndata.forcing
    static_gpu  = g0_gpu.ndata.static
    out_single  = model_gpu(g0_gpu, state_gpu, forcing_gpu, static_gpu, forcing_gpu)
    println("Single-graph GPU forward: ", size(out_single), "  expected ($(n_state), $n)")

    # --- Full rollout on GPU ---
    # Build a minimal forcing array (2 steps) from the first two graphs
    forcing_arr = Array{Float32}(undef, n_forcing, n, 2)
    forcing_arr[:, :, 1] = graphs[1].ndata.forcing
    forcing_arr[:, :, 2] = graphs[2].ndata.forcing
    pred = rollout(model_pre, g0, forcing_arr; device = :gpu, timesteps = 2)
    println("rollout GPU output:  ", size(pred), "  expected ($(n_state), $n, 2)")
else
    println("CUDA not functional — skipping GPU checks")
end

println("Done.")
