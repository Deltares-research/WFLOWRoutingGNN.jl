"""
typecheck_forward.jl  (temporary performance-diagnostic script)

Confirm/deny the hypothesis that the abstract adjacency fields
(`SparseConv.A::AbstractMatrix`, `MassBalanceLayer.A_routing::AbstractMatrix`)
introduce type instability / dynamic dispatch in the mission-critical forward
path.

Diagnoses on CPU (type instability is a property of the *declared field type*,
device-independent). Prints:
  * the concrete field types actually stored,
  * `@code_warntype` of each hot layer forward and the top-level model call,
  * BenchmarkTools allocation/time for the processor layer forward (a size-
    independent allocation signature is the fingerprint of a runtime dispatch).

Usage:
    julia --project=. scripts/typecheck_forward.jl <experiment_dir>
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_wflow_graph, build_gnn_model,
                      load_schema
using Flux
using JLD2
using InteractiveUtils
using BenchmarkTools

EXP_DIR = isempty(ARGS) ? error("usage: typecheck_forward.jl <experiment_dir>") : ARGS[1]

config_path = joinpath(EXP_DIR, "config.toml")
model_path  = joinpath(EXP_DIR, "model", "model.jld2")
isfile(model_path) || (model_path = joinpath(EXP_DIR, "model.jld2"))

ds, ms, ts = parse_run_config(config_path)
staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")
schema = load_schema(ds.wflow_schema)

graphs, norm_stats, _grid, postscale, static_arr =
    build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)

model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file, 1;
                        h_loss_scale = ts.h_loss_scale)
Flux.loadmodel!(model, JLD2.load(model_path, "model_state"))

# CPU inputs for a single graph.
g       = graphs[1]
state   = g.ndata.state
forcing = g.ndata.forcing
static  = Array{Float32}(static_arr)
fnext   = forcing
h       = randn(Float32, ms.hidden_dim, g.num_nodes)

sc = model.processor.layers[1]           # first SparseConv

println("="^72)
println("CONCRETE FIELD TYPES (should be concrete, not AbstractMatrix)")
println("="^72)
println("typeof(model)                       = ", typeof(model))
println("typeof(SparseConv)                  = ", typeof(sc))
println("typeof(sc.A)                        = ", typeof(sc.A))
println("typeof(sc.A_batched)                = ", typeof(sc.A_batched))
println("fieldtype(SparseConv, :A)           = ", fieldtype(typeof(sc), :A))
if model.mass_balance !== nothing
    mb = model.mass_balance
    println("typeof(mb.A_routing)                = ", typeof(mb.A_routing))
    println("fieldtype(MassBalanceLayer,:A_routing) = ", fieldtype(typeof(mb), :A_routing))
end
println("isconcretetype(typeof(model))       = ", isconcretetype(typeof(model)))
println("isconcretetype(typeof(sc))          = ", isconcretetype(typeof(sc)))
model.mass_balance !== nothing &&
    println("isconcretetype(typeof(mb))          = ", isconcretetype(typeof(model.mass_balance)))

println()
println("="^72)
println("@code_warntype  SparseConv forward  (sc(g, h))")
println("="^72)
@code_warntype sc(g, h)

if model.mass_balance !== nothing
    println()
    println("="^72)
    println("@code_warntype  MassBalanceLayer forward")
    println("="^72)
    @code_warntype model.mass_balance(g, state, forcing, fnext, state[1:1, :])
end

println()
println("="^72)
println("@code_warntype  top-level model forward")
println("="^72)
@code_warntype model(g, state, forcing, static, fnext)

println()
println("="^72)
println("BenchmarkTools: processor-layer forward (alloc = dispatch fingerprint)")
println("="^72)
sc(g, h)  # warm
b = @benchmark $sc($g, $h)
println("min time   = ", minimum(b).time / 1e3, " µs")
println("median     = ", median(b).time / 1e3, " µs")
println("allocs     = ", minimum(b).allocs)
println("alloc mem  = ", minimum(b).memory, " bytes")
