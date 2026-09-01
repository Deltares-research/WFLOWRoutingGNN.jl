"""
regen_mb_diagnostics.jl

Regenerate `mb_diagnostics.png` for an already-trained run WITHOUT retraining.

Loads the trained model + config, rebuilds the graph time series, restores the
validation split from `data/val.jld2`, recomputes the mass-balance diagnostics,
and re-plots. Use this to refresh the diagnostic after a fix to
`rollout_mb_diagnostics` / `plot_mb_diagnostics`.

Usage:
    julia --project=. scripts/regen_mb_diagnostics.jl <experiment_dir> [out.png]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using WflowRoutingGNN: parse_run_config, build_wflow_graph, build_gnn_model,
                      load_schema, rollout_mb_diagnostics, plot_mb_diagnostics
using Flux
using JLD2
using NCDatasets
using Dates

length(ARGS) >= 1 || error("usage: regen_mb_diagnostics.jl <experiment_dir> [out.png]")
const EXP_DIR = ARGS[1]
const OUT_PNG = length(ARGS) >= 2 ? ARGS[2] : joinpath(EXP_DIR, "plots", "mb_diagnostics.png")
const OUT_CSV = joinpath(EXP_DIR, "metrics", "mb_diagnostics.csv")

config_path = joinpath(EXP_DIR, "config.toml")
model_path  = joinpath(EXP_DIR, "model", "model.jld2")
val_path    = joinpath(EXP_DIR, "output", "data", "val.jld2")
isfile(config_path) || error("config.toml not found in $EXP_DIR")
isfile(model_path)  || error("model.jld2 not found in $EXP_DIR")
isfile(val_path)    || error("output/data/val.jld2 not found in $EXP_DIR")

@info "Loading config from $config_path"
ds, ms, ts = parse_run_config(config_path)

staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")
isfile(staticmaps_file) || error("staticmaps not found: $staticmaps_file")
isfile(output_file)     || error("output.nc not found: $output_file")

schema = load_schema(ds.wflow_schema)

@info "Building graph time series (domain=$(ms.domain)) …"
graphs, norm_stats, _grid, postscale, static_arr =
    build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)

model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file, 1;
                        h_loss_scale = ts.h_loss_scale)
Flux.loadmodel!(model, JLD2.load(model_path, "model_state"))
@info "Restored trained weights from $model_path"

split_data = JLD2.load(val_path, "data")
@info "Loaded validation split: $(length(split_data)) windows"

@info "Computing mass balance diagnostics …"
mb_diags = rollout_mb_diagnostics(model, split_data, static_arr)

plot_mb_diagnostics(mb_diags; path = OUT_PNG, csv_path = OUT_CSV)
@info "Wrote $OUT_PNG and $OUT_CSV"
