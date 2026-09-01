# Spatial performance evaluation for a trained WflowRoutingGNN experiment.
#
# Operates on the gridded prediction/truth NetCDFs that `run_wflow_gnn` already
# writes (`output/<split>_pred.nc` / `output/<split>_true.nc`), so it can be run
# post-hoc on any existing experiment without retraining.
#
# It produces, in the experiment directory:
#   output/<split>_spatial_metrics.nc   — per-cell RMSE, bias, overpred. freq, peak
#                                  error, normalised bias, NSE (lon×lat maps)
#   plots/<split>_spatial_metrics.png   — those maps rendered as heatmaps
#   plots/<split>_q_overpred_vs_ramp.png + metrics/<split>_q_overpred_vs_ramp.csv
#                                  Q overprediction vs the true ramp
#                                  rate g = (Qt−Qt-1)/Qt (scatter, per-class
#                                  bars, per-cell corr map, Pearson/Spearman)
#
# Usage:
#   julia --project=. scripts/spatial_eval.jl <experiment_dir> [split]
#     <experiment_dir> : folder containing output/<split>_pred.nc /
#                        output/<split>_true.nc and model/*.toml
#     [split]          : "val" (default) or "train"

using WflowRoutingGNN
using NCDatasets
using Printf

# Locate `staticmaps.nc`. Experiments copied from another machine (e.g. an HPC
# via scp) keep the original absolute `wflow_model_path`, which won't resolve
# locally, so fall back to `<repo>/models/<model_name>/staticmaps.nc`.
function resolve_staticmaps(wflow_model_path::AbstractString, exp_dir::AbstractString)
    direct = joinpath(wflow_model_path, "staticmaps.nc")
    isfile(direct) && return direct
    model_name = basename(rstrip(wflow_model_path, ['/', '\\']))
    repo_root  = dirname(dirname(exp_dir))          # <repo>/experiments/<exp> -> <repo>
    local_path = joinpath(repo_root, "models", model_name, "staticmaps.nc")
    isfile(local_path) && return local_path
    error("could not locate staticmaps.nc (tried \"$direct\" and \"$local_path\")")
end

function load_regrid_nc(path::AbstractString, state_vars::Vector{String})
    grids = Dict{String, Array{Float32,3}}()
    NCDataset(path, "r") do ds
        for v in state_vars
            haskey(ds, v) || continue
            raw = ds[v][:, :, :]                       # (lon, lat, time), may hold `missing`
            grids[v] = Float32.(coalesce.(raw, NaN32))
        end
    end
    return grids
end

function main()
    length(ARGS) >= 1 || error("usage: julia --project=. scripts/spatial_eval.jl <experiment_dir> [split]")
    exp_dir = abspath(ARGS[1])
    split   = length(ARGS) >= 2 ? ARGS[2] : "val"
    isdir(exp_dir) || error("not a directory: $exp_dir")

    ms = load_model_settings(joinpath(exp_dir, "model", "model_settings.toml"))
    ds = load_data_settings(joinpath(exp_dir, "model", "data_settings.toml"))
    schema = load_schema(ds.wflow_schema)
    staticmaps_file = resolve_staticmaps(ds.wflow_model_path, exp_dir)

    plots_dir   = joinpath(exp_dir, "plots")
    metrics_dir = joinpath(exp_dir, "metrics")
    output_dir  = joinpath(exp_dir, "output")
    mkpath(plots_dir)
    mkpath(metrics_dir)
    mkpath(output_dir)

    state_vars = DOMAIN_VARS[ms.domain]["state"]
    pred_path  = joinpath(output_dir, "$(split)_pred.nc")
    true_path  = joinpath(output_dir, "$(split)_true.nc")
    isfile(pred_path) || error("missing $pred_path")
    isfile(true_path) || error("missing $true_path")

    @info "Loading $(split) grids from $exp_dir (domain=$(ms.domain))"
    p_grids = load_regrid_nc(pred_path, state_vars)
    t_grids = load_regrid_nc(true_path, state_vars)

    # ── Per-cell spatial error maps ───────────────────────────────────────────
    @info "Computing per-cell spatial error metrics"
    metrics = spatial_error_metrics(p_grids, t_grids, ms.domain)

    nc_out = joinpath(output_dir, "$(split)_spatial_metrics.nc")
    write_spatial_metrics_to_netcdf(metrics, staticmaps_file, nc_out; schema)
    @info "Wrote $nc_out"

    csv_out = joinpath(metrics_dir, "$(split)_spatial_metrics.csv")
    write_spatial_metrics_to_csv(metrics, csv_out)
    @info "Wrote $csv_out"

    png_out = joinpath(plots_dir, "$(split)_spatial_metrics.png")
    plot_spatial_metrics(metrics, ms.domain; path = png_out)
    @info "Wrote $png_out"

    # Console summary of median metric over active cells.
    println("\n=== per-cell metric medians ($(split)) : $exp_dir ===")
    for v in state_vars
        haskey(metrics, v) || continue
        @printf("%-10s | ", v)
        for m in SPATIAL_METRIC_NAMES
            vals = filter(isfinite, vec(metrics[v][m]))
            med  = isempty(vals) ? NaN32 : sort(vals)[cld(length(vals), 2)]
            @printf("%s=%.4g  ", m, med)
        end
        println()
    end

    # ── Q overprediction vs ramp rate ─────────────────────────────────────────
    if "river_q" in state_vars
        @info "Analysing Q overprediction vs ramp rate"
        ramp = overprediction_vs_ramp(p_grids, t_grids)
        rc_png = joinpath(plots_dir, "$(split)_q_overpred_vs_ramp.png")
        rc_csv = joinpath(metrics_dir, "$(split)_q_overpred_vs_ramp.csv")
        plot_overprediction_vs_ramp(ramp; path = rc_png, csv_path = rc_csv)
        @info "Wrote $rc_png and $rc_csv"

        println("\n=== Q overprediction vs ramp g=(Qt-Qt-1)/Qt ($(split)) ===")
        @printf("pairs=%d  Pearson(e,g)=%.3f  Pearson(max(e,0),g)=%.3f  Spearman(e,g)=%.3f\n",
                ramp.n, ramp.pearson_e_g, ramp.pearson_op_g, ramp.spearman_e_g)
        @printf("%-22s %10s %12s %14s %10s\n",
                "ramp class", "n", "mean_err", "mean_overpred", "frac_over")
        for b in 1:length(ramp.bin_labels)
            @printf("%-22s %10d %12.4g %14.4g %10.3f\n",
                    ramp.bin_labels[b], ramp.bin_n[b], ramp.bin_mean_err[b],
                    ramp.bin_mean_overpred[b], ramp.bin_frac_over[b])
        end
    end

    @info "Done."
end

main()
