# Spatial performance evaluation for a trained WflowRoutingGNN run.
#
# Thin CLI over `spatial_eval_run` (src/analysis.jl). Operates on the gridded
# prediction/truth NetCDFs written by the inference stage
# (`output/<split>_pred.nc` / `output/<split>_true.nc`), so it can be run post-hoc
# on any existing run without retraining, including runs from before the staged
# pipeline.
#
# It produces, in the run directory:
#   output/<split>_spatial_metrics.nc   per-cell RMSE, bias, overpred. freq, peak
#                                       error, normalised bias, NSE (lon x lat maps)
#   plots/<split>_spatial_metrics.png   those maps rendered as heatmaps
#   plots/<split>_q_overpred_vs_ramp.png + metrics/<split>_q_overpred_vs_ramp.csv
#                                       Q overprediction vs the true ramp
#                                       rate g = (Qt-Qt-1)/Qt (scatter, per-class
#                                       bars, per-cell corr map, Pearson/Spearman)
#
# Usage:
#   julia --project=. scripts/spatial_eval.jl <run_dir> [split]
#     <run_dir> : folder containing output/<split>_pred.nc, output/<split>_true.nc
#                 and model/*.toml
#     [split]   : "val" (default) or "train"

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using Printf

function main(argv)
    length(argv) >= 1 || error("usage: julia --project=. scripts/spatial_eval.jl <run_dir> [split]")
    run_dir = abspath(argv[1])
    split   = length(argv) >= 2 ? argv[2] : "val"
    res     = spatial_eval_run(run_dir; split)

    ms         = WflowRoutingGNN.load_model_settings(joinpath(run_dir, "model", "model_settings.toml"))
    state_vars = DOMAIN_VARS[ms.domain]["state"]

    println("\n=== per-cell metric medians ($(split)) : $run_dir ===")
    for v in state_vars
        haskey(res.metrics, v) || continue
        @printf("%-10s | ", v)
        for m in SPATIAL_METRIC_NAMES
            vals = filter(isfinite, vec(res.metrics[v][m]))
            med  = isempty(vals) ? NaN32 : sort(vals)[cld(length(vals), 2)]
            @printf("%s=%.4g  ", m, med)
        end
        println()
    end

    ramp = res.ramp
    if ramp !== nothing
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

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
