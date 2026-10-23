# Subcatchment error attribution for a trained WflowRoutingGNN run (TODO 3e).
#
# Thin CLI over `subcatch_error_attribution_run` (src/attribution.jl). Post-hoc: works on
# the gridded prediction/truth NetCDFs of an existing run, no model or retraining needed.
# It asks which subcatchments *inject* discharge error (e_local = ΔQ(outlet) − Σ ΔQ(inlets))
# rather than carry error advected in from upstream, and rank-correlates that local error
# with static attributes.
#
# Writes to the run directory:
#   metrics/<split>_subcatch_error_attribution.csv   one row per subcatchment
#   metrics/<split>_subcatch_attribute_ranking.csv   Spearman rho of attributes vs error
#   plots/<split>_subcatch_{error_map,mainstem,attributes}.png
#
# Usage:
#   julia --project=. scripts/subcatch_error_attribution.jl <run_dir> [split]
#         [--gauges VAR] [--min-area KM2] [--staticmaps PATH]
#     <run_dir>      run folder with output/<split>_pred.nc, output/<split>_true.nc, model/*.toml
#     [split]        "val" (default) or "train"
#     --gauges VAR   Route A: outlets = cells of this staticmaps gauge raster, e.g.
#                    wflow_gauges_grdc. Default: Route B, outlets derived from the LDD
#                    (river sinks + branch ends above confluences, upstream area >= --min-area).
#     --min-area KM2 minimum area of a retained subcatchment, and Route B branch threshold
#                    (default 100)
#     --staticmaps   explicit staticmaps.nc (default: resolved from the run's data settings)

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN
using Printf

function main(argv)
    pos = String[]
    gauge_var = nothing
    min_area = 100.0
    staticmaps = nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--gauges"
            gauge_var = argv[i+1]; i += 2
        elseif a == "--min-area"
            min_area = parse(Float64, argv[i+1]); i += 2
        elseif a == "--staticmaps"
            staticmaps = argv[i+1]; i += 2
        else
            push!(pos, a); i += 1
        end
    end
    length(pos) >= 1 || error("usage: julia --project=. scripts/subcatch_error_attribution.jl <run_dir> [split] [--gauges VAR] [--min-area KM2] [--staticmaps PATH]")
    run_dir = abspath(pos[1])
    split   = length(pos) >= 2 ? pos[2] : "val"

    res = subcatch_error_attribution_run(run_dir; split, gauge_var,
                                         min_area_km2 = min_area, staticmaps_file = staticmaps)

    ck = res.tree.check
    println("\n=== subcatchment error attribution ($(split), route = $(res.route)) : $run_dir ===")
    @printf("subcatchments=%d (retained >= %g km2: %d)  unassigned cells=%d/%d  area check: median %.2g, max %.2g\n",
            length(res.rows), res.min_area_km2, count(r -> r.retained, res.rows),
            ck.n_unassigned, ck.n_active, ck.area_rel_err_median, ck.area_rel_err_max)
    @printf("%-10s %9s %9s %7s %9s %9s %9s %8s\n",
            "outlet", "inc_km2", "cum_km2", "NSE", "rms_tot", "rms_adv", "rms_loc", "loc/tot")
    for r in sort(res.rows; by = r -> -r.rms_local)
        @printf("%-10d %9.0f %9.0f %7.3f %9.3g %9.3g %9.3g %8.2f%s\n",
                r.outlet_id, r.inc_area_km2, r.cum_area_km2, r.nse, r.rms_total, r.rms_adv,
                r.rms_local, r.local_over_total, r.retained ? "" : "  (not retained)")
    end
    for (t, rk) in sort(collect(res.ranking); by = first)
        println("\n--- Spearman rho of attributes vs $t (retained subcatchments) ---")
        isempty(rk) && println("(no attribute with enough variation / pairs)")
        for r in rk
            @printf("%-20s rho=%6.2f  [%5.2f, %5.2f]  n=%d  p=%.3f  p_bonf=%.3f%s\n",
                    r.attribute, r.rho, r.ci_lo, r.ci_hi, r.n, r.p, r.p_bonferroni,
                    r.low_power ? "  (low power)" : "")
        end
    end
    println("\nWrote: $(res.csv)")
    isnothing(res.ranking_csv) ? println("       (no attribute ranking written: no attribute with enough variation / pairs)") :
                                 println("       $(res.ranking_csv)")
    foreach(p -> println("       $p"), res.plots)
end

main(ARGS)
