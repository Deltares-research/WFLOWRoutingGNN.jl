#!/usr/bin/env julia
# scripts/check_mass_balance_details.jl
#
# Verify data consistency properties in wflow output.nc:
#
#  1. river_flow_q_av          ≈ river_q
#  2. river_flow_h             ≈ river_h
#  3. river_bc_inwater         ≈ river_inwater
#  4. river_bc_external_inflow ≈ 0
#  5. river_bc_abstraction     ≈ 0
#  6. river_bc_external_abst_av≈ 0
#  7. water_balance_storage_prev[t] ≈ river_flow_storage[t-1]
#  8. river_flow_qin_av[t]     ≈ Σ river_flow_q_av[t-1] over inflowing cells
#  9. river_flow_qin_av[t]     ≈ Σ river_flow_q_av[t]   (same timestep)
# 12. river_flow_storage       ≈ river_flow_h × river_width × river_length
#
# The checks live in `src/consistency.jl` (`river_layout`, `graph_topology_checks`,
# `river_mask_consistency`, `wflow_output_consistency`); this script only prints them.
#
# Usage:
#   julia --project=<repo_root> scripts/check_mass_balance_details.jl \
#       <staticmaps.nc> <output.nc>

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using Statistics
using Printf
using WflowRoutingGNN

# Print residual statistics for one check.
function print_check(c)
    prefix = @sprintf("  [%d] %s", c.id, c.description)
    if isnothing(c.residual)
        @printf("%-70s  SKIP (variable(s) missing)\n", prefix)
        return
    end
    s = residual_summary(c.residual; ref = c.ref)
    if isnothing(s)
        @printf("%-70s  SKIP (no finite values)\n", prefix)
        return
    end
    ref_str = isnothing(s.median_abs_ref) ? "" : @sprintf("  (median|ref|=%.3g)", s.median_abs_ref)
    @printf("%-70s  MAE=%.3g  RMSE=%.3g  max=%.3g  exact(≤1e-6)=%.1f%%%s\n",
            prefix, s.mae, s.rmse, s.max_abs, 100 * s.frac_exact, ref_str)
end

passfail(ok, fail = "✗ FAIL") = ok ? "✓ PASS" : fail

function main(argv)
    if length(argv) < 2
        println(stderr,
            "Usage: julia --project=<repo_root> scripts/check_mass_balance_details.jl " *
            "<staticmaps.nc> <output.nc>")
        exit(1)
    end
    staticmaps_file, output_file = argv[1], argv[2]
    isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
    isfile(output_file)     || (println(stderr, "Not found: $output_file");     exit(1))

    # ── 1. Compact river graph ───────────────────────────────────────────────
    println("Reading staticmaps: $staticmaps_file")
    layout = river_layout(staticmaps_file, output_file)
    println("  River cells : $(layout.n_nodes)")
    println("  River edges : $(length(layout.edge_src))")
    println("  Axis flip   : dim1=$(layout.dim1_flip), dim2=$(layout.dim2_flip)")

    # ── 2. Output variables ──────────────────────────────────────────────────
    println("\nReading output: $output_file")
    vars = NCDataset(output_file, "r") do ds
        println("  Variable presence:")
        for vn in WFLOW_CONSISTENCY_VARS
            status = vn in keys(ds) ? "✓ found" : "✗ NOT FOUND"
            @printf("    %-40s %s\n", vn, status)
        end
        Dict(vn => read_river_series(ds, layout, vn) for vn in WFLOW_CONSISTENCY_VARS)
    end

    # ── 3. Graph topology and mask consistency ───────────────────────────────
    topo = graph_topology_checks(layout.edge_src, layout.edge_tgt, layout.n_nodes)
    println("\nGraph topology check:")
    @printf("  nodes=%d  edges=%d  headwaters=%d  outlets=%d  isolated=%d\n",
            topo.n_nodes, topo.n_edges, length(topo.headwaters), length(topo.outlets),
            length(topo.isolated))
    @printf("  Headwater upstream_sum == 0 : %s\n", passfail(topo.headwater_ok))
    if topo.n_single_upstream > 0
        @printf("  Single-upstream nodes=%d  upstream_sum==q[upstream] : %s\n",
                topo.n_single_upstream,
                passfail(topo.single_upstream_mismatches == 0,
                         "✗ FAIL ($(topo.single_upstream_mismatches) mismatches)"))
    end

    println("\nRiver mask consistency check:")
    mc = river_mask_consistency(staticmaps_file, layout)
    @printf("  river_mask non-zero cells : %d  (graph nodes: %d)  %s\n",
            mc.mask_count, mc.n_nodes, mc.mask_count == mc.n_nodes ? "✓ match" : "✗ MISMATCH")
    @printf("  LDD → river cell (should be an edge) : %d\n", mc.ldd_to_river)
    @printf("  LDD → non-river cell (no edge added)  : %d\n", mc.ldd_to_nonriver)
    @printf("  LDD == 5 (pit/outlet)                 : %d\n", mc.ldd_is_pit)
    @printf("  LDD missing/unknown                   : %d\n", mc.ldd_missing)
    @printf("  sum                                   : %d  (should == %d)\n",
            mc.ldd_to_river + mc.ldd_to_nonriver + mc.ldd_is_pit + mc.ldd_missing, mc.n_nodes)
    @printf("  graph edges vs. LDD→river count       : %d vs %d  %s\n",
            mc.n_edges, mc.ldd_to_river, mc.n_edges == mc.ldd_to_river ? "✓ match" : "✗ MISMATCH")
    @printf("  river cells with NO adjacent river neighbour (8-conn): %d  %s\n",
            mc.not_adjacent, mc.not_adjacent == 0 ? "✓ connected" : "✗ isolated pixels present")
    @printf("  outlets whose LDD targets a river cell but have no edge: %d  %s\n",
            length(mc.missed_edges),
            isempty(mc.missed_edges) ? "✓ none" : "✗ BUG: edge was missed")
    if !isempty(mc.missed_edges)
        i = first(mc.missed_edges)
        @printf("    example: node %d at (%d,%d)\n", i, layout.rows[i], layout.cols[i])
    end

    # ── 4. Checks ────────────────────────────────────────────────────────────
    println("\n" * "─"^110)
    println("Check results:")
    println("─"^110)
    foreach(print_check, wflow_output_consistency(vars, layout))
    println("─"^110)
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
