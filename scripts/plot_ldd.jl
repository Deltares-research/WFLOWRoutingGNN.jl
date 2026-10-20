#!/usr/bin/env julia
# scripts/plot_ldd.jl
#
# Plot the Local Drainage Direction (LDD) value for every cell in the river_mask.
# Thin CLI over `plot_ldd` (src/plot.jl).
# LDD encoding (PCRaster / wflow):
#
#   7  8  9
#   4  5  6     5 = pit / outlet
#   1  2  3
#
# Usage:
#   julia --project=<repo_root> scripts/plot_ldd.jl \
#       <staticmaps.nc> [out.png]

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN

function main(argv)
    if isempty(argv)
        println(stderr,
            "Usage: julia --project=<repo_root> scripts/plot_ldd.jl <staticmaps.nc> [out.png]")
        exit(1)
    end
    staticmaps_file = argv[1]
    out_path        = length(argv) >= 2 ? argv[2] : "ldd_river.png"
    isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
    plot_ldd(staticmaps_file; path = out_path)
    println("Saved: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
