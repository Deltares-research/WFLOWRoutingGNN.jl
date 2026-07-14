#!/usr/bin/env julia
# scripts/plot_ldd.jl
#
# Plot the Local Drainage Direction (LDD) value for every cell in the river_mask.
# LDD encoding (PCRaster / wflow):
#
#   7  8  9
#   4  5  6     5 = pit / outlet
#   1  2  3
#
# Usage:
#   julia --project=<repo_root> scripts/plot_ldd.jl \
#       <staticmaps.nc> [out.png]

if isempty(ARGS)
    println(stderr,
        "Usage: julia --project=<repo_root> scripts/plot_ldd.jl " *
        "<staticmaps.nc> [out.png]")
    exit(1)
end

staticmaps_file = ARGS[1]
out_path        = length(ARGS) >= 2 ? ARGS[2] : "ldd_river.png"

isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using CairoMakie
using Printf

println("Reading: $staticmaps_file")

ldd_grid, mask_grid, lats, lons = NCDataset(staticmaps_file, "r") do ds
    ldd  = ds["local_drain_direction"][:, :]      # (row, col)
    mask = ds["river_mask"][:, :]
    # Try common coordinate dimension names
    lat_key = findfirst(k -> lowercase(k) in ("lat", "latitude", "y"), dimnames(ds["river_mask"]))
    lon_key = findfirst(k -> lowercase(k) in ("lon", "longitude", "x"), dimnames(ds["river_mask"]))
    dim_names = dimnames(ds["river_mask"])
    lon_dim = dim_names[1]   # first dim = longitude (x)
    lat_dim = dim_names[2]   # second dim = latitude (y)
    lons = Float64.(ds[lon_dim][:])
    lats = Float64.(ds[lat_dim][:])
    ldd, mask, lats, lons
end

nrows, ncols = size(ldd_grid)

# Build a Float32 grid: NaN outside river_mask, LDD value inside
plot_grid = fill(NaN32, nrows, ncols)
for r in 1:nrows, c in 1:ncols
    m = mask_grid[r, c]
    l = ldd_grid[r, c]
    if !ismissing(m) && m != 0 && !ismissing(l)
        plot_grid[r, c] = Float32(l)
    end
end

n_river = sum(!isnan, plot_grid)
println("  River cells with LDD: $n_river")

# LDD labels and corresponding arrow directions (dx, dy) for the annotation overlay
# PCRaster layout: row increases downward, so row+1 = south in pixel coords
const LDD_DIRS = Dict(
    1 => (name="SW", dx=-1, dy= 1),
    2 => (name="S",  dx= 0, dy= 1),
    3 => (name="SE", dx= 1, dy= 1),
    4 => (name="W",  dx=-1, dy= 0),
    5 => (name="pit",dx= 0, dy= 0),
    6 => (name="E",  dx= 1, dy= 0),
    7 => (name="NW", dx=-1, dy=-1),
    8 => (name="N",  dx= 0, dy=-1),
    9 => (name="NE", dx= 1, dy=-1),
)

# Discrete colormap for 9 LDD values (1-9)
ldd_colors = [
    Makie.ColorSchemes.colorschemes[:tableau_10].colors[i] for i in 1:9
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (900, 700))
ax  = Axis(fig[1, 1];
    title       = "Local Drainage Direction – river cells only",
    xlabel       = "Longitude (°E)",
    ylabel       = "Latitude (°N)",
    aspect      = DataAspect(),
)

# Plot the grid using a discrete colormap
# We map values 1–9 to levels, NaN cells stay transparent
hm = heatmap!(ax, lons, lats, plot_grid;
    colormap    = ldd_colors,
    colorrange  = (0.5, 9.5),
    lowclip     = :transparent,
    nan_color   = :transparent,
)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cb = Colorbar(fig[1, 2], hm;
    label      = "LDD value",
    ticks      = (1:9, [string(v, " (", LDD_DIRS[v].name, ")") for v in 1:9]),
    ticksize   = 6,
)

save(out_path, fig; px_per_unit = 2)
println("Saved: $out_path")
println("Done.")
