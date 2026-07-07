#!/usr/bin/env julia
# scripts/inspect_scaling.jl
#
# Plot histograms of river_q, river_h, and river_inwater at three stages:
#   1. Raw (physical units straight from output.nc)
#   2. After the variable-specific custom scaler applied in preprocess.jl
#      - river_q  : divided by meta_upstream_area  →  [m³/s / m² = m/s]
#      - river_h  : multiplied by w·l / area        →  [dimensionless]
#      - river_inwater : no custom scaler
#   3. After z-score normalisation of the custom-scaled values
#
# Usage:
#   julia --project=<repo_root> scripts/inspect_scaling.jl \
#       <staticmaps.nc> <output.nc> [output.png]

if length(ARGS) < 2
    println(stderr, "Usage: julia --project=<repo_root> scripts/inspect_scaling.jl " *
                    "<staticmaps.nc> <output.nc> [out.png]")
    exit(1)
end

staticmaps_file = ARGS[1]
output_file     = ARGS[2]
out_path        = length(ARGS) >= 3 ? ARGS[3] : "scaling_histograms.png"

isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
isfile(output_file)     || (println(stderr, "Not found: $output_file");     exit(1))

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using Statistics
using CairoMakie
using Printf

coerce(x) = ismissing(x) ? NaN32 : Float32(x)

# ── 1. Read river mask and per-node scaling constants from staticmaps ────────
println("Reading staticmaps: $staticmaps_file")

nrows, ncols, rows, cols, area, river_width, river_length =
    NCDataset(staticmaps_file, "r") do ds
        mask       = ds["river_mask"][:, :]          # (nrows, ncols)
        nr, nc     = size(mask)
        rs, cs     = Int[], Int[]
        for c in 1:nc, r in 1:nr
            if !ismissing(mask[r, c]) && mask[r, c] != 0
                push!(rs, r)
                push!(cs, c)
            end
        end
        n = length(rs)
        a_arr  = [coerce(ds["meta_upstream_area"][rs[i], cs[i]]) for i in 1:n]
        w_arr  = [coerce(ds["river_width"][rs[i],        cs[i]]) for i in 1:n]
        l_arr  = [coerce(ds["river_length"][rs[i],       cs[i]]) for i in 1:n]
        nr, nc, rs, cs, a_arr, w_arr, l_arr
    end

n_nodes = length(rows)
println("  River cells: $n_nodes")

# ── 2. Detect axis flip between staticmaps and output.nc ────────────────────
dim1_flip, dim2_flip = NCDataset(staticmaps_file, "r") do sm
    NCDataset(output_file, "r") do out
        sm_dims  = [d for d in dimnames(sm["local_drain_direction"]) if d ∉ ("time","layer")]
        out_dims = [d for d in dimnames(out["river_q"])              if d ∉ ("time","layer")]
        flips = map(1:2) do i
            sc = Float64.(sm[sm_dims[i]][:])
            oc = Float64.(out[out_dims[i]][:])
            isapprox(sc[1], oc[end]; rtol=1e-5) && isapprox(sc[end], oc[1]; rtol=1e-5)
        end
        flips[1], flips[2]
    end
end
println("  Axis flip: dim1=$dim1_flip, dim2=$dim2_flip")

# ── 3. Read time series for river cells from output.nc ──────────────────────
println("Reading output: $output_file")

vars_to_read = filter(v -> true, ["river_q", "river_h", "river_inwater"])

q_raw, h_raw, inwater_raw = NCDataset(output_file, "r") do ds
    # Check which variables are present
    for v in ["river_q", "river_h", "river_inwater"]
        haskey(ds, v) || error("Variable '$v' not found in $output_file")
    end

    ntimes = size(ds["river_q"], 3)
    println("  Time steps: $ntimes")

    # Read full 3-D arrays once (much faster than per-cell reads)
    q_full  = coerce.(ds["river_q"][:, :, :])
    h_full  = coerce.(ds["river_h"][:, :, :])
    iw_full = coerce.(ds["river_inwater"][:, :, :])

    q_mat  = Matrix{Float32}(undef, n_nodes, ntimes)
    h_mat  = Matrix{Float32}(undef, n_nodes, ntimes)
    iw_mat = Matrix{Float32}(undef, n_nodes, ntimes)

    for (i, (r, c)) in enumerate(zip(rows, cols))
        r_out = dim1_flip ? nrows - r + 1 : r
        c_out = dim2_flip ? ncols - c + 1 : c
        q_mat[i, :]  = q_full[r_out, c_out, :]
        h_mat[i, :]  = h_full[r_out, c_out, :]
        iw_mat[i, :] = iw_full[r_out, c_out, :]
    end

    q_mat, h_mat, iw_mat
end

# ── 4. Apply custom scalers ──────────────────────────────────────────────────
# river_q : divide by meta_upstream_area  →  [m/s]
q_scaled = copy(q_raw)
for i in 1:n_nodes
    if isfinite(area[i]) && area[i] > 0f0
        q_scaled[i, :] ./= area[i]
    end
end

# river_h : multiply by (w·l / area)  →  [dimensionless volume proxy]
h_scaled = copy(h_raw)
for i in 1:n_nodes
    a, w, l = area[i], river_width[i], river_length[i]
    if isfinite(a) && isfinite(w) && isfinite(l) && a > 0f0
        h_scaled[i, :] .*= (w * l / a)
    end
end

# river_inwater : no custom scaler
inwater_scaled = inwater_raw

# ── 5. Z-score normalise ─────────────────────────────────────────────────────
function zscore(x::AbstractMatrix{Float32})
    v = filter(isfinite, vec(x))
    μ = mean(v)
    σ = std(v);  σ = σ == 0f0 ? 1f0 : σ
    (x .- μ) ./ σ, μ, σ
end

q_norm,       μ_q,  σ_q  = zscore(q_scaled)
h_norm,       μ_h,  σ_h  = zscore(h_scaled)
inwater_norm, μ_iw, σ_iw = zscore(inwater_scaled)

# ── 6. Summary stats ─────────────────────────────────────────────────────────
function describe(label, x; log_transform=false)
    v = filter(isfinite, vec(x))
    log_transform && (v = filter(>(0), v); v = log10.(v))
    @printf("  %-30s  n=%7d  min=%12.4g  median=%12.4g  max=%12.4g  std=%12.4g\n",
            label, length(v), minimum(v), median(v), maximum(v), std(v))
end

using Printf
println("\n── Statistics ───────────────────────────────────────────────────────────────")
describe("river_q raw [m³/s]",            q_raw;       log_transform=true)
describe("river_q / area [m/s]",          q_scaled;    log_transform=true)
describe("river_q z-scored",              q_norm)
describe("river_h raw [m]",               h_raw)
describe("river_h × w·l/A [–]",           h_scaled)
describe("river_h z-scored",              h_norm)
describe("river_inwater raw [m³/s]",      inwater_raw; log_transform=true)
describe("river_inwater z-scored",        inwater_norm)
@printf("  %-30s  μ=%.4g  σ=%.4g\n", "river_q z-score params",       μ_q,  σ_q)
@printf("  %-30s  μ=%.4g  σ=%.4g\n", "river_h z-score params",       μ_h,  σ_h)
@printf("  %-30s  μ=%.4g  σ=%.4g\n", "river_inwater z-score params", μ_iw, σ_iw)

# ── 7. Plot ───────────────────────────────────────────────────────────────────
println("\nPlotting ...")
NBINS = 80

# Take log10 of strictly-positive values; skip ≤ 0
log10_pos(x) = Float32.(log10.(filter(v -> isfinite(v) && v > 0f0, vec(x))))
finite_vec(x) = Float32.(filter(isfinite, vec(x)))

fig = Figure(size = (1400, 960))

# ── Column header labels ──────────────────────────────────────────────────────
Label(fig[1, 2], "Raw (physical units)"; fontsize = 13, font = :bold)
Label(fig[1, 3], "After custom scaling";  fontsize = 13, font = :bold)
Label(fig[1, 4], "After z-score";         fontsize = 13, font = :bold)

# ── Row 1: river_q ────────────────────────────────────────────────────────────
Label(fig[2, 1], "river_q"; fontsize = 13, font = :bold, rotation = π/2)
ax_q1 = Axis(fig[2, 2]; xlabel = "log₁₀(Q)  [log(m³/s)]",  ylabel = "count", yscale = log10)
ax_q2 = Axis(fig[2, 3]; xlabel = "log₁₀(Q/A)  [log(m/s)]", ylabel = "count", yscale = log10)
ax_q3 = Axis(fig[2, 4]; xlabel = "normalised  [–]",          ylabel = "count", yscale = log10)
hist!(ax_q1, log10_pos(q_raw);    bins = NBINS, color = (:steelblue,   0.75))
hist!(ax_q2, log10_pos(q_scaled); bins = NBINS, color = (:darkorange,  0.75))
hist!(ax_q3, finite_vec(q_norm);  bins = NBINS, color = (:forestgreen, 0.75))

# ── Row 2: river_h ────────────────────────────────────────────────────────────
Label(fig[3, 1], "river_h"; fontsize = 13, font = :bold, rotation = π/2)
ax_h1 = Axis(fig[3, 2]; xlabel = "h  [m]",          ylabel = "count", yscale = log10)
ax_h2 = Axis(fig[3, 3]; xlabel = "h·w·l/A  [–]",    ylabel = "count", yscale = log10)
ax_h3 = Axis(fig[3, 4]; xlabel = "normalised  [–]",  ylabel = "count", yscale = log10)
hist!(ax_h1, finite_vec(h_raw);    bins = NBINS, color = (:steelblue,   0.75))
hist!(ax_h2, finite_vec(h_scaled); bins = NBINS, color = (:darkorange,  0.75))
hist!(ax_h3, finite_vec(h_norm);   bins = NBINS, color = (:forestgreen, 0.75))

# ── Row 3: river_inwater ──────────────────────────────────────────────────────
Label(fig[4, 1], "river_inwater"; fontsize = 13, font = :bold, rotation = π/2)
ax_i1 = Axis(fig[4, 2]; xlabel = "log₁₀(inwater)  [log(m³/s)]", ylabel = "count", yscale = log10)
ax_i3 = Axis(fig[4, 4]; xlabel = "normalised  [–]",               ylabel = "count", yscale = log10)
hist!(ax_i1, log10_pos(inwater_raw);  bins = NBINS, color = (:steelblue,   0.75))
hist!(ax_i3, finite_vec(inwater_norm); bins = NBINS, color = (:forestgreen, 0.75))
Label(fig[4, 3], "no custom\nscaling"; fontsize = 12, color = :gray60)

# ── Overall title ─────────────────────────────────────────────────────────────
Label(fig[0, 1:4],
      "River variable distributions — $(basename(output_file))";
      fontsize = 15, font = :bold)

# Make row-label column narrow
colsize!(fig.layout, 1, Fixed(90))

save(out_path, fig)
println("Saved → $out_path")
