#!/usr/bin/env julia
# scripts/check_mass_balance.jl
#
# Verify whether the kinematic-wave mass balance holds in the wflow output data.
#
# wflow uses a same-timestep upstream sum (confirmed by check_mass_balance_details.jl):
#   qin_av[t] = Σ q_av[t]  (upstream cells at the same timestep)
#
# Three discretisation schemes are compared:
#   (A) semi-implicit (corrected):  h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t+1] + iw[t]   - q[t+1])
#   (B) fully-implicit:             h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t+1] + iw[t+1] - q[t+1])
#   (C) lagged upstream (old):      h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t]   + iw[t]   - q[t+1])
#
# If (A) or (B) residuals are small the MassBalanceLayer constraint is consistent
# with the training data.
#
# Usage:
#   julia --project=<repo_root> scripts/check_mass_balance.jl \
#       <staticmaps.nc> <output.nc> [out.png]

if length(ARGS) < 2
    println(stderr,
        "Usage: julia --project=<repo_root> scripts/check_mass_balance.jl " *
        "<staticmaps.nc> <output.nc> [out.png]")
    exit(1)
end

staticmaps_file = ARGS[1]
output_file     = ARGS[2]
out_path        = length(ARGS) >= 3 ? ARGS[3] : "mass_balance_check.png"

isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
isfile(output_file)     || (println(stderr, "Not found: $output_file");     exit(1))

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using Statistics
using CairoMakie
using Printf
using WflowRoutingGNN

coerce(x) = ismissing(x) ? NaN32 : Float32(x)

# ── 1. Build compact river graph from staticmaps ─────────────────────────────
println("Reading staticmaps: $staticmaps_file")

# Timestep and axis alignment via src functions
dt        = get_timestep(output_file)
alignment = check_and_correct_grid_alignment(staticmaps_file, output_file, "river")
println("  dt = $(dt) s  ($(round(dt/3600, digits=2)) h)")
println("  Axis flip: dim1=$(alignment.dim1_flip), dim2=$(alignment.dim2_flip)")

# Edge list via src function (sparse linear indices → compact 1..n_nodes)
src_raw, tgt_raw = ldd_to_graph(staticmaps_file, "river")

nrows, ncols, rows, cols, node_ids = NCDataset(staticmaps_file, "r") do ds
    mask = ds["river_mask"][:, :]
    nr, nc = size(mask)
    ids = sort([((c - 1) * nr + r)
                for c in 1:nc, r in 1:nr
                if !ismissing(mask[r, c]) && mask[r, c] != 0])
    rs  = [(id - 1) % nr + 1 for id in ids]
    cs  = [(id - 1) ÷ nr + 1 for id in ids]
    nr, nc, rs, cs, ids
end

id_to_idx = Dict(id => i for (i, id) in enumerate(node_ids))
sources   = [id_to_idx[s] for s in src_raw]
targets   = [id_to_idx[t] for t in tgt_raw]

# Node geometry (width, length) from staticmaps
width, length_m = NCDataset(staticmaps_file, "r") do ds
    n = length(rows)
    w = [coerce(ds["river_width"][rows[i],  cols[i]]) for i in 1:n]
    l = [coerce(ds["river_length"][rows[i], cols[i]]) for i in 1:n]
    w, l
end

n_nodes = length(rows)
println("  River cells : $n_nodes")
println("  River edges : $(length(sources))")

# ── 2. Read q, h, inwater time series for river cells ────────────────────────
println("Reading output: $output_file")

q_mat, h_mat, iw_mat, mbe_mat_raw, ntimes = NCDataset(output_file, "r") do ds
    nt = size(ds["river_q"], 3)
    q_f  = coerce.(ds["river_q"][:, :, :])
    h_f  = coerce.(ds["river_h"][:, :, :])
    iw_f = coerce.(ds["river_inwater"][:, :, :])
    has_mbe = "river_routing_water_balance" in keys(ds) ||
              "mass_balance_error"          in keys(ds)
    mbe_key = has_mbe ?
              ("river_routing_water_balance" in keys(ds) ? "river_routing_water_balance" : "mass_balance_error") :
              nothing
    qm  = Matrix{Float32}(undef, n_nodes, nt)
    hm  = Matrix{Float32}(undef, n_nodes, nt)
    iwm = Matrix{Float32}(undef, n_nodes, nt)
    mbm = isnothing(mbe_key) ? nothing : Matrix{Float32}(undef, n_nodes, nt)
    for (i, (r, c)) in enumerate(zip(rows, cols))
        r_out = alignment.dim1_flip ? nrows - r + 1 : r
        c_out = alignment.dim2_flip ? ncols - c + 1 : c
        qm[i, :]  = q_f[r_out, c_out, :]
        hm[i, :]  = h_f[r_out, c_out, :]
        iwm[i, :] = iw_f[r_out, c_out, :]
        if !isnothing(mbm)
            mbm[i, :] = coerce.(ds[mbe_key][r_out, c_out, :])
        end
    end
    isnothing(mbe_key) ? println("  mass_balance_error: not found in output") :
                         println("  mass_balance_error: found as '$mbe_key'")
    qm, hm, iwm, mbm, nt
end
println("  Timesteps : $ntimes")

# Convert wflow MBE from flux [m³/s] to h-equivalent [m]: Δh = mbe * dt / (w * l)
mbe_h = if isnothing(mbe_mat_raw)
    nothing
else
    m = fill(NaN32, n_nodes, ntimes)
    for i in 1:n_nodes
        w = width[i]; l = length_m[i]
        if isfinite(w) && isfinite(l) && w > 0f0 && l > 0f0
            m[i, :] = mbe_mat_raw[i, :] .* (dt / (w * l))
        end
    end
    m
end

# ── 3. Build upstream-sum lookup (adjacency: sources[k] -> targets[k]) ───────
# For each node, upstream_q[i,t] = sum of q[j,t] for all j where targets[k]==i
function compute_upstream_q(q_col::AbstractVector{Float32})
    uq = zeros(Float32, n_nodes)
    for k in eachindex(sources)
        uq[targets[k]] += q_col[sources[k]]
    end
    uq
end

# ── 4. Compute MB residuals for three discretisation schemes ─────────────────
# (A) semi-implicit corrected: h[t+1] = h[t] + dt/(w*l) * (Σq[t+1] + iw[t]   - q[t+1])
# (B) fully-implicit:          h[t+1] = h[t] + dt/(w*l) * (Σq[t+1] + iw[t+1] - q[t+1])
# (C) lagged upstream (old):   h[t+1] = h[t] + dt/(w*l) * (Σq[t]   + iw[t]   - q[t+1])
println("Computing mass balance residuals...")

T_steps  = ntimes - 1
residual_si  = Matrix{Float32}(undef, n_nodes, T_steps)  # semi-implicit
residual_ex  = Matrix{Float32}(undef, n_nodes, T_steps)  # explicit
residual_fi  = Matrix{Float32}(undef, n_nodes, T_steps)  # fully implicit
h_pred_si    = Matrix{Float32}(undef, n_nodes, T_steps)
h_pred_ex    = Matrix{Float32}(undef, n_nodes, T_steps)
h_pred_fi    = Matrix{Float32}(undef, n_nodes, T_steps)
h_true_next  = @view h_mat[:, 2:end]

for t in 1:T_steps
    uq_t   = compute_upstream_q(q_mat[:, t])
    uq_tp1 = compute_upstream_q(q_mat[:, t+1])
    for i in 1:n_nodes
        w = width[i]; l = length_m[i]
        if isfinite(w) && isfinite(l) && w > 0f0 && l > 0f0
            coeff = dt / (w * l)
            dh_si = coeff * (uq_tp1[i] + iw_mat[i, t]   - q_mat[i, t+1])  # (A) corrected SI
            dh_ex = coeff * (uq_tp1[i] + iw_mat[i, t+1] - q_mat[i, t+1])  # (B) fully-implicit
            dh_fi = coeff * (uq_t[i]   + iw_mat[i, t]   - q_mat[i, t+1])  # (C) lagged upstream
            h_pred_si[i, t]   = max(0f0, h_mat[i, t] + dh_si)
            h_pred_ex[i, t]   = max(0f0, h_mat[i, t] + dh_ex)
            h_pred_fi[i, t]   = max(0f0, h_mat[i, t] + dh_fi)
            residual_si[i, t] = h_mat[i, t+1] - h_pred_si[i, t]
            residual_ex[i, t] = h_mat[i, t+1] - h_pred_ex[i, t]
            residual_fi[i, t] = h_mat[i, t+1] - h_pred_fi[i, t]
        else
            h_pred_si[i, t]   = NaN32;  residual_si[i, t] = NaN32
            h_pred_ex[i, t]   = NaN32;  residual_ex[i, t] = NaN32
            h_pred_fi[i, t]   = NaN32;  residual_fi[i, t] = NaN32
        end
    end
end

# ── 5. Summary statistics ────────────────────────────────────────────────────
using Printf

finite_h = filter(isfinite, vec(h_mat[:, 2:end]))

for (label, residual) in (("(A) semi-impl. corrected  Σq[t+1]+iw[t]  -q[t+1]", residual_si),
                           ("(B) fully-implicit        Σq[t+1]+iw[t+1]-q[t+1]", residual_ex),
                           ("(C) lagged upstream (old) Σq[t]  +iw[t]  -q[t+1]", residual_fi))
    finite_res = filter(isfinite, vec(residual))
    rel_res    = filter(isfinite, vec(residual ./ max.(abs.(h_mat[:, 2:end]), 1f-6)))
    @printf("\n── MB residual (%s)  h_true − h_MB  [m] ───\n", label)
    @printf("  n finite      : %d / %d  (%.1f%%)\n",
            length(finite_res), length(residual),
            100*length(finite_res)/length(residual))
    @printf("  mean          : %+.4g m\n",  mean(finite_res))
    @printf("  std           : %.4g m\n",   std(finite_res))
    @printf("  median        : %+.4g m\n",  median(finite_res))
    @printf("  p5 / p95      : %+.4g  /  %+.4g m\n",
            quantile(finite_res, 0.05), quantile(finite_res, 0.95))
    @printf("  max |residual|: %.4g m\n",   maximum(abs, finite_res))
    @printf("  median |rel|  : %.4g\n",     median(abs.(rel_res)))
    @printf("  p95    |rel|  : %.4g\n",     quantile(abs.(rel_res), 0.95))
end
if !isnothing(mbe_h)
    finite_mbe = filter(isfinite, vec(mbe_h))
    @printf("\n── Wflow mass_balance_error (converted to Δh = mbe·dt/(w·l)) ────────────\n")
    @printf("  n finite      : %d / %d  (%.1f%%)\n",
            length(finite_mbe), length(mbe_h),
            100*length(finite_mbe)/length(mbe_h))
    @printf("  mean          : %+.4g m\n",  mean(finite_mbe))
    @printf("  std           : %.4g m\n",   std(finite_mbe))
    @printf("  median        : %+.4g m\n",  median(finite_mbe))
    @printf("  p5 / p95      : %+.4g  /  %+.4g m\n",
            quantile(finite_mbe, 0.05), quantile(finite_mbe, 0.95))
    @printf("  max |error|   : %.4g m\n",   maximum(abs, finite_mbe))
end
@printf("\n── True h range ─────────────────────────────────────────────────────────\n")
@printf("  min / median / max h: %.4g / %.4g / %.4g m\n",
        minimum(finite_h), median(finite_h), maximum(finite_h))

# ── 6. Plot ───────────────────────────────────────────────────────────────────
println("\nPlotting...")

nanmedian(m) = Float32[let v = filter(isfinite, view(m, :, t))
                           isempty(v) ? NaN32 : median(v) end for t in axes(m,2)]
nanp(m, p)   = Float32[let v = filter(isfinite, view(m, :, t))
                           isempty(v) ? NaN32 : quantile(v, p) end for t in axes(m,2)]

xs = 1:T_steps

n_extra_rows = isnothing(mbe_h) ? 0 : 2   # +1 time-series panel, +1 histogram
fig = Figure(size = (1200, 1550 + n_extra_rows * 250))
Label(fig[0, 1:2], "Mass balance check  —  $(basename(output_file))";
      fontsize = 14, font = :bold)

# Panel 1: h_true vs all three MB-predicted versions
ax1 = Axis(fig[1, 1]; title = "True vs MB-predicted h (spatial median + p10–p90)",
           ylabel = "h [m]")
hidexdecorations!(ax1; ticks = false)
band!(ax1, xs, nanp(h_pred_si, 0.1), nanp(h_pred_si, 0.9); color = (:steelblue,   0.20))
l_hp_si = lines!(ax1, xs, nanmedian(h_pred_si); color = :steelblue)
band!(ax1, xs, nanp(h_pred_ex, 0.1), nanp(h_pred_ex, 0.9); color = (:forestgreen, 0.20))
l_hp_ex = lines!(ax1, xs, nanmedian(h_pred_ex); color = :forestgreen, linestyle = :dash)
band!(ax1, xs, nanp(h_pred_fi, 0.1), nanp(h_pred_fi, 0.9); color = (:purple,      0.20))
l_hp_fi = lines!(ax1, xs, nanmedian(h_pred_fi); color = :purple,      linestyle = :dot)
band!(ax1, xs, nanp(h_true_next, 0.1), nanp(h_true_next, 0.9); color = (:orangered, 0.25))
l_ht = lines!(ax1, xs, nanmedian(h_true_next); color = :orangered)
Legend(fig[1, 2], [l_hp_si, l_hp_ex, l_hp_fi, l_ht],
       ["(A) SI corrected", "(B) fully-impl.", "(C) lagged", "true h"];
       framevisible = false)

# Panel 2: residuals over time — all three schemes
ax2 = Axis(fig[2, 1]; title = "Residuals h_true − h_MB over time (spatial median + p10–p90)",
           ylabel = "Δh [m]")
hidexdecorations!(ax2; ticks = false)
band!(ax2, xs, nanp(residual_si, 0.1), nanp(residual_si, 0.9); color = (:steelblue,   0.20))
l_rs = lines!(ax2, xs, nanmedian(residual_si); color = :steelblue)
band!(ax2, xs, nanp(residual_ex, 0.1), nanp(residual_ex, 0.9); color = (:forestgreen, 0.20))
l_re = lines!(ax2, xs, nanmedian(residual_ex); color = :forestgreen, linestyle = :dash)
band!(ax2, xs, nanp(residual_fi, 0.1), nanp(residual_fi, 0.9); color = (:purple,      0.20))
l_rf = lines!(ax2, xs, nanmedian(residual_fi); color = :purple,      linestyle = :dot)
hlines!(ax2, [0f0]; color = :black, linestyle = :dash)
Legend(fig[2, 2], [l_rs, l_re, l_rf],
       ["(A) SI corrected", "(B) fully-impl.", "(C) lagged"];
       framevisible = false)

# Panel 2b (optional): wflow mass_balance_error in h-equivalent units
if !isnothing(mbe_h)
    xs_mbe = 1:ntimes
    ax2b = Axis(fig[3, 1];
                title = "Wflow mass_balance_error  (flux → Δh = mbe·dt/(w·l), spatial median + p10–p90)",
                ylabel = "Δh [m]")
    hidexdecorations!(ax2b; ticks = false)
    band!(ax2b, xs_mbe, nanp(mbe_h, 0.1), nanp(mbe_h, 0.9); color = (:darkorange, 0.25))
    l_mbe = lines!(ax2b, xs_mbe, nanmedian(mbe_h); color = :darkorange)
    hlines!(ax2b, [0f0]; color = :black, linestyle = :dash)
    Legend(fig[3, 2], [l_mbe], ["wflow MBE (Δh)"];
           framevisible = false)
end

# Adjust row offset for subsequent panels
_row_off = isnothing(mbe_h) ? 0 : 1

# Panels 3, 4, 5: residual histograms
finite_res_si = filter(isfinite, vec(residual_si))
finite_res_ex = filter(isfinite, vec(residual_ex))
finite_res_fi = filter(isfinite, vec(residual_fi))

for (row, (label, color, finite_res)) in enumerate([
        ("(A) semi-impl. corrected  Σq[t+1]+iw[t]  -q[t+1]", :steelblue,   finite_res_si),
        ("(B) fully-implicit        Σq[t+1]+iw[t+1]-q[t+1]", :forestgreen, finite_res_ex),
        ("(C) lagged upstream (old) Σq[t]  +iw[t]  -q[t+1]", :purple,      finite_res_fi)])
    ax = Axis(fig[2 + _row_off + row, 1]; title = "Residual histogram — $label",
              xlabel = "h_true − h_MB  [m]", ylabel = "count")
    lo, hi = quantile(finite_res, 0.01), quantile(finite_res, 0.99)
    hist!(ax, clamp.(finite_res, lo, hi); bins = 80, color = (color, 0.7))
    vlines!(ax, [0f0]; color = :black, linestyle = :dash)
    Legend(fig[2 + _row_off + row, 2], [PolyElement(color = (color, 0.7))], ["p1–p99 clamped"];
           framevisible = false)
end

if !isnothing(mbe_h)
    finite_mbe_plot = filter(isfinite, vec(mbe_h))
    ax_mbe_hist = Axis(fig[2 + _row_off + 4, 1];
                       title = "Wflow mass_balance_error histogram (Δh = mbe·dt/(w·l))",
                       xlabel = "Δh [m]", ylabel = "count")
    lo_m, hi_m = quantile(finite_mbe_plot, 0.01), quantile(finite_mbe_plot, 0.99)
    hist!(ax_mbe_hist, clamp.(finite_mbe_plot, lo_m, hi_m); bins = 80, color = (:darkorange, 0.7))
    vlines!(ax_mbe_hist, [0f0]; color = :black, linestyle = :dash)
    Legend(fig[2 + _row_off + 4, 2], [PolyElement(color = (:darkorange, 0.7))], ["p1–p99 clamped"];
           framevisible = false)
end

colsize!(fig.layout, 2, Fixed(160))
save(out_path, fig)
println("Saved → $out_path")
