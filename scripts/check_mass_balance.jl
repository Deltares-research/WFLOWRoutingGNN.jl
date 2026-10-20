#!/usr/bin/env julia
# scripts/check_mass_balance.jl
#
# Verify whether the kinematic-wave mass balance holds in the wflow output data.
#
# wflow uses a same-timestep upstream sum (confirmed by check_mass_balance_details.jl):
#   qin_av[t] = Σ q_av[t]  (upstream cells at the same timestep)
#
# Three discretisation schemes are compared (`MB_SCHEMES`, src/consistency.jl):
#   (A) semi-implicit (corrected):  h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t+1] + iw[t]   - q[t+1])
#   (B) fully-implicit:             h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t+1] + iw[t+1] - q[t+1])
#   (C) lagged upstream (old):      h[t+1] ≈ h[t] + dt/(w·l)·(Σq[t]   + iw[t]   - q[t+1])
#
# If (A) or (B) residuals are small the MassBalanceLayer constraint is consistent
# with the training data. The residual computation lives in `src/consistency.jl`;
# this script prints the statistics and draws the figure.
#
# Usage:
#   julia --project=<repo_root> scripts/check_mass_balance.jl \
#       <staticmaps.nc> <output.nc> [out.png]

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using Statistics
using CairoMakie
using Printf
using WflowRoutingGNN

const SCHEME_LABELS = Dict(
    :semi_implicit   => "(A) semi-impl. corrected  Σq[t+1]+iw[t]  -q[t+1]",
    :fully_implicit  => "(B) fully-implicit        Σq[t+1]+iw[t+1]-q[t+1]",
    :lagged_upstream => "(C) lagged upstream (old) Σq[t]  +iw[t]  -q[t+1]",
)

nanmedian(m) = Float32[let v = filter(isfinite, view(m, :, t))
                           isempty(v) ? NaN32 : median(v) end for t in axes(m, 2)]
nanp(m, p)   = Float32[let v = filter(isfinite, view(m, :, t))
                           isempty(v) ? NaN32 : quantile(v, p) end for t in axes(m, 2)]

function print_summary(header, s)
    @printf("\n── %s ───\n", header)
    @printf("  n finite      : %d / %d  (%.1f%%)\n", s.n_finite, s.n, 100 * s.n_finite / s.n)
    @printf("  mean          : %+.4g m\n", s.mean)
    @printf("  std           : %.4g m\n", s.std)
    @printf("  median        : %+.4g m\n", s.median)
    @printf("  p5 / p95      : %+.4g  /  %+.4g m\n", s.p5, s.p95)
end

function main(argv)
    if length(argv) < 2
        println(stderr,
            "Usage: julia --project=<repo_root> scripts/check_mass_balance.jl " *
            "<staticmaps.nc> <output.nc> [out.png]")
        exit(1)
    end
    staticmaps_file, output_file = argv[1], argv[2]
    out_path = length(argv) >= 3 ? argv[3] : "mass_balance_check.png"
    isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
    isfile(output_file)     || (println(stderr, "Not found: $output_file");     exit(1))

    # ── 1. River graph and geometry ──────────────────────────────────────────
    println("Reading staticmaps: $staticmaps_file")
    layout = river_layout(staticmaps_file, output_file)
    (; n_nodes, dt, width, length_m) = layout
    println("  dt = $(dt) s  ($(round(dt/3600, digits=2)) h)")
    println("  Axis flip: dim1=$(layout.dim1_flip), dim2=$(layout.dim2_flip)")
    println("  River cells : $n_nodes")
    println("  River edges : $(length(layout.edge_src))")

    # ── 2. q, h, inwater (+ wflow's own mass-balance error) ──────────────────
    println("Reading output: $output_file")
    q_mat, h_mat, iw_mat, mbe_raw = NCDataset(output_file, "r") do ds
        mbe_key = "river_routing_water_balance" in keys(ds) ? "river_routing_water_balance" :
                  "mass_balance_error"          in keys(ds) ? "mass_balance_error" : nothing
        isnothing(mbe_key) ? println("  mass_balance_error: not found in output") :
                             println("  mass_balance_error: found as '$mbe_key'")
        (read_river_series(ds, layout, "river_q"),
         read_river_series(ds, layout, "river_h"),
         read_river_series(ds, layout, "river_inwater"),
         isnothing(mbe_key) ? nothing : read_river_series(ds, layout, mbe_key))
    end
    any(isnothing, (q_mat, h_mat, iw_mat)) &&
        error("output file needs river_q, river_h and river_inwater")
    ntimes = size(q_mat, 2)
    println("  Timesteps : $ntimes")

    # wflow MBE from flux [m³/s] to h-equivalent [m]: Δh = mbe * dt / (w * l)
    mbe_h = if isnothing(mbe_raw)
        nothing
    else
        m = fill(NaN32, n_nodes, ntimes)
        for i in 1:n_nodes
            w = width[i]; l = length_m[i]
            if isfinite(w) && isfinite(l) && w > 0f0 && l > 0f0
                m[i, :] = mbe_raw[i, :] .* (dt / (w * l))
            end
        end
        m
    end

    # ── 3. Residuals of the three schemes ────────────────────────────────────
    println("Computing mass balance residuals...")
    wl = width .* length_m
    res = Dict(s => mb_scheme_residuals(q_mat, h_mat, iw_mat;
                                        edge_src = layout.edge_src, edge_tgt = layout.edge_tgt,
                                        wl, dt, scheme = s) for s in MB_SCHEMES)
    residual_si, residual_ex, residual_fi = (res[s].residual for s in MB_SCHEMES)
    h_pred_si,   h_pred_ex,   h_pred_fi   = (res[s].h_pred   for s in MB_SCHEMES)
    h_true_next = @view h_mat[:, 2:end]
    T_steps     = ntimes - 1

    # ── 4. Summary statistics ────────────────────────────────────────────────
    finite_h = filter(isfinite, vec(h_true_next))
    for s in MB_SCHEMES
        residual = res[s].residual
        st       = residual_summary(residual)
        rel_res  = filter(isfinite, vec(residual ./ max.(abs.(h_true_next), 1f-6)))
        print_summary("MB residual ($(SCHEME_LABELS[s]))  h_true − h_MB  [m]", st)
        @printf("  max |residual|: %.4g m\n", st.max_abs)
        @printf("  median |rel|  : %.4g\n",   median(abs.(rel_res)))
        @printf("  p95    |rel|  : %.4g\n",   quantile(abs.(rel_res), 0.95))
    end
    if !isnothing(mbe_h)
        st = residual_summary(mbe_h)
        print_summary("Wflow mass_balance_error (converted to Δh = mbe·dt/(w·l))", st)
        @printf("  max |error|   : %.4g m\n", st.max_abs)
    end
    @printf("\n── True h range ─────────────────────────────────────────────────────────\n")
    @printf("  min / median / max h: %.4g / %.4g / %.4g m\n",
            minimum(finite_h), median(finite_h), maximum(finite_h))

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    println("\nPlotting...")
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
        Legend(fig[3, 2], [l_mbe], ["wflow MBE (Δh)"]; framevisible = false)
    end

    _row_off = isnothing(mbe_h) ? 0 : 1

    # Panels 3, 4, 5: residual histograms
    for (row, (s, color)) in enumerate(zip(MB_SCHEMES, (:steelblue, :forestgreen, :purple)))
        finite_res = filter(isfinite, vec(res[s].residual))
        ax = Axis(fig[2 + _row_off + row, 1]; title = "Residual histogram — $(SCHEME_LABELS[s])",
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
