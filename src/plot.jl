using CairoMakie
import CairoMakie: record

"""
    plot_losses(train_rollout, val_rollout, train_1step, val_1step;
                path = nothing) -> Figure

Plot the four per-epoch loss arrays returned by `train_model!`.

Two panels are drawn side-by-side:
- Left  : multi-step rollout loss (train and validation).
- Right : 1-step-ahead MSE        (train and validation).

Arguments:
- `train_rollout` : `Vector{Float32}` -- training rollout loss per epoch.
- `val_rollout`   : `Vector{Float32}` -- validation rollout loss per epoch.
- `train_1step`   : `Vector{Float32}` -- training 1-step loss per epoch.
- `val_1step`     : `Vector{Float32}` -- validation 1-step loss per epoch.
- `path`          : optional file path; if given the figure is saved there
                    (format inferred from the extension, e.g. `.png`, `.pdf`).

Returns the `Figure` object.
"""
function plot_losses(train_rollout, val_rollout, train_1step, val_1step;
                     train_q_1step = nothing,
                     val_q_1step   = nothing,
                     train_h_1step = nothing,
                     val_h_1step   = nothing,
                     path = nothing)
    epochs         = 1:length(train_rollout)
    has_components = !isnothing(train_q_1step) &&
                     !isempty(train_q_1step)   &&
                     any(isfinite, train_q_1step)

    fig = Figure(size = (600, has_components ? 700 : 400))

    ax1 = Axis(fig[1, 1];
               title  = "Training losses",
               xlabel = "Epoch",
               ylabel = "MSE")
    lines!(ax1, epochs, train_rollout; label = "train rollout", color = :steelblue)
    lines!(ax1, epochs, val_rollout;   label = "val rollout",   color = :steelblue,
           linestyle = :dash)
    lines!(ax1, epochs, train_1step;   label = "train 1-step",  color = :orangered)
    lines!(ax1, epochs, val_1step;     label = "val 1-step",    color = :orangered,
           linestyle = :dash)
    axislegend(ax1; position = :rt)

    if has_components
        ax2 = Axis(fig[2, 1];
                   title  = "1-step loss components (Q vs H)",
                   xlabel = "Epoch",
                   ylabel = "MSE")
        lines!(ax2, epochs, train_q_1step; label = "train Q", color = :steelblue)
        lines!(ax2, epochs, val_q_1step;   label = "val Q",   color = :steelblue,
               linestyle = :dash)
        lines!(ax2, epochs, train_h_1step; label = "train H", color = :forestgreen)
        lines!(ax2, epochs, val_h_1step;   label = "val H",   color = :forestgreen,
               linestyle = :dash)
        axislegend(ax2; position = :rt)
    end

    isnothing(path) || save(path, fig)

    return fig
end

"""
    plot_validation_movie(pred_grids, true_grids, domain; path, framerate, timestamps) -> Figure

Record an animated movie comparing ground-truth and predicted states on the
full raster grid, with one row of panels per state variable.

Each row contains three panels:
- **Truth**      : ground-truth state at each timestep.
- **Prediction** : model-predicted state at each timestep.
- **|error|**    : absolute pointwise error |pred − truth| at each timestep.

Colour limits are fixed globally across all timesteps so the animation is
comparable frame-to-frame.

Arguments:
- `pred_grids`  : `Dict{String, Array{Float32,3}}` as returned by `regrid`,
                  mapping variable name to `(nrows, ncols, T)` array.
- `true_grids`  : same structure for ground-truth states.
- `domain`      : routing domain string (key of `DOMAIN_VARS`).
- `path`        : output file path; format is inferred from the extension
                  (`.mp4` requires FFmpeg, `.gif` has no extra dependencies).
- `framerate`   : frames per second of the output movie.
- `timestamps`  : optional `Vector` of labels (e.g. `DateTime` or `String`)
                  of length `T`. When provided, each frame shows the
                  corresponding label instead of "t = i / T".

Returns the `Figure` object (after recording is complete).
"""
function plot_validation_movie(
        pred_grids  :: Dict{String, Array{Float32,3}},
        true_grids  :: Dict{String, Array{Float32,3}},
        domain      :: String;
        path        :: String  = "validation.mp4",
        framerate   :: Int     = 10,
        timestamps           = nothing)

    state_vars = DOMAIN_VARS[domain]["state"]
    isempty(state_vars) && throw(ArgumentError("domain \"$domain\" has no state variables"))
    nvars = length(state_vars)
    T     = size(first(values(true_grids)), 3)

    t_obs = Observable(1)

    fig = Figure(size = (1100, 340 * nvars + 200))

    for (row, vname) in enumerate(state_vars)
        tg = true_grids[vname]   # (nrows, ncols, T)
        pg = pred_grids[vname]

        # Global colour limits (ignore NaN — non-active cells)
        valid_true = filter(!isnan, vec(tg))

        vmin = isempty(valid_true) ? 0f0 : minimum(valid_true)
        vmax = isempty(valid_true) ? 1f0 : maximum(valid_true)
        vmax = vmin == vmax ? vmin + 1f0 : vmax

        # Pointwise relative error: (pred - truth) / truth
        # NaN where truth == 0 or is NaN; take abs for colour range.
        rel_err_all = Float32[
            let t_ = tg[k], p_ = pg[k]
                (isnan(t_) || t_ == 0f0) ? NaN32 : (p_ - t_) / t_
            end
            for k in eachindex(tg)]
        valid_rel = filter(!isnan, vec(rel_err_all))
        emax = isempty(valid_rel) ? 1f0 : maximum(abs, valid_rel)
        emax = emax == 0f0 ? 1f0 : emax
        emax = min(emax, 1f2)  # cap at 100% for better colour contrast

        tg_obs = @lift(tg[:, :, $t_obs])
        pg_obs = @lift(pg[:, :, $t_obs])
        eg_obs = @lift(Float32[
            let t_ = tg[i, j, $t_obs], p_ = pg[i, j, $t_obs]
                (isnan(t_) || t_ == 0f0) ? NaN32 : (p_ - t_) / t_
            end
            for i in axes(tg, 1), j in axes(tg, 2)])

        ax1 = Axis(fig[row, 1]; title = "$vname — truth",            aspect = DataAspect())
        ax2 = Axis(fig[row, 2]; title = "$vname — prediction",       aspect = DataAspect())
        ax3 = Axis(fig[row, 3]; title = "$vname — (pred-truth)/truth", aspect = DataAspect())

        hm1 = heatmap!(ax1, tg_obs; colorrange = (vmin, vmax),   colormap = :viridis)
        hm2 = heatmap!(ax2, pg_obs; colorrange = (vmin, vmax),   colormap = :viridis)
        hm3 = heatmap!(ax3, eg_obs; colorrange = (-emax, emax),  colormap = :RdBu)

        Colorbar(fig[row, 4], hm1; label = vname)
        Colorbar(fig[row, 5], hm3; label = "(pred-truth)/truth")
    end

    # --- Mean relative error timeseries with moving frame marker ---
    # Pre-compute the mean (pred-truth)/truth per variable per timestep
    # (averaging over all active cells where truth ≠ 0 and is non-NaN).
    mean_errors = [
        Float32[let vals = filter(!isnan, Float32[
                                let t_ = true_grids[vn][k, t], p_ = pred_grids[vn][k, t]
                                    (isnan(t_) || t_ == 0f0) ? NaN32 : (p_ - t_) / t_
                                end
                                for k in CartesianIndices(view(true_grids[vn], :, :, 1))])
                    isempty(vals) ? NaN32 : mean(vals)
                end
                for t in 1:T]
        for vn in state_vars
    ]

    n_ticks_err  = min(T, 6)
    tick_idx_err = unique(round.(Int, range(1, T; length = n_ticks_err)))
    err_ax = Axis(fig[nvars + 1, 1:3];
                  title              = "Mean (pred−truth)/truth over graph nodes",
                  xlabel             = isnothing(timestamps) ? "Timestep" : "Time",
                  ylabel             = "Mean relative error",
                  xticklabelrotation = isnothing(timestamps) ? 0.0 : π/4,
                  xticklabelalign    = isnothing(timestamps) ?
                                       (:center, :top) : (:right, :top),
                  xticks             = isnothing(timestamps) ? Makie.automatic :
                                       (tick_idx_err, string.(timestamps[tick_idx_err])))

    palette = cgrad(:tab10; categorical = true)
    for (vi, vname) in enumerate(state_vars)
        lines!(err_ax, 1:T, mean_errors[vi];
               label  = vname,
               color  = palette[vi])
        # Scatter marker tracking the current frame on each line
        scatter!(err_ax,
                 @lift([$t_obs]),
                 @lift([mean_errors[vi][$t_obs]]);
                 color      = palette[vi],
                 markersize = 12)
    end
    length(state_vars) > 1 && axislegend(err_ax; position = :rt)

    Label(fig[nvars + 2, 1:3],
          @lift(isnothing(timestamps) ? "t = $($t_obs) / $T" :
                string(timestamps[$t_obs]));
          halign = :center, tellwidth = false)

    record(fig, path, 1:T; framerate = framerate) do t
        t_obs[] = t
    end

    return fig
end

"""
    plot_timeseries(pred_grids, true_grids, domain, row, col; path) -> Figure

Plot the predicted and ground-truth timeseries for a single grid cell.

One panel per state variable is drawn, each showing truth and prediction over
all timesteps on the same axes.

Arguments:
- `pred_grids` : `Dict{String, Array{Float32,3}}` as returned by `ungrid`.
- `true_grids` : same structure for ground-truth states.
- `domain`     : routing domain string (key of `DOMAIN_VARS`).
- `row`, `col` : 1-based raster position of the cell to inspect.
- `path`       : optional output file path; format inferred from extension.

Returns the `Figure` object.
"""
function plot_timeseries(
        pred_grids :: Dict{String, Array{Float32,3}},
        true_grids :: Dict{String, Array{Float32,3}},
        domain     :: String,
        row        :: Int,
        col        :: Int;
        path       = nothing,
        timestamps = nothing)

    state_vars = DOMAIN_VARS[domain]["state"]
    isempty(state_vars) && throw(ArgumentError("domain \"$domain\" has no state variables"))
    nvars = length(state_vars)
    T     = size(first(values(true_grids)), 3)
    ts    = 1:T

    all(isnan, true_grids[state_vars[1]][row, col, :]) &&
        @warn "Cell ($row, $col) is inactive (all NaN); plot will be empty"

    fig = Figure(size = (1200, 300 * nvars))

    for (vi, vname) in enumerate(state_vars)
        truth = true_grids[vname][row, col, :]
        pred  = pred_grids[vname][row, col, :]

        # --- timeseries panel ---
        ax_ts = Axis(fig[vi, 1];
                     title                = "$vname  —  cell ($row, $col)",
                     xlabel               = isnothing(timestamps) ? "Timestep" : "Time",
                     ylabel               = vname,
                     xticklabelrotation   = isnothing(timestamps) ? 0.0 : π/4,
                     xticklabelalign      = isnothing(timestamps) ?
                                            (:center, :top) : (:right, :top))
        if !isnothing(timestamps)
            n_ticks    = min(T, 6)
            tick_idx   = unique(round.(Int, range(1, T; length = n_ticks)))
            ax_ts.xticks = (tick_idx, string.(timestamps[tick_idx]))
        end
        lines!(ax_ts, ts, truth; label = "truth",      color = :steelblue)
        lines!(ax_ts, ts, pred;  label = "prediction", color = :orangered,
               linestyle = :dash)
        axislegend(ax_ts; position = :rt)

        # --- pred vs truth scatter panel ---
        mask        = isfinite.(truth) .& isfinite.(pred)
        truth_valid = truth[mask]
        pred_valid  = pred[mask]

        ax_sc = Axis(fig[vi, 2];
                     title  = "$vname  —  pred vs truth",
                     xlabel = "truth",
                     ylabel = "prediction",
                     aspect = 1)
        if !isempty(truth_valid)
            scatter!(ax_sc, truth_valid, pred_valid;
                     color = (:steelblue, 0.4), markersize = 4)
            lo = min(minimum(truth_valid), minimum(pred_valid))
            hi = max(maximum(truth_valid), maximum(pred_valid))
            lines!(ax_sc, [lo, hi], [lo, hi]; color = :black, linewidth = 1.5)
        end
    end

    isnothing(path) || save(path, fig)
    return fig
end

"""
    plot_downstream_timeseries(pred_grids, true_grids, domain, grid, upstream_area; path) -> Figure

Identify the most downstream active node of the river network using the
provided per-node upstream area values and call `plot_timeseries` for it.

The most downstream node is the one with the largest upstream catchment area
among all active (non-NaN) nodes.

Arguments:
- `pred_grids`    : `Dict{String, Array{Float32,3}}` as returned by `ungrid`.
- `true_grids`    : same structure for ground-truth states.
- `domain`        : routing domain string (key of `DOMAIN_VARS`).
- `grid`          : `NamedTuple` `(rows, cols, nrows, ncols)` as returned by
                    `build_wflow_graph`.
- `upstream_area` : `Vector{Float32}` of per-node upstream catchment area values
                    (length = number of graph nodes), e.g. from `meta_upstream_area`
                    in staticmaps. NaN values are ignored.
- `path`          : optional output file path.

Returns the `Figure` object.
"""
function plot_downstream_timeseries(
        pred_grids    :: Dict{String, Array{Float32,3}},
        true_grids    :: Dict{String, Array{Float32,3}},
        domain        :: String,
        grid          :: NamedTuple,
        upstream_area :: AbstractVector{<:Real};
        path          = nothing,
        timestamps    = nothing)

    # Most downstream node = largest upstream catchment area (ignore NaN)
    outlet_idx = argmax(i -> isnan(upstream_area[i]) ? -Inf : upstream_area[i],
                        1:length(upstream_area))

    row = grid.rows[outlet_idx]
    col = grid.cols[outlet_idx]

    return plot_timeseries(pred_grids, true_grids, domain, row, col; path, timestamps)
end

"""
    plot_mb_diagnostics(diags; path=nothing, timestamps=nothing) -> Figure

Plot mass-balance diagnostic terms over the validation rollout timeseries.
`diags` is the NamedTuple returned by `rollout_mb_diagnostics`.

Four rows:
1. Predicted vs true Q (median ± 10th–90th percentile over nodes).
2. Predicted H, true H, and MB-with-true-Q verification line.
3. Flux terms: upstream Q, lateral inflow, predicted Q_out, net flux.
4. h_raw before the ≥0 floor (median) + fraction of nodes where h_raw < 0.

The verification line in row 2 answers whether the equation itself is correct:
if `MB(true Q)` ≈ `true H`, the formulation is sound.
"""
function plot_mb_diagnostics(diags; path=nothing, timestamps=nothing)
    T  = size(diags.pred_q, 2)
    xs = 1:T

    function pct(m, lo=10, hi=90)
        med = Float32[let v = filter(isfinite, view(m, :, t))
                          isempty(v) ? NaN32 : median(v) end for t in 1:T]
        lo_ = Float32[let v = filter(isfinite, view(m, :, t))
                          isempty(v) ? NaN32 : quantile(v, lo/100) end for t in 1:T]
        hi_ = Float32[let v = filter(isfinite, view(m, :, t))
                          isempty(v) ? NaN32 : quantile(v, hi/100) end for t in 1:T]
        med, lo_, hi_
    end

    nanmedian(m) = Float32[let v = filter(isfinite, view(m, :, t))
                               isempty(v) ? NaN32 : median(v) end for t in 1:T]

    function dticks!(ax)
        isnothing(timestamps) && return
        idxs = round.(Int, range(1, T; length = min(6, T)))
        ax.xticks = (idxs, string.(timestamps[idxs]))
        ax.xticklabelrotation = π/4
    end

    fig = Figure(size = (1100, 1050))
    Label(fig[0, 1:2], "Mass balance diagnostics"; fontsize = 14, font = :bold)

    # ── Row 1: Q ──────────────────────────────────────────────────────────────
    ax1 = Axis(fig[1, 1:2]; title = "Discharge Q [m³/s]", ylabel = "m³/s")
    pq_med, pq_lo, pq_hi = pct(diags.pred_q)
    tq_med, tq_lo, tq_hi = pct(diags.true_q)
    band!(ax1, xs, pq_lo, pq_hi; color = (:steelblue, 0.25))
    el_pq = lines!(ax1, xs, pq_med; color = :steelblue)
    band!(ax1, xs, tq_lo, tq_hi; color = (:orangered, 0.25))
    el_tq = lines!(ax1, xs, tq_med; color = :orangered)
    hidexdecorations!(ax1; ticks = false); dticks!(ax1)

    # ── Row 2: H + verification ───────────────────────────────────────────────
    ax2 = Axis(fig[2, 1:2]; title = "Water depth H [m]", ylabel = "m")
    ph_med, ph_lo, ph_hi = pct(diags.pred_h)
    th_med, th_lo, th_hi = pct(diags.true_h)
    mv_med, mv_lo, mv_hi = pct(diags.mb_verify_h)
    band!(ax2, xs, ph_lo, ph_hi; color = (:steelblue,   0.25))
    el_ph = lines!(ax2, xs, ph_med; color = :steelblue)
    band!(ax2, xs, th_lo, th_hi; color = (:orangered,   0.25))
    el_th = lines!(ax2, xs, th_med; color = :orangered)
    band!(ax2, xs, mv_lo, mv_hi; color = (:forestgreen, 0.20))
    el_mv = lines!(ax2, xs, mv_med; color = :forestgreen, linestyle = :dash)
    hidexdecorations!(ax2; ticks = false); dticks!(ax2)

    # ── Row 3: flux terms ─────────────────────────────────────────────────────
    ax3 = Axis(fig[3, 1:2]; title = "Flux terms (median over nodes) [m³/s]",
               ylabel = "m³/s")
    el_uq  = lines!(ax3, xs, nanmedian(diags.upstream_q); color = :steelblue)
    el_iw  = lines!(ax3, xs, nanmedian(diags.inwater);    color = :forestgreen)
    el_qo  = lines!(ax3, xs, nanmedian(diags.pred_q);     color = :orangered)
    el_nf  = lines!(ax3, xs, nanmedian(diags.net_flux);
                    color = :black, linestyle = :dash)
    hlines!(ax3, [0f0]; color = :gray, linestyle = :dot)
    hidexdecorations!(ax3; ticks = false); dticks!(ax3)

    # ── Row 4a: h_raw ─────────────────────────────────────────────────────────
    ax4a = Axis(fig[4, 1]; title = "h_raw before ≥0 floor (median)",
                xlabel = "timestep", ylabel = "m")
    el_hr = lines!(ax4a, xs, nanmedian(diags.h_raw); color = :steelblue)
    hlines!(ax4a, [0f0]; color = :black, linestyle = :dash)
    dticks!(ax4a)

    # ── Row 4b: fraction negative ─────────────────────────────────────────────
    ax4b = Axis(fig[4, 2]; title = "Fraction of nodes with h_raw < 0",
                xlabel = "timestep", ylabel = "fraction")
    frac_neg = Float32[let v = filter(isfinite, view(diags.h_raw, :, t))
                           isempty(v) ? NaN32 : mean(v .< 0) end for t in 1:T]
    el_fn = lines!(ax4b, xs, frac_neg; color = :orangered)
    dticks!(ax4b)

    # ── Figure-level legend ───────────────────────────────────────────────────
    Legend(fig[5, 1:2],
           [el_pq,         el_tq,    el_ph,    el_th,   el_mv,
            el_uq,         el_iw,    el_qo,    el_nf,   el_hr,   el_fn],
           ["pred Q (p10–p90)", "true Q", "pred H (p10–p90)", "true H",
            "MB(true Q)",  "upstream_q", "inwater", "q_out (pred)",
            "net_flux",    "h_raw median", "frac h_raw<0"];
           orientation = :horizontal, tellheight = true, nbanks = 3,
           framevisible = false)

    rowsize!(fig.layout, 5, Auto())
    isnothing(path) || save(path, fig)
    return fig
end
