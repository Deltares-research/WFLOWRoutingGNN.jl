using CairoMakie
import CairoMakie: record

# Derive a `.csv` path from a figure `path` by swapping the extension.
# Returns `nothing` when `path` is `nothing`.
_csv_from_path(path) = isnothing(path) ? nothing : string(splitext(path)[1], ".csv")

# Write named columns to a CSV file (no external dependency; mirrors the manual
# CSV writers in `hparsearch.jl` / `lr_range_test.jl`). `header` is a Vector of
# column names; `columns` is a Vector of equal-length column vectors.
function _write_plot_csv(path::AbstractString, header::Vector{<:AbstractString}, columns::Vector)
    isempty(columns) && return
    n = length(first(columns))
    all(c -> length(c) == n, columns) ||
        throw(ArgumentError("all CSV columns must have the same length"))
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:n
            println(io, join((string(col[i]) for col in columns), ","))
        end
    end
    return path
end

"""
    write_fixed_horizon_anchor_table(path, fh, rmse_anchor, peak_ratio_anchor) -> path

Write the per-anchor fixed-horizon diagnostics to CSV. Rows are anchors and
columns are:
- `anchor_index`            : 1-based anchor id in the batched eval.
- `start_step`              : 1-based start index in the flattened validation
                              timeseries (graph index).
- `start_flow_q_phys`       : anchor start-state basin-mean physical discharge.
- `start_flow_percentile`   : empirical percentile of `start_flow_q_phys`
                              within the validation start-flow distribution.
- `fixed_rmse`              : per-anchor fixed-horizon q RMSE (physical units).
- `peak_ratio`              : per-anchor fixed-horizon peak ratio.
"""
function write_fixed_horizon_anchor_table(path::AbstractString,
                                          fh,
                                          rmse_anchor::AbstractVector,
                                          peak_ratio_anchor::AbstractVector)
    length(rmse_anchor) == fh.B ||
        throw(ArgumentError("rmse_anchor length ($(length(rmse_anchor))) must equal fh.B ($(fh.B))"))
    length(peak_ratio_anchor) == fh.B ||
        throw(ArgumentError("peak_ratio_anchor length ($(length(peak_ratio_anchor))) must equal fh.B ($(fh.B))"))

    header = ["anchor_index", "start_step", "start_flow_q_phys",
              "start_flow_percentile", "fixed_rmse", "peak_ratio"]
    columns = Any[
        collect(1:fh.B),
        fh.anchor_starts,
        fh.start_q_phys,
        fh.start_q_percentile,
        collect(Float32, rmse_anchor),
        collect(Float32, peak_ratio_anchor),
    ]
    _write_plot_csv(path, header, columns)
    return path
end

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
- `csv`           : when `true` (default) and a `path` (or `csv_path`) is
                    available, the plotted per-epoch loss arrays are written to
                    a CSV file alongside the figure, together with any supplied
                    `grad_norm`, `lr` and `steps` diagnostic columns.
- `csv_path`      : optional explicit CSV output path; defaults to `path` with
                    its extension swapped for `.csv`.
- `grad_norm`,
  `lr`, `steps`   : optional per-epoch training-log diagnostics; when supplied
                    they are appended as extra columns to the CSV (not plotted).

Returns the `Figure` object.
"""
function plot_losses(train_rollout, val_rollout, train_1step, val_1step;
                     train_q_1step = nothing,
                     val_q_1step   = nothing,
                     train_h_1step = nothing,
                     val_h_1step   = nothing,
                     grad_norm = nothing,
                     lr        = nothing,
                     steps     = nothing,
                     path = nothing,
                     csv  = true,
                     csv_path = nothing)
    epochs         = 1:length(train_rollout)
    has_components = !isnothing(train_q_1step) &&
                     !isempty(train_q_1step)   &&
                     any(isfinite, train_q_1step)

    fig = Figure(size = (600, has_components ? 700 : 400))

    ax1 = Axis(fig[1, 1];
               title  = "Training losses",
               xlabel = "Epoch",
               ylabel = "MSE",
               yscale = log10)
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
                   ylabel = "MSE",
                   yscale = log10)
        lines!(ax2, epochs, train_q_1step; label = "train Q", color = :steelblue)
        lines!(ax2, epochs, val_q_1step;   label = "val Q",   color = :steelblue,
               linestyle = :dash)
        lines!(ax2, epochs, train_h_1step; label = "train H", color = :forestgreen)
        lines!(ax2, epochs, val_h_1step;   label = "val H",   color = :forestgreen,
               linestyle = :dash)
        axislegend(ax2; position = :rt)
    end

    isnothing(path) || save(path, fig)

    # Write the plotted per-epoch arrays to CSV.
    csv_out = isnothing(csv_path) ? _csv_from_path(path) : csv_path
    if csv && !isnothing(csv_out)
        header  = ["epoch", "train_rollout", "val_rollout", "train_1step", "val_1step"]
        columns = Any[collect(epochs), train_rollout, val_rollout, train_1step, val_1step]
        if has_components
            append!(header, ["train_q_1step", "val_q_1step", "train_h_1step", "val_h_1step"])
            push!(columns, train_q_1step, val_q_1step, train_h_1step, val_h_1step)
        end
        # Extra per-epoch training-log diagnostics (CSV-only, not plotted here).
        if !isnothing(grad_norm) && !isempty(grad_norm)
            push!(header, "grad_norm"); push!(columns, grad_norm)
        end
        if !isnothing(lr) && !isempty(lr)
            push!(header, "lr"); push!(columns, lr)
        end
        if !isnothing(steps) && !isempty(steps)
            push!(header, "steps"); push!(columns, steps)
        end
        _write_plot_csv(csv_out, header, columns)
    end

    return fig
end

"""
    plot_amplification(train_amp, val_amp; mb_gain = nothing, path = nothing,
                       csv = true, csv_path = nothing) -> Figure

Plot the per-epoch **q→h error amplification** produced by
[`mb_amplification`](@ref): the realised ratio `RMS(h_pred − h_ref) /
RMS(q_pred − q_true)` in normalised units, where `h_ref = MB(q_true, …)` is the
counterfactual depth for a perfect discharge. Values above `1` (dashed grey
reference) mean a one-step discharge error is *magnified* into a depth error by
the hard mass-balance decoder — the mechanism behind rollout h-overshoots.

If `mb_gain` is supplied, the analytic self-gain `θ·dt·σ_q/σ_h` is drawn as a
reference line (constant unless θ changes during training).

Writes the plotted arrays to CSV alongside the figure when `csv` is `true`.
Returns the `Figure`.
"""
function plot_amplification(train_amp, val_amp;
                            mb_gain  = nothing,
                            path     = nothing,
                            csv      = true,
                            csv_path = nothing)
    epochs = 1:length(train_amp)
    fig = Figure(size = (600, 400))
    ax  = Axis(fig[1, 1];
               title  = "Q→H error amplification through the mass balance",
               xlabel = "Epoch",
               ylabel = "RMS(Δh_norm from Q err) / RMS(Δq_norm)")
    lines!(ax, epochs, train_amp; label = "train", color = :steelblue)
    lines!(ax, epochs, val_amp;   label = "val",   color = :steelblue, linestyle = :dash)
    hlines!(ax, [1f0]; color = :gray, linestyle = :dot)  # amplification threshold
    if !isnothing(mb_gain) && !isempty(mb_gain) && any(isfinite, mb_gain)
        lines!(ax, epochs, mb_gain; label = "analytic gain θ·dt·σq/σh",
               color = :orangered, linestyle = :dashdot)
    end
    axislegend(ax; position = :rt)

    isnothing(path) || save(path, fig)

    csv_out = isnothing(csv_path) ? _csv_from_path(path) : csv_path
    if csv && !isnothing(csv_out)
        header  = ["epoch", "train_amp", "val_amp"]
        columns = Any[collect(epochs), train_amp, val_amp]
        if !isnothing(mb_gain) && !isempty(mb_gain)
            push!(header, "mb_gain")
            push!(columns, mb_gain)
        end
        _write_plot_csv(csv_out, header, columns)
    end

    return fig
end

"""
    plot_fixed_horizon(val_fixed_rmse, val_peak_ratio;
                       val_peak_ratio_frac_gt2 = nothing,
                       val_fixed_rmse_highflow = nothing,
                       horizon = nothing, path = nothing, csv = true,
                       csv_path = nothing) -> Figure

Plot the per-epoch **fixed-horizon** validation metrics from `train_model!`: the
constant-length autoregressive discharge RMSE (physical units) and the peak
amplification ratio `max|q_pred| / max|q_truth|`. Unlike `val_rollout`, whose
horizon grows with the curriculum, these are directly comparable epoch-to-epoch
and drive early stopping.

Two stacked panels are drawn:
- Top    : fixed-horizon discharge RMSE (lower is better).
- Bottom : peak ratio, with a dashed reference line at `1` (values above `1`
           indicate the rollout over-amplifies the hydrograph peak).

When provided, `val_peak_ratio_frac_gt2` and `val_fixed_rmse_highflow` are
appended as CSV-only columns (not drawn) so anti-masking fixed-horizon guard
signals are persisted per epoch.

Writes the plotted arrays to CSV alongside the figure when `csv` is `true`.
Returns the `Figure`.
"""
function plot_fixed_horizon(val_fixed_rmse, val_peak_ratio;
                            val_peak_ratio_frac_gt2 = nothing,
                            val_fixed_rmse_highflow = nothing,
                            horizon  = nothing,
                            path     = nothing,
                            csv      = true,
                            csv_path = nothing)
    epochs = 1:length(val_fixed_rmse)
    htxt   = isnothing(horizon) ? "" : " (H=$(horizon))"
    fig = Figure(size = (600, 600))

    ax1 = Axis(fig[1, 1];
               title  = "Fixed-horizon validation discharge RMSE$(htxt)",
               xlabel = "Epoch",
               ylabel = "RMSE(q) [m³/s]")
    lines!(ax1, epochs, val_fixed_rmse; color = :steelblue)

    ax2 = Axis(fig[2, 1];
               title  = "Fixed-horizon peak ratio$(htxt)",
               xlabel = "Epoch",
               ylabel = "max|q_pred| / max|q_truth|")
    lines!(ax2, epochs, val_peak_ratio; color = :orangered)
    hlines!(ax2, [1f0]; color = :gray, linestyle = :dash)

    isnothing(path) || save(path, fig)

    csv_out = isnothing(csv_path) ? _csv_from_path(path) : csv_path
    if csv && !isnothing(csv_out)
        header  = ["epoch", "val_fixed_rmse", "val_peak_ratio"]
        columns = Any[collect(epochs), val_fixed_rmse, val_peak_ratio]
        if !isnothing(val_peak_ratio_frac_gt2) && !isempty(val_peak_ratio_frac_gt2)
            push!(header, "val_peak_ratio_frac_gt2")
            push!(columns, val_peak_ratio_frac_gt2)
        end
        if !isnothing(val_fixed_rmse_highflow) && !isempty(val_fixed_rmse_highflow)
            push!(header, "val_fixed_rmse_highflow")
            push!(columns, val_fixed_rmse_highflow)
        end
        _write_plot_csv(csv_out, header, columns)
    end

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
- `csv`        : when `true` (default) and a `path` (or `csv_path`) is available,
                 the plotted timeseries (truth, prediction and absolute error per
                 state variable) are written to a CSV file alongside the figure.
- `csv_path`   : optional explicit CSV output path; defaults to `path` with its
                 extension swapped for `.csv`.

Returns the `Figure` object.
"""
function plot_timeseries(
        pred_grids :: Dict{String, Array{Float32,3}},
        true_grids :: Dict{String, Array{Float32,3}},
        domain     :: String,
        row        :: Int,
        col        :: Int;
        path       = nothing,
        timestamps = nothing,
        csv        = true,
        csv_path   = nothing)

    state_vars = DOMAIN_VARS[domain]["state"]
    isempty(state_vars) && throw(ArgumentError("domain \"$domain\" has no state variables"))
    nvars = length(state_vars)
    T     = size(first(values(true_grids)), 3)
    ts    = 1:T

    all(isnan, true_grids[state_vars[1]][row, col, :]) &&
        @warn "Cell ($row, $col) is inactive (all NaN); plot will be empty"

    fig = Figure(size = (1600, 300 * nvars))

    # Accumulate the plotted series for optional CSV export. First column is the
    # time axis; then truth / prediction / absolute error per state variable.
    csv_header  = [isnothing(timestamps) ? "timestep" : "time"]
    csv_columns = Any[isnothing(timestamps) ? collect(ts) : string.(collect(timestamps))]

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

        # --- RMSE over time panel ---
        abserr = abs.(pred .- truth)
        rmse_val = let v = filter(isfinite, abserr)
            isempty(v) ? NaN32 : Float32(sqrt(mean(v .^ 2)))
        end

        append!(csv_header, ["$(vname)_truth", "$(vname)_pred", "$(vname)_abserr"])
        push!(csv_columns, collect(truth), collect(pred), collect(abserr))

        ax_err = Axis(fig[vi, 3];
                      title              = "$vname  —  absolute error",
                      xlabel             = isnothing(timestamps) ? "Timestep" : "Time",
                      ylabel             = "|pred − truth|",
                      xticklabelrotation = isnothing(timestamps) ? 0.0 : π/4,
                      xticklabelalign    = isnothing(timestamps) ?
                                           (:center, :top) : (:right, :top))
        if !isnothing(timestamps)
            n_ticks  = min(T, 6)
            tick_idx = unique(round.(Int, range(1, T; length = n_ticks)))
            ax_err.xticks = (tick_idx, string.(timestamps[tick_idx]))
        end
        lines!(ax_err, ts, abserr; color = :purple)
        if isfinite(rmse_val)
            hlines!(ax_err, [rmse_val]; color = :black, linestyle = :dash, linewidth = 1.5)
            valid_err = filter(isfinite, abserr)
            ypos = isempty(valid_err) ? rmse_val : maximum(valid_err) * 0.97
            text!(ax_err, T * 0.02f0, ypos;
                  text     = "RMSE = $(round(rmse_val; sigdigits = 4))",
                  fontsize = 11,
                  align    = (:left, :top))
        end
    end

    isnothing(path) || save(path, fig)

    csv_out = isnothing(csv_path) ? _csv_from_path(path) : csv_path
    (csv && !isnothing(csv_out)) && _write_plot_csv(csv_out, csv_header, csv_columns)

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
- `csv`           : when `true` (default) and a `path` (or `csv_path`) is
                    available, the plotted timeseries are written to CSV
                    (forwarded to `plot_timeseries`).
- `csv_path`      : optional explicit CSV output path.

Returns the `Figure` object.
"""
function plot_downstream_timeseries(
        pred_grids    :: Dict{String, Array{Float32,3}},
        true_grids    :: Dict{String, Array{Float32,3}},
        domain        :: String,
        grid          :: NamedTuple,
        upstream_area :: AbstractVector{<:Real};
        path          = nothing,
        timestamps    = nothing,
        csv           = true,
        csv_path      = nothing)

    # Most downstream node = largest upstream catchment area (ignore NaN)
    outlet_idx = argmax(i -> isnan(upstream_area[i]) ? -Inf : upstream_area[i],
                        1:length(upstream_area))

    row = grid.rows[outlet_idx]
    col = grid.cols[outlet_idx]

    return plot_timeseries(pred_grids, true_grids, domain, row, col;
                           path, timestamps, csv, csv_path)
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

When `csv` is `true` (default) and a `path` (or `csv_path`) is available, the
plotted per-timestep reductions — the median (and p10/p90 for the Q/H series) of
every diagnostic term — are written to a companion CSV alongside the figure.
"""
function plot_mb_diagnostics(diags; path=nothing, timestamps=nothing,
                             csv=true, csv_path=nothing)
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

    fig = Figure(size = (1200, 1050))
    Label(fig[0, 1], "Mass balance diagnostics"; fontsize = 14, font = :bold)

    # ── Row 1: Q ──────────────────────────────────────────────────────────────
    ax1 = Axis(fig[1, 1]; title = "Discharge Q [m³/s]", ylabel = "m³/s")
    pq_med, pq_lo, pq_hi = pct(diags.pred_q)
    tq_med, tq_lo, tq_hi = pct(diags.true_q)
    band!(ax1, xs, pq_lo, pq_hi; color = (:steelblue, 0.25))
    el_pq = lines!(ax1, xs, pq_med; color = :steelblue)
    band!(ax1, xs, tq_lo, tq_hi; color = (:orangered, 0.25))
    el_tq = lines!(ax1, xs, tq_med; color = :orangered)
    hidexdecorations!(ax1; ticks = false); dticks!(ax1)
    Legend(fig[1, 2], [el_pq, el_tq], ["pred Q (p10–p90)", "true Q"];
           framevisible = false, tellwidth = true)

    # ── Row 2: H + verification ───────────────────────────────────────────────
    ax2 = Axis(fig[2, 1]; title = "Water depth H [m]", ylabel = "m")
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
    Legend(fig[2, 2], [el_ph, el_th, el_mv], ["pred H (p10–p90)", "true H", "MB(true Q)"];
           framevisible = false, tellwidth = true)

    # ── Row 3: flux terms ─────────────────────────────────────────────────────
    ax3 = Axis(fig[3, 1]; title = "Flux terms (median over nodes) [m³/s]",
               ylabel = "m³/s")
    el_uq = lines!(ax3, xs, nanmedian(diags.upstream_q); color = :steelblue)
    el_iw = lines!(ax3, xs, nanmedian(diags.inwater);    color = :forestgreen)
    el_qo = lines!(ax3, xs, nanmedian(diags.pred_q);     color = :orangered)
    el_nf = lines!(ax3, xs, nanmedian(diags.net_flux);   color = :black, linestyle = :dash)
    hlines!(ax3, [0f0]; color = :gray, linestyle = :dot)
    hidexdecorations!(ax3; ticks = false); dticks!(ax3)
    Legend(fig[3, 2], [el_uq, el_iw, el_qo, el_nf],
           ["upstream_q", "inwater", "q_out (pred)", "net_flux"];
           framevisible = false, tellwidth = true)

    # ── Row 4a: h_raw ─────────────────────────────────────────────────────────
    ax4a = Axis(fig[4, 1]; title = "h_raw before ≥0 floor",
                xlabel = "timestep", ylabel = "m")
    el_hr = lines!(ax4a, xs, nanmedian(diags.h_raw); color = :steelblue)
    hlines!(ax4a, [0f0]; color = :black, linestyle = :dash)
    frac_neg = Float32[let v = filter(isfinite, view(diags.h_raw, :, t))
                           isempty(v) ? NaN32 : mean(v .< 0) end for t in 1:T]
    ax4b = Axis(fig[4, 1]; ylabel = "frac h_raw<0",
                yaxisposition = :right, yticklabelcolor = :orangered)
    hidespines!(ax4b); hidexdecorations!(ax4b)
    el_fn = lines!(ax4b, xs, frac_neg; color = :orangered)
    dticks!(ax4a)
    Legend(fig[4, 2], [el_hr, el_fn], ["h_raw median [m]", "frac h_raw<0"];
           framevisible = false, tellwidth = true)

    # Fix legend column width so all panels share the same plot area
    colsize!(fig.layout, 2, Fixed(160))
    isnothing(path) || save(path, fig)

    # Companion CSV: per-timestep median (and p10/p90 for the Q/H series)
    # reductions of every plotted diagnostic term.
    csv_out = isnothing(csv_path) ? _csv_from_path(path) : csv_path
    if csv && !isnothing(csv_out)
        header  = String["timestep"]
        columns = Any[collect(xs)]
        if !isnothing(timestamps) && length(timestamps) >= T
            push!(header, "timestamp"); push!(columns, string.(timestamps[1:T]))
        end
        for (name, series) in (("pred_q", diags.pred_q),
                               ("true_q", diags.true_q),
                               ("pred_h", diags.pred_h),
                               ("true_h", diags.true_h),
                               ("mb_verify_h", diags.mb_verify_h))
            med, lo_, hi_ = pct(series)
            append!(header, ["$(name)_med", "$(name)_p10", "$(name)_p90"])
            push!(columns, med, lo_, hi_)
        end
        append!(header, ["upstream_q_med", "inwater_med", "net_flux_med",
                         "h_raw_med", "frac_h_raw_neg"])
        push!(columns, nanmedian(diags.upstream_q), nanmedian(diags.inwater),
                       nanmedian(diags.net_flux), nanmedian(diags.h_raw), frac_neg)
        _write_plot_csv(csv_out, header, columns)
    end

    return fig
end

# Largest colour-range width that stays representable in Float32. CairoMakie
# scales each pixel as (x - lo)/(hi - lo) in Float32; if (hi - lo) overflows to
# Inf the scaling yields NaN and the colormap lookup errors. Keeping the width
# to floatmax/4 leaves ample headroom.
const _PLOT_MAX_WIDTH = floatmax(Float32) / 4

# Coerce a (lo, hi) colour range to a finite, non-degenerate Float32 pair whose
# width cannot overflow Float32.
function _bound_range(lo::Real, hi::Real)
    lo32, hi32 = Float32(lo), Float32(hi)
    (isfinite(lo32) && isfinite(hi32)) || return (-1f0, 1f0)
    lo32 == hi32 && return (lo32 - 1f0, hi32 + 1f0)
    if !isfinite(hi32 - lo32) || (hi32 - lo32) > _PLOT_MAX_WIDTH
        mid  = clamp((lo32 + hi32) / 2, -_PLOT_MAX_WIDTH, _PLOT_MAX_WIDTH)
        half = _PLOT_MAX_WIDTH / 2
        return (mid - half, mid + half)
    end
    return (lo32, hi32)
end

# Prepare a metric map for heatmapping: non-finite entries become NaN (drawn with
# nan_color) and finite entries are clamped into [lo, hi] so the Float32 colour
# scaling in CairoMakie can never overflow to NaN/Inf.
function _clamp_for_plot(m::AbstractMatrix, lo::Real, hi::Real)
    lo32, hi32 = Float32(lo), Float32(hi)
    return map(m) do x
        xf = Float32(x)
        isfinite(xf) ? clamp(xf, lo32, hi32) : NaN32
    end
end

# Symmetric colour range about a centre from the finite values of a map.
function _sym_range(m::AbstractMatrix, centre::Real)
    v = filter(isfinite, vec(m))
    isempty(v) && return (Float32(centre) - 1f0, Float32(centre) + 1f0)
    r = maximum(abs.(v .- centre))
    r = r == 0 ? 1f0 : Float32(r)
    return _bound_range(Float32(centre) - r, Float32(centre) + r)
end

# Colour range spanning the finite values of a map. Guards against empty /
# all-non-finite maps and degenerate (constant) maps, either of which would make
# Makie's automatic colorrange include Inf/NaN and crash the colorbar tick
# formatter.
function _finite_range(m::AbstractMatrix)
    v = filter(isfinite, vec(m))
    isempty(v) && return (0f0, 1f0)
    return _bound_range(minimum(v), maximum(v))
end

"""
    plot_spatial_metrics(metrics, domain; path = nothing) -> Figure

Render the per-cell error maps from [`spatial_error_metrics`](@ref) as heatmaps,
one row per state variable. Columns: RMSE, bias, overprediction frequency, peak
error, normalised bias, NSE. Signed metrics use a diverging colormap centred on
their neutral value (0, or 0.5 for `overpred_freq`).
"""
function plot_spatial_metrics(metrics ::Dict{String, Dict{String, Matrix{Float32}}},
                              domain  ::String;
                              path    = nothing)

    state_vars = [v for v in DOMAIN_VARS[domain]["state"] if haskey(metrics, v)]
    isempty(state_vars) && throw(ArgumentError("no state variables present in metrics"))

    # (metric key, title, colormap, centre-or-nothing)
    panels = [("rmse",          "RMSE",              :viridis, nothing),
              ("bias",          "bias (pred−truth)", :RdBu,    0.0),
              ("overpred_freq", "overpred. freq",    :RdBu,    0.5),
              ("peak_err",      "peak error",        :RdBu,    0.0),
              ("peak_lag",      "peak lag [Δt]",     :RdBu,    0.0),
              ("nbias",         "bias / σ(truth)",   :RdBu,    0.0),
              ("nse",           "NSE",               :viridis, nothing)]

    nrows = length(state_vars)
    ncols = length(panels)
    fig = Figure(size = (330 * ncols, 300 * nrows + 40))
    Label(fig[0, 1:(2*ncols)], "Spatial error metrics"; fontsize = 15, font = :bold)

    for (ri, vname) in enumerate(state_vars), (ci, (key, ttl, cmap, centre)) in enumerate(panels)
        m = get(metrics[vname], key, nothing)
        ax = Axis(fig[ri, 2ci - 1];
                  title  = "$vname — $ttl",
                  aspect = DataAspect())
        hidedecorations!(ax)
        if isnothing(m)
            continue
        end
        if key == "nse"
            v = filter(isfinite, vec(m))
            lo = isempty(v) ? -1f0 : max(minimum(v), -1f0)  # clamp NSE floor for contrast
            lo = min(lo, 1f0 - eps(Float32))                # keep the range non-degenerate
            crange = _bound_range(lo, 1f0)
        elseif isnothing(centre)
            crange = _finite_range(m)
        else
            crange = _sym_range(m, centre)
        end
        hm = heatmap!(ax, _clamp_for_plot(m, crange[1], crange[2]);
                      colorrange = crange, colormap = cmap)
        Colorbar(fig[ri, 2ci], hm)
    end

    isnothing(path) || save(path, fig)
    return fig
end

"""
    plot_overprediction_vs_ramp(ramp; path = nothing, csv = true, csv_path = nothing)
        -> Figure

Visualise the [`overprediction_vs_ramp`](@ref) result: whether discharge
overprediction grows with the true Q ramp rate `g = (Qₜ−Qₜ₋₁)/Qₜ`.

Panels: (1) subsampled scatter of error `e` vs `g` with zero reference lines and
the Pearson/Spearman coefficients; (2) mean overprediction per ramp class as a
bar chart with counts; (3) a per-cell map of the `e`–`g` correlation. The
per-class summary is also written to CSV.
"""
function plot_overprediction_vs_ramp(ramp; path = nothing, csv = true, csv_path = nothing)
    fig = Figure(size = (1500, 480))

    # ── Panel 1: scatter e vs g ───────────────────────────────────────────────
    ax1 = Axis(fig[1, 1];
               title  = "Q error vs ramp rate  (Pearson e~g = $(round(ramp.pearson_e_g; digits=3)), " *
                        "Spearman = $(round(ramp.spearman_e_g; digits=3)))",
               xlabel = "true ramp g = (Qₜ−Qₜ₋₁)/Qₜ",
               ylabel = "error e = pred − truth  [m³/s]")
    if !isempty(ramp.scatter_g)
        scatter!(ax1, ramp.scatter_g, ramp.scatter_e;
                 color = (:steelblue, 0.15), markersize = 3)
    end
    hlines!(ax1, [0f0]; color = :black, linestyle = :dash)
    vlines!(ax1, [0f0]; color = :gray,  linestyle = :dot)

    # ── Panel 2: mean overprediction per ramp class ───────────────────────────
    nb  = length(ramp.bin_labels)
    xs  = 1:nb
    ax2 = Axis(fig[1, 2];
               title  = "mean overprediction per ramp class",
               ylabel = "mean max(e,0)  [m³/s]",
               xticks = (collect(xs), ramp.bin_labels),
               xticklabelrotation = π/5)
    vals = [isfinite(v) ? v : 0f0 for v in ramp.bin_mean_overpred]
    barplot!(ax2, collect(xs), vals; color = :orangered)
    for b in xs
        n = ramp.bin_n[b]
        text!(ax2, b, vals[b]; text = "n=$n", align = (:center, :bottom),
              fontsize = 10, offset = (0, 2))
    end

    # ── Panel 3: per-cell e–g correlation map ─────────────────────────────────
    ax3 = Axis(fig[1, 3]; title = "per-cell corr(e, g)", aspect = DataAspect())
    hidedecorations!(ax3)
    hm = heatmap!(ax3, ramp.corr_map; colorrange = (-1f0, 1f0), colormap = :RdBu)
    Colorbar(fig[1, 4], hm; label = "corr(e, g)")

    isnothing(path) || save(path, fig)

    # ── CSV of per-class summary ──────────────────────────────────────────────
    if csv
        out_csv = isnothing(csv_path) ? _csv_from_path(path) : csv_path
        if !isnothing(out_csv)
            _write_plot_csv(out_csv,
                ["ramp_class", "n", "mean_err", "mean_overpred", "frac_over"],
                Any[ramp.bin_labels, ramp.bin_n, ramp.bin_mean_err,
                    ramp.bin_mean_overpred, ramp.bin_frac_over])
        end
    end

    return fig
end
