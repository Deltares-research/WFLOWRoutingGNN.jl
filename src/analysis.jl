import JLD2

# ---------------------------------------------------------------------------
# Pipeline stage 4 (analyze)
#
#   analyze_run(prep, ms, ts; dirs, ...) -> summaries
#
# Reads the gridded `<split>_pred.nc` / `<split>_true.nc` written by `infer_run`
# (src/pipeline.jl), gathers the river cells back into node order with the
# preprocessing-owned node↔cell mapping (`prep.grid`) and produces every metric,
# plot and CSV that used to be fused into the rollout loop. It never runs the
# model for the standard diagnostics. The kinematic-wave mass-balance residual is
# recomputed layer-free from the predicted q/h fields for every model; the
# layer-internal mass-balance diagnostics (floor fraction, stiffness, volume
# budget) need the trained mass-balance layer and are skipped otherwise.
# ---------------------------------------------------------------------------

"""
    analyze_run(prep, ms, ts; dirs, model = nothing, train_summary = nothing,
                infer = nothing, metrics_toml = true)
        -> (; spatial_summary, ramp, river_q_perf, volume_budget, mb_floor, mb_stiffness,
              mb_residual)

Stage 4 of the pipeline. Analyses the validation split of a run from its gridded
NetCDF outputs (`<dirs.output>/val_pred.nc`, `val_true.nc`) and writes

    <dirs.plots>/    validation.mp4, downstream_timeseries.png, spatial_metrics.png,
                     q_overprediction_vs_ramp.png, [mb_diagnostics.png, volume_budget.png],
                     [validation_daterange.mp4, downstream_timeseries_daterange.png]
    <dirs.metrics>/  downstream_timeseries.csv, spatial_metrics.csv,
                     q_overprediction_vs_ramp.csv, mb_residual.csv, mb_residual_by_node.csv,
                     [mb_*.csv, volume_budget.csv],
                     [downstream_timeseries_daterange.csv], metrics.toml
    <dirs.output>/   spatial_metrics.nc

Node-order arrays (for [`river_q_performance_metrics`](@ref)) are recovered with
[`gather_nodes`](@ref) and the node↔cell mapping `prep.grid`; the gauge/network
inset uses `prep.edge_src`/`prep.edge_tgt` and `prep.postscale`.

Keyword arguments:
- `dirs`          : value with `output`, `plots`, `metrics` fields (e.g. `_run_dirs(ds)`;
                    `model` as well when `metrics_toml = true` and `train_summary` is
                    not given).
- `model`         : trained model, only needed for the layer-internal mass-balance
                    diagnostics (mass-balance models; a no-op for `NoMassBalance`).
                    Skipped, with a notice, when `nothing`. May be on any device.
- `train_summary` : `(; losses, train_duration_s, n_params)` as returned by
                    [`train_run`](@ref); loaded from `<dirs.model>/train_summary.jld2`
                    when `nothing` and `metrics_toml = true`.
- `infer`         : result of [`infer_run`](@ref), for the validation rollout timing.
- `metrics_toml`  : write the per-run `metrics.toml` summary (default). Periodic
                    checkpoint evaluations pass `false`.

Returns the summaries that feed `metrics.toml` (each `nothing` when not
applicable): `spatial_summary` from [`spatial_metric_summary`](@ref), `ramp` from
[`overprediction_vs_ramp`](@ref), `river_q_perf` from
[`river_q_performance_metrics`](@ref), `volume_budget` from
[`volume_budget_diagnostics`](@ref), `mb_floor` / `mb_stiffness` from the
mass-balance layer and `mb_residual` (`(; theta, pred, reference)` scalar summaries)
from [`mb_residual_diagnostics`](@ref).
"""
function analyze_run(prep::PrepBundle, ms::ModelSettings, ts::TrainSettings;
                     dirs, model = nothing, train_summary = nothing,
                     infer = nothing, metrics_toml::Bool = true)
    foreach(mkpath, (dirs.output, dirs.plots, dirs.metrics))
    schema     = load_schema(prep.wflow_schema)
    domain     = ms.domain
    state_vars = DOMAIN_VARS[domain]["state"]
    grid       = prep.grid

    pred_path = joinpath(dirs.output, "val_pred.nc")
    true_path = joinpath(dirs.output, "val_true.nc")
    isfile(pred_path) && isfile(true_path) ||
        throw(ArgumentError("missing $pred_path / $true_path (run infer_run first)"))

    p_grids     = load_regrid_nc(pred_path, state_vars)
    t_grids     = load_regrid_nc(true_path, state_vars)
    split_times = load_regrid_times(pred_path)
    p_states    = gather_nodes(p_grids, grid, domain)
    t_states    = gather_nodes(t_grids, grid, domain)

    spatial_summary       = nothing
    ramp_summary          = nothing
    river_q_perf          = nothing
    volume_budget_summary = nothing
    mb_floor_summary      = nothing
    mb_stiffness_summary  = nothing

    plot_validation_movie(p_grids, t_grids, domain;
                          path       = joinpath(dirs.plots, "validation.mp4"),
                          framerate  = 10,
                          timestamps = split_times)

    plot_downstream_timeseries(p_grids, t_grids, domain, grid,
                               prep.postscale["river_q"];  # upstream area per node
                               path       = joinpath(dirs.plots, "downstream_timeseries.png"),
                               upstream_points = ts.upstream_points,
                               edge_sources = prep.edge_src,
                               edge_targets = prep.edge_tgt,
                               timestamps = split_times,
                               csv_path   = joinpath(dirs.metrics, "downstream_timeseries.csv"))

    # Spatial performance: per-cell error maps + Q overprediction vs ramp
    sp_metrics = spatial_error_metrics(p_grids, t_grids, domain)
    write_spatial_metrics_to_netcdf(sp_metrics, prep.staticmaps_file,
                                    joinpath(dirs.output, "spatial_metrics.nc"); schema)
    write_spatial_metrics_to_csv(sp_metrics, joinpath(dirs.metrics, "spatial_metrics.csv"))
    plot_spatial_metrics(sp_metrics, domain;
                         path = joinpath(dirs.plots, "spatial_metrics.png"))
    spatial_summary = spatial_metric_summary(sp_metrics)

    if "river_q" in state_vars
        ramp = overprediction_vs_ramp(p_grids, t_grids)
        @info "Q overprediction vs ramp: Pearson(e,g)=$(round(ramp.pearson_e_g; digits=3)) " *
              "Spearman=$(round(ramp.spearman_e_g; digits=3)) over $(ramp.n) (cell,step) pairs"
        plot_overprediction_vs_ramp(ramp;
            path     = joinpath(dirs.plots, "q_overprediction_vs_ramp.png"),
            csv_path = joinpath(dirs.metrics, "q_overprediction_vs_ramp.csv"))
        ramp_summary = ramp

        qi = findfirst(==("river_q"), state_vars)
        river_q_perf = river_q_performance_metrics(
            p_states[qi, :, :], t_states[qi, :, :], prep.postscale["river_q"])
        @info "River q performance — pooled: KGE=$(round(river_q_perf.pooled.kge; digits=3)) " *
              "MAE=$(round(river_q_perf.pooled.mae; digits=3))  |  " *
              "outlet gauge (node $(river_q_perf.gauge.outlet_idx)): " *
              "KGE=$(round(river_q_perf.gauge.kge; digits=3)) MAE=$(round(river_q_perf.gauge.mae; digits=3))"
    end

    # Layer-free mass-balance residual: recomputed from the predicted/true q & h
    # fields, so it applies to any model (with or without a mass-balance layer).
    mb_residual_summary = nothing
    if all(in(state_vars), ("river_q", "river_h")) &&
       "river_inwater" in DOMAIN_VARS[domain]["forcing"]
        mb_residual_summary = _analyze_mb_residual(prep, ms, p_states, t_states,
                                                   state_vars, dirs, split_times)
    end

    # Layer-internal mass-balance diagnostics: dispatched on the formulation
    # (`NoMassBalance` has none) and need the trained layer.
    if model === nothing
        mb_formulation_kind(ms) == :none ||
            @info "No model supplied: skipping layer-internal mass-balance diagnostics"
    else
        layer = layer_mb_analysis(model.mass_balance, model, prep, dirs, split_times)
        if layer !== nothing
            volume_budget_summary = layer.volume_budget
            mb_floor_summary      = layer.mb_floor
            mb_stiffness_summary  = layer.mb_stiffness
        end
    end

    # Optional date-range rollout on the validation split (written by infer_run)
    dr_pred_path = joinpath(dirs.output, "val_daterange_pred.nc")
    dr_true_path = joinpath(dirs.output, "val_daterange_true.nc")
    if !isnothing(ts.val_daterange) && isfile(dr_pred_path) && isfile(dr_true_path)
        dr_p_grids    = load_regrid_nc(dr_pred_path, state_vars)
        dr_t_grids    = load_regrid_nc(dr_true_path, state_vars)
        dr_pred_times = load_regrid_times(dr_pred_path)

        plot_validation_movie(dr_p_grids, dr_t_grids, domain;
                              path       = joinpath(dirs.plots, "validation_daterange.mp4"),
                              framerate  = 10,
                              timestamps = dr_pred_times)

        plot_downstream_timeseries(dr_p_grids, dr_t_grids, domain, grid,
                                   prep.postscale["river_q"];
                                   path       = joinpath(dirs.plots, "downstream_timeseries_daterange.png"),
                                   upstream_points = ts.upstream_points,
                                   edge_sources = prep.edge_src,
                                   edge_targets = prep.edge_tgt,
                                   timestamps = dr_pred_times,
                                   csv_path   = joinpath(dirs.metrics, "downstream_timeseries_daterange.csv"))
    end

    out = (; spatial_summary, ramp = ramp_summary, river_q_perf,
           volume_budget = volume_budget_summary,
           mb_floor = mb_floor_summary, mb_stiffness = mb_stiffness_summary,
           mb_residual = mb_residual_summary)

    if metrics_toml
        summary = train_summary === nothing ? load_train_summary(dirs.model) : train_summary
        # Scalar per-run summary (token-cheap single file for evaluation agents):
        # final/best training-history metrics + aggregated spatial-error and
        # overprediction-vs-ramp diagnostics, so no CSV/NetCDF parsing is needed.
        write_run_metrics_toml(joinpath(dirs.metrics, "metrics.toml"), summary.losses, ts,
                               (; n_params                = summary.n_params,
                                  train_duration_s        = summary.train_duration_s,
                                  val_rollout_duration_s  = infer === nothing ? nothing :
                                                            infer.val_rollout_duration_s,
                                  val_n_timesteps         = size(p_states, 3)),
                               out.spatial_summary, out.ramp, out.river_q_perf,
                               out.volume_budget, out.mb_floor, out.mb_stiffness,
                               out.mb_residual)
    end
    return out
end

# ---------------------------------------------------------------------------
# Mass-balance diagnostics
# ---------------------------------------------------------------------------

"""
    layer_mb_analysis(mb, model, prep, dirs, split_times)
        -> Union{Nothing, NamedTuple{(:volume_budget, :mb_floor, :mb_stiffness)}}

Layer-internal mass-balance diagnostics (floor fraction, stiffness, volume budget
plus their plots/CSVs), dispatched on the mass-balance formulation `mb`
(`model.mass_balance`). Returns `nothing` for [`NoMassBalance`](@ref): there is no
layer to inspect, and the layer-free residual of [`mb_residual_diagnostics`](@ref)
covers such models.
"""
layer_mb_analysis(::NoMassBalance, model, prep, dirs, split_times) = nothing

function layer_mb_analysis(::ThetaKinematicWave, model, prep, dirs, split_times)
    @info "Computing mass balance diagnostics on validation split"
    cpu_model = Flux.cpu(model)
    mb_diags = rollout_mb_diagnostics(cpu_model, prep.dataset.val, prep.static_arr)
    plot_mb_diagnostics(mb_diags;
                        path       = joinpath(dirs.plots, "mb_diagnostics.png"),
                        timestamps = split_times,
                        csv_path   = joinpath(dirs.metrics, "mb_diagnostics.csv"))
    _write_mb_floor_fraction_by_node(
        joinpath(dirs.metrics, "mb_floor_fraction_by_node.csv"),
        mb_diags.frac_h_raw_neg_node,
        mb_diags.n_valid_h_raw_node,
    )
    mb_floor = _mb_floor_summary(mb_diags)
    _write_mb_stiffness_by_node(
        joinpath(dirs.metrics, "mb_stiffness_by_node.csv"),
        cpu_model.mass_balance,
    )
    mb_stiffness = _mb_stiffness_summary(cpu_model.mass_balance)

    volume_budget = volume_budget_diagnostics(
        mb_diags;
        postscale_q = cpu_model.mass_balance.postscale_q,
        postscale_h = cpu_model.mass_balance.postscale_h,
        dt = cpu_model.mass_balance.dt,
        upstream_area = prep.postscale["river_q"])
    plot_volume_budget(volume_budget;
                       path       = joinpath(dirs.plots, "volume_budget.png"),
                       timestamps = split_times,
                       csv_path   = joinpath(dirs.metrics, "volume_budget.csv"))
    @info "Volume budget: outlet=$(volume_budget.outlet_idx) " *
          "pbias=$(round(volume_budget.volume_pbias; digits=3))% " *
          "drift=$(round(volume_budget.volume_drift; digits=3)) m^3/step"
    return (; volume_budget, mb_floor, mb_stiffness)
end

# Physical lateral inflow [m³/s] of the validation split, `(n_nodes, T)` aligned
# with the frames of `val_pred.nc` (frame t ↔ flattened graph t+1; the first graph is
# the initial condition). Undoes the per-node z-score and the area normalisation.
function _val_inwater_physical(prep::PrepBundle, domain::String)
    forcing_vars = DOMAIN_VARS[domain]["forcing"]
    fi = findfirst(==("river_inwater"), forcing_vars)
    fi === nothing && throw(ArgumentError("domain \"$domain\" has no river_inwater forcing"))
    split  = prep.dataset.val
    graphs = vcat(split[1], [w[end] for w in split[2:end]])
    n = prep.n_nodes
    T = length(graphs) - 1
    st = prep.norm_stats["river_inwater"]
    stat_vec(x) = x isa AbstractVector ? Float32.(x) : fill(Float32(x), n)
    μ, σ = stat_vec(st.mean), stat_vec(st.std)
    ps = get(prep.postscale, "river_inwater", ones(Float32, n))
    iw = Matrix{Float32}(undef, n, T)
    for t in 1:T
        iw[:, t] = (graphs[t + 1].ndata.forcing[fi, :] .* σ .+ μ) .* ps
    end
    return iw
end

# Layer-free residual of the predicted and the true (wflow) trajectories; writes
# `mb_residual.csv` (per step) and `mb_residual_by_node.csv`; returns the scalar
# summaries for `metrics.toml`.
function _analyze_mb_residual(prep::PrepBundle, ms::ModelSettings, p_states, t_states,
                              state_vars, dirs, split_times)
    qi = findfirst(==("river_q"), state_vars)
    hi = findfirst(==("river_h"), state_vars)
    iw = _val_inwater_physical(prep, ms.domain)
    wl = prep.postscale["river_q"] ./ prep.postscale["river_h"]    # channel area w·l [m²]
    θ  = ms.mb_theta
    kw = (; edge_src = prep.edge_src, edge_tgt = prep.edge_tgt, wl, dt = prep.dt, θ)

    pred = mb_residual_diagnostics(p_states[qi, :, :], p_states[hi, :, :], iw; kw...)
    ref  = mb_residual_diagnostics(t_states[qi, :, :], t_states[hi, :, :], iw; kw...)

    T = size(p_states, 3)
    step_rms(R) = Float32[let v = filter(isfinite, @view R[:, k])
                              isempty(v) ? NaN32 : Float32(sqrt(mean(abs2, v)))
                          end for k in 1:size(R, 2)]
    step_med(R) = Float32[let v = filter(isfinite, @view R[:, k])
                              isempty(v) ? NaN32 : Float32(median(abs.(v)))
                          end for k in 1:size(R, 2)]
    times = length(split_times) == T ? string.(split_times[2:end]) :
            fill("", T - 1)
    _write_plot_csv(joinpath(dirs.metrics, "mb_residual.csv"),
        ["step", "time", "rms_pred_m", "rms_true_m", "median_abs_pred_m", "median_abs_true_m",
         "volume_residual_pred_m3", "volume_residual_true_m3"],
        Any[collect(2:T), times, step_rms(pred.residual), step_rms(ref.residual),
            step_med(pred.residual), step_med(ref.residual),
            pred.volume_residual, ref.volume_residual])
    _write_plot_csv(joinpath(dirs.metrics, "mb_residual_by_node.csv"),
        ["node_idx", "upstream_area_m2", "rms_pred_m", "rms_true_m", "n_valid"],
        Any[collect(1:prep.n_nodes), prep.postscale["river_q"], pred.rms_node, ref.rms_node,
            pred.n_valid_node])

    @info "Mass-balance residual (θ=$θ): predicted rms=$(round(pred.summary.rms; sigdigits=3)) m, " *
          "reference (wflow) rms=$(round(ref.summary.rms; sigdigits=3)) m"
    return (; theta = Float32(θ), pred = pred.summary, reference = ref.summary)
end

# ---------------------------------------------------------------------------
# Stage-4-only entry points on an existing run directory
# ---------------------------------------------------------------------------

"""
    resolve_staticmaps(wflow_model_path, run_dir) -> path

Locate `staticmaps.nc` of a run. Runs copied from another machine (e.g. an HPC via
`scp`) keep the original absolute `wflow_model_path`, which does not resolve locally,
so fall back to `<repo>/models/<model_name>/staticmaps.nc`, where
`<repo> = dirname(dirname(run_dir))` (`<repo>/experiments/<run>`).
"""
function resolve_staticmaps(wflow_model_path::AbstractString, run_dir::AbstractString)
    direct = joinpath(wflow_model_path, "staticmaps.nc")
    isfile(direct) && return direct
    model_name = basename(rstrip(wflow_model_path, ['/', '\\']))
    # `<repo>/experiments/<run>` or, for grouped runs, `<repo>/experiments/<group>/<run>`
    tried = String[direct]
    dir = dirname(abspath(run_dir))
    for _ in 1:3
        local_path = joinpath(dirname(dir), "models", model_name, "staticmaps.nc")
        isfile(local_path) && return local_path
        push!(tried, local_path)
        dir = dirname(dir)
    end
    error("could not locate staticmaps.nc (tried " * join(("\"$p\"" for p in tried), ", ") * ")")
end

"""
    spatial_eval_run(run_dir; split = "val") -> (; metrics, ramp)

Spatial evaluation of an existing run from its gridded `output/<split>_pred.nc` /
`output/<split>_true.nc` (`split` is `"val"` or `"train"`), without a model, a
[`PrepBundle`](@ref) or any retraining; settings come from `<run_dir>/model/*.toml`.
Also works on runs from before the staged pipeline. Writes

    output/<split>_spatial_metrics.nc, metrics/<split>_spatial_metrics.csv,
    plots/<split>_spatial_metrics.png,
    plots/<split>_q_overpred_vs_ramp.png, metrics/<split>_q_overpred_vs_ramp.csv

(the ramp analysis only when `river_q` is a state variable; `ramp` is `nothing`
otherwise). [`analyze_run`](@ref) produces the same for the validation split under
split-free file names.
"""
function spatial_eval_run(run_dir::AbstractString; split::AbstractString = "val")
    run_dir = abspath(run_dir)
    isdir(run_dir) || throw(ArgumentError("not a directory: $run_dir"))
    dirs = _run_dirs(run_dir)
    ms = load_model_settings(joinpath(dirs.model, "model_settings.toml"))
    ds = load_data_settings(joinpath(dirs.model, "data_settings.toml"))
    schema = load_schema(ds.wflow_schema)
    staticmaps_file = resolve_staticmaps(ds.wflow_model_path, run_dir)
    foreach(mkpath, (dirs.plots, dirs.metrics, dirs.output))

    state_vars = DOMAIN_VARS[ms.domain]["state"]
    pred_path  = joinpath(dirs.output, "$(split)_pred.nc")
    true_path  = joinpath(dirs.output, "$(split)_true.nc")
    isfile(pred_path) || throw(ArgumentError("missing $pred_path"))
    isfile(true_path) || throw(ArgumentError("missing $true_path"))

    @info "Loading $(split) grids from $run_dir (domain=$(ms.domain))"
    p_grids = load_regrid_nc(pred_path, state_vars)
    t_grids = load_regrid_nc(true_path, state_vars)

    @info "Computing per-cell spatial error metrics"
    metrics = spatial_error_metrics(p_grids, t_grids, ms.domain)
    nc_out  = joinpath(dirs.output,  "$(split)_spatial_metrics.nc")
    csv_out = joinpath(dirs.metrics, "$(split)_spatial_metrics.csv")
    png_out = joinpath(dirs.plots,   "$(split)_spatial_metrics.png")
    write_spatial_metrics_to_netcdf(metrics, staticmaps_file, nc_out; schema)
    write_spatial_metrics_to_csv(metrics, csv_out)
    plot_spatial_metrics(metrics, ms.domain; path = png_out)
    @info "Wrote $nc_out, $csv_out and $png_out"

    ramp = nothing
    if "river_q" in state_vars
        @info "Analysing Q overprediction vs ramp rate"
        ramp = overprediction_vs_ramp(p_grids, t_grids)
        rc_png = joinpath(dirs.plots,   "$(split)_q_overpred_vs_ramp.png")
        rc_csv = joinpath(dirs.metrics, "$(split)_q_overpred_vs_ramp.csv")
        plot_overprediction_vs_ramp(ramp; path = rc_png, csv_path = rc_csv)
        @info "Wrote $rc_png and $rc_csv"
    end
    return (; metrics, ramp)
end

"""
    regen_mb_diagnostics(run_dir) -> Union{Nothing, NamedTuple}

Regenerate the layer-internal mass-balance diagnostics of a trained run without
retraining or re-running the stages before it: loads the [`PrepBundle`](@ref)
([`load_prep`](@ref)) and the trained model ([`load_model`](@ref)) of `run_dir` and
runs [`layer_mb_analysis`](@ref) on the validation split, rewriting
`plots/mb_diagnostics.png`, `plots/volume_budget.png` and the `mb_*` /
`volume_budget` CSVs in `metrics/`. Use it after changing the diagnostics or their
plots. Returns `nothing` for models without a mass-balance layer.

Needs `prep/preprocessed.jld2`, so runs produced before the staged pipeline must be
re-preprocessed ([`preprocess_run`](@ref)) first. Timestamps are taken from
`output/val_pred.nc` when present.
"""
function regen_mb_diagnostics(run_dir::AbstractString)
    run_dir = abspath(run_dir)
    dirs  = _run_dirs(run_dir)
    prep  = load_prep(run_dir)
    ms    = load_model_settings(joinpath(dirs.model, "model_settings.toml"))
    model = load_model(prep, ms, dirs.model)
    pred_path   = joinpath(dirs.output, "val_pred.nc")
    split_times = isfile(pred_path) ? load_regrid_times(pred_path) : nothing
    foreach(mkpath, (dirs.plots, dirs.metrics))
    return layer_mb_analysis(model.mass_balance, model, prep, dirs, split_times)
end