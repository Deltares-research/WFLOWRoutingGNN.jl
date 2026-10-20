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
# model for the standard diagnostics; the layer-internal mass-balance
# diagnostics (floor fraction, stiffness, volume budget) still need the trained
# model and are skipped when none is supplied (they become post-hoc in R5).
# ---------------------------------------------------------------------------

"""
    analyze_run(prep, ms, ts; dirs, model = nothing, train_summary = nothing,
                infer = nothing, metrics_toml = true)
        -> (; spatial_summary, ramp, river_q_perf, volume_budget, mb_floor, mb_stiffness)

Stage 4 of the pipeline. Analyses the validation split of a run from its gridded
NetCDF outputs (`<dirs.output>/val_pred.nc`, `val_true.nc`) and writes

    <dirs.plots>/    validation.mp4, downstream_timeseries.png, spatial_metrics.png,
                     q_overprediction_vs_ramp.png, [mb_diagnostics.png, volume_budget.png],
                     [validation_daterange.mp4, downstream_timeseries_daterange.png]
    <dirs.metrics>/  downstream_timeseries.csv, spatial_metrics.csv,
                     q_overprediction_vs_ramp.csv, [mb_*.csv, volume_budget.csv],
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
                    diagnostics (mass-balance models). Skipped, with a notice, when
                    `nothing`. May be on any device.
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
mass-balance layer.
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

    # Layer-internal mass-balance diagnostics (need the trained MB layer).
    if model !== nothing && has_mass_balance(model.mass_balance)
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
        mb_floor_summary = _mb_floor_summary(mb_diags)
        _write_mb_stiffness_by_node(
            joinpath(dirs.metrics, "mb_stiffness_by_node.csv"),
            cpu_model.mass_balance,
        )
        mb_stiffness_summary = _mb_stiffness_summary(cpu_model.mass_balance)

        volume_budget_summary = volume_budget_diagnostics(
            mb_diags;
            postscale_q = cpu_model.mass_balance.postscale_q,
            postscale_h = cpu_model.mass_balance.postscale_h,
            dt = cpu_model.mass_balance.dt,
            upstream_area = prep.postscale["river_q"])
        plot_volume_budget(volume_budget_summary;
                           path       = joinpath(dirs.plots, "volume_budget.png"),
                           timestamps = split_times,
                           csv_path   = joinpath(dirs.metrics, "volume_budget.csv"))
        @info "Volume budget: outlet=$(volume_budget_summary.outlet_idx) " *
              "pbias=$(round(volume_budget_summary.volume_pbias; digits=3))% " *
              "drift=$(round(volume_budget_summary.volume_drift; digits=3)) m^3/step"
    elseif model === nothing && mb_formulation_kind(ms) != :none
        @info "No model supplied: skipping layer-internal mass-balance diagnostics"
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
           mb_floor = mb_floor_summary, mb_stiffness = mb_stiffness_summary)

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
                               out.volume_budget, out.mb_floor, out.mb_stiffness)
    end
    return out
end
