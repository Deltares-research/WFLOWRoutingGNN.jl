import JLD2
import Dates
using CUDA, cuDNN
using SparseArrays

"""
    run_wflow_gnn_from_toml(toml_path) -> model

Load all settings from a single TOML file and call `run_wflow_gnn`.

The TOML file must contain four top-level tables:

```toml
[data]
run_name         = "my_run"
runs_dir         = "runs"
wflow_model_path = "wflow_model/wflow_test_full"
train_frac       = 0.6
val_frac         = 0.2

[model]
domain          = "river"
hidden_dim      = 64
nlayers         = 3
enc_activation  = "swish"
proc_activation = "swish"

[train]
epochs     = 50
batch_size = 8
lr_start   = 1e-3
lr_final   = 1e-5
lr_steps   = 10
device     = "cpu"

[train.strategy]
steps       = [1, 2, 4]
durations   = [10, 20, 20]
noise_scale = 0.0
```

Relative paths in `[data]` are resolved relative to the directory containing
the TOML file.

Returns the trained model.
"""
function run_wflow_gnn_from_toml(toml_path::String)
    ds, ms, ts = parse_run_config(toml_path)
    return first(run_wflow_gnn(ds, ms, ts))
end

"""
    parse_run_config(toml_path) -> (ds, ms, ts)

Parse an experiment config TOML into `(DataSettings, ModelSettings,
TrainSettings)` without running any training. Relative paths in `[data]` are
resolved relative to the directory containing the TOML file.

Shared by [`run_wflow_gnn_from_toml`](@ref) and standalone tuning scripts (e.g.
the LR range test) so they interpret configs identically. The optional
`[train]` keys `lr_warmup_epochs` and `lr_peak_decay` feed the curriculum LR
schedule; `lr_steps` is retained for backward compatibility but unused by it.
"""
function parse_run_config(toml_path::String)
    isfile(toml_path) || throw(ArgumentError("TOML file not found: $toml_path"))
    toml_dir = dirname(abspath(toml_path))
    d        = TOML.parsefile(toml_path)

    haskey(d, "data")  || throw(ArgumentError("TOML missing [data] table"))
    haskey(d, "model") || throw(ArgumentError("TOML missing [model] table"))
    haskey(d, "train") || throw(ArgumentError("TOML missing [train] table"))

    # Resolve relative paths against the directory of the TOML file
    resolve(p) = isabspath(p) ? p : normpath(joinpath(toml_dir, p))

    dd = d["data"]
    ds = DataSettings(
        run_name         = dd["run_name"],
        runs_dir         = resolve(dd["runs_dir"]),
        wflow_model_path = resolve(dd["wflow_model_path"]),
        train_frac       = dd["train_frac"],
        val_frac         = dd["val_frac"],
        output_run_dir   = get(dd, "output_run_dir", "run_default"),
        wflow_schema     = get(dd, "wflow_schema",   "v1"),
    )

    md = d["model"]
    ms = ModelSettings(
        domain          = md["domain"],
        hidden_dim      = get(md, "hidden_dim",      64),
        nlayers         = get(md, "nlayers",          3),
        mlp_layers      = get(md, "mlp_layers",       1),
        enc_activation  = ACTIVATIONS[get(md, "enc_activation",  "swish")],
        proc_activation = ACTIVATIONS[get(md, "proc_activation", "swish")],
        enforce_mass_balance = get(md, "enforce_mass_balance", true),
        mb_theta        = Float32(get(md, "mb_theta", 1.0)),
        mb_augment_decoder = get(md, "mb_augment_decoder", false),
    )

    td = d["train"]
    sd = get(td, "strategy", Dict{String,Any}())
    strategy = TrainingStrategy(
        get(sd, "steps",       [1]),
        get(sd, "durations",   [td["epochs"]]),
        get(sd, "noise_scale", 0.0),
    )
    ts = TrainSettings(
        epochs           = td["epochs"],
        batch_size       = td["batch_size"],
        lr_start         = td["lr_start"],
        lr_final         = td["lr_final"],
        lr_steps         = get(td, "lr_steps", 10),
        lr_warmup_epochs = get(td, "lr_warmup_epochs", 1),
        lr_peak_decay    = get(td, "lr_peak_decay", 0.7),
        grad_clip        = get(td, "grad_clip", 1.0),
        h_loss_scale     = Symbol(get(td, "h_loss_scale", "absolute")),
        phase_backoff_factor = get(td, "phase_backoff_factor", 0.5),
        eval_horizon     = get(td, "eval_horizon", 30),
        eval_anchors     = get(td, "eval_anchors", 32),
        early_stopping   = get(td, "early_stopping", false),
        early_stopping_patience = get(td, "early_stopping_patience", 20),
        checkpoint_every = get(td, "checkpoint_every", 0),
        checkpoint_full_eval = get(td, "checkpoint_full_eval", false),
        strategy         = strategy,
        device           = Symbol(get(td, "device", "cpu")),
        val_daterange    = if haskey(td, "val_daterange")
            r = td["val_daterange"]
            (Dates.DateTime(r[1]), Dates.DateTime(r[2]))
        else
            nothing
        end,
    )

    return (ds, ms, ts)
end

"""
    build_gnn_model(ms, graphs, norm_stats, postscale, output_file, batch_size;
                    strategy = nothing, h_loss_scale = :absolute) -> WflowGNN

Construct a `WflowGNN` (with block-diagonal adjacency pre-computed for
`batch_size`) from a built graph time series. For the `"river"` domain this also
builds the `MassBalanceLayer` and, when `strategy` is supplied, sets
`strategy.h_loss_weight` to the mass-balance-consistent value.

`h_loss_scale` selects how the water-depth term of the loss is normalised:
- `:absolute`  → weight `σ_h/(dt·σ_q)` (balances the q- and h-loss magnitudes).
- `:increment` → weight `(σ_h/(dt·σ_q))²`, i.e. h-error measured on the
  mass-balance increment scale `dt·σ_q`. This makes `∂h_norm/∂q_norm ≈ O(1)`
  and removes the stiff gradient amplification of the hard mass-balance decoder
  (at the cost of weaker h-supervision once discharge is well fit).

Shared by [`run_wflow_gnn`](@ref) and standalone tuning scripts so the model is
built identically everywhere.
"""
function build_gnn_model(ms::ModelSettings, graphs, norm_stats, postscale,
                         output_file::AbstractString, batch_size::Int;
                         strategy::Union{Nothing,TrainingStrategy} = nothing,
                         h_loss_scale::Symbol = :absolute)
    g0      = graphs[1]
    n_nodes = g0.num_nodes

    # Sparse adjacency with self-loops: A[i,j]=1 means node j is an upstream
    # neighbour of node i (each node also aggregates its own state).
    src_edges, tgt_edges = edge_index(g0)
    all_src  = vcat(src_edges, collect(1:n_nodes))
    all_tgt  = vcat(tgt_edges, collect(1:n_nodes))
    A_sparse = sparse(all_tgt, all_src, ones(Float32, length(all_src)), n_nodes, n_nodes)

    if ms.domain == "river" && ms.enforce_mass_balance
        dt     = get_timestep(output_file)
        pq_vec = postscale["river_q"]
        ph_vec = postscale["river_h"]
        # Routing-only adjacency (no self-loops): A_routing[i,j]=1 means j flows into i.
        A_routing = sparse(all_tgt[1:length(src_edges)], all_src[1:length(src_edges)],
                           ones(Float32, length(src_edges)), n_nodes, n_nodes)
        mb = MassBalanceLayer(
            pq_vec,
            ph_vec,
            ph_vec ./ pq_vec,
            Float32(norm_stats["river_q"].mean),
            Float32(norm_stats["river_q"].std),
            Float32(norm_stats["river_h"].mean),
            Float32(norm_stats["river_h"].std),
            Float32(norm_stats["river_inwater"].mean),
            Float32(norm_stats["river_inwater"].std),
            dt,
            A_routing,
            nothing,  # A_routing_batched — set via precompute_batched
            0,        # batch_size
            ms.mb_theta,
        )
        # Base weight balances the q- and h-loss magnitudes; ∂h_norm/∂q_norm of
        # the hard mass-balance decoder equals `dt·σ_q/σ_h = 1/base`, so the
        # h-branch injects a q-gradient amplified by `1/base` at :absolute.
        # :increment squares the weight (measures h on the `dt·σ_q` increment
        # scale), cancelling that amplification so ∂h_norm/∂q_norm ≈ O(1).
        base_weight = mb.σ_h / (mb.dt * mb.σ_q)
        h_weight    = h_loss_scale === :increment ? base_weight^2 : base_weight
        @info "Mass balance h_loss_weight = $(round(h_weight; sigdigits=3)) " *
              "[scale=$(h_loss_scale)]  " *
              "(σ_h=$(round(mb.σ_h; sigdigits=3)), σ_q=$(round(mb.σ_q; sigdigits=3)), dt=$(mb.dt) s)"
        if mb.θ != 1f0
            @info "Mass balance θ = $(mb.θ) (mixed implicit/explicit; " *
                  "θ=1 fully implicit, θ=0 fully explicit). Effective stiff " *
                  "gain ∂h_norm/∂q_norm scaled by θ."
        end
        strategy === nothing || (strategy.h_loss_weight = h_weight)
        model = WflowGNN(ms, mb, A_sparse)
    else
        if ms.domain == "river" && !ms.enforce_mass_balance
            @info "Mass balance DISABLED: river_q and river_h are predicted " *
                  "independently (decoder out_dim = " *
                  "$(length(DOMAIN_VARS["river"]["state"])))"
        end
        model = WflowGNN(ms, A_sparse)
    end

    # Block-diagonal adjacency for the batch size. Must be built on CPU
    # (blockdiag needs SparseMatrixCSC); device movement happens at the call site.
    return precompute_batched(model, batch_size)
end

"""
    evaluate_and_write(cpu_model, dataset, norm_stats, grid, postscale, static_arr,
                       ms, ts, output_file, staticmaps_file, all_times, schema, run_dir)
        -> (val_rollout_duration_s, val_n_timesteps)

Run the full post-training evaluation for `cpu_model` and write all artefacts to
`run_dir`: per-split autoregressive trajectory rollouts, predicted/true NetCDF,
the validation movie, downstream timeseries, spatial-error maps, the Q
overprediction-vs-ramp diagnostic, mass-balance diagnostics (when applicable) and
the optional `val_daterange` rollout. `cpu_model` must already be on the CPU.

Shared by [`run_wflow_gnn`](@ref) for the final model and, when
`ts.checkpoint_full_eval` is set, for each periodic checkpoint.

Returns a `NamedTuple` `(; val_rollout_duration, val_n_timesteps,
spatial_summary, ramp)`; the last two feed the per-run `metrics.toml` summary
(`spatial_summary` from [`spatial_metric_summary`](@ref), `ramp` from
[`overprediction_vs_ramp`](@ref)) and are `nothing` when not applicable.
"""
function evaluate_and_write(model, dataset, norm_stats, grid, postscale,
                            static_arr, ms::ModelSettings, ts::TrainSettings,
                            output_file, staticmaps_file, all_times, schema,
                            output_dir, plots_dir, metrics_dir)

    eval_device = ts.device

    val_rollout_duration = 0.0
    val_n_timesteps      = 0
    spatial_summary      = nothing
    ramp_summary         = nothing

    for (split_name, split_data, t_offset) in (
            ("train", dataset.train, 0),
            ("val",   dataset.val,   length(dataset.train)))

        t0 = time_ns()
        p_states, t_states = evaluate_trajectory(
            model, split_data, norm_stats, ms.domain, static_arr;
            device = eval_device, postscale)
        if split_name == "val"
            val_rollout_duration = (time_ns() - t0) / 1e9
            val_n_timesteps      = size(p_states, 3)
        end
        p_grids = regrid(p_states, grid, ms.domain)
        t_grids = regrid(t_states, grid, ms.domain)

        n_frames    = size(p_states, 3)
        split_times = [all_times[clamp(t_offset + 1 + i, 1, length(all_times))]
                       for i in 1:n_frames]

        write_regrid_to_netcdf(p_grids, staticmaps_file, split_times,
                       joinpath(output_dir, "$(split_name)_pred.nc"); schema)
        write_regrid_to_netcdf(t_grids, staticmaps_file, split_times,
                       joinpath(output_dir, "$(split_name)_true.nc"); schema)

        if split_name == "val"
            plot_validation_movie(p_grids, t_grids, ms.domain;
                                  path       = joinpath(plots_dir, "validation.mp4"),
                                  framerate  = 10,
                                  timestamps = split_times)

            plot_downstream_timeseries(p_grids, t_grids, ms.domain, grid,
                                       postscale["river_q"];  # upstream area per node
                                       path       = joinpath(plots_dir, "downstream_timeseries.png"),
                                       timestamps = split_times,
                                       csv_path   = joinpath(metrics_dir, "downstream_timeseries.csv"))

            # Spatial performance: per-cell error maps + Q overprediction vs ramp
            sp_metrics = spatial_error_metrics(p_grids, t_grids, ms.domain)
            write_spatial_metrics_to_netcdf(sp_metrics, staticmaps_file,
                                joinpath(output_dir, "spatial_metrics.nc"); schema)
            write_spatial_metrics_to_csv(sp_metrics,
                             joinpath(metrics_dir, "spatial_metrics.csv"))
            plot_spatial_metrics(sp_metrics, ms.domain;
                         path = joinpath(plots_dir, "spatial_metrics.png"))
            spatial_summary = spatial_metric_summary(sp_metrics)
            if "river_q" in DOMAIN_VARS[ms.domain]["state"]
                ramp = overprediction_vs_ramp(p_grids, t_grids)
                @info "Q overprediction vs ramp: Pearson(e,g)=$(round(ramp.pearson_e_g; digits=3)) " *
                      "Spearman=$(round(ramp.spearman_e_g; digits=3)) over $(ramp.n) (cell,step) pairs"
                plot_overprediction_vs_ramp(ramp;
                    path     = joinpath(plots_dir, "q_overprediction_vs_ramp.png"),
                    csv_path = joinpath(metrics_dir, "q_overprediction_vs_ramp.csv"))
                ramp_summary = ramp
            end

            if !isnothing(model.mass_balance)
                @info "Computing mass balance diagnostics on validation split"
                cpu_model = Flux.cpu(model)
                mb_diags = rollout_mb_diagnostics(cpu_model, split_data, static_arr)
                plot_mb_diagnostics(mb_diags;
                                    path       = joinpath(plots_dir, "mb_diagnostics.png"),
                                    timestamps = split_times,
                                    csv_path   = joinpath(metrics_dir, "mb_diagnostics.csv"))
            end

            # Optional date-range rollout on the validation split
            if !isnothing(ts.val_daterange)
                dr_start, dr_stop = ts.val_daterange

                # Find which val-split windows fall within the date range.
                # split_times[i] is the timestamp of predicted frame i (graph t+1
                # of the split). The initial condition graph is split_data[1][1],
                # so the window starting at split index `w` covers times starting
                # at split_times[w]. We want the first window whose initial-state
                # time is ≥ dr_start, and we run until the last frame ≤ dr_stop.
                n_train_w = length(dataset.train)
                # all_times index of the first val graph (the initial condition)
                val_graph_times = [all_times[clamp(n_train_w + i, 1, length(all_times))]
                                   for i in 1:length(split_data)]

                # Window w uses split_data[w] as initial condition at val_graph_times[w]
                w_start = findfirst(t -> t ≥ dr_start, val_graph_times)
                w_stop  = findlast( t -> t ≤ dr_stop,  val_graph_times)

                if isnothing(w_start) || isnothing(w_stop) || w_start > w_stop
                    @warn "val_daterange $dr_start – $dr_stop does not overlap the val split; skipping"
                else
                    dr_split = split_data[w_start:w_stop]

                    dr_p_states, dr_t_states = evaluate_trajectory(
                        model, dr_split, norm_stats, ms.domain, static_arr;
                        device = eval_device, postscale)
                    dr_p_grids = regrid(dr_p_states, grid, ms.domain)
                    dr_t_grids = regrid(dr_t_states, grid, ms.domain)

                    # evaluate_trajectory flattens windows into a consecutive sequence
                    # and returns nhorizon + n_windows - 2 frames — more than n_windows.
                    # Derive timestamps directly from all_times for the exact frame count.
                    n_dr_frames   = size(dr_p_states, 3)
                    dr_pred_times = [all_times[clamp(n_train_w + w_start + i, 1, length(all_times))]
                                     for i in 1:n_dr_frames]

                    plot_validation_movie(dr_p_grids, dr_t_grids, ms.domain;
                                          path       = joinpath(plots_dir, "validation_daterange.mp4"),
                                          framerate  = 10,
                                          timestamps = dr_pred_times)

                    plot_downstream_timeseries(dr_p_grids, dr_t_grids, ms.domain, grid,
                                               postscale["river_q"];
                                               path       = joinpath(plots_dir, "downstream_timeseries_daterange.png"),
                                               timestamps = dr_pred_times,
                                               csv_path   = joinpath(metrics_dir, "downstream_timeseries_daterange.csv"))
                end
            end
        end
    end

    return (; val_rollout_duration, val_n_timesteps, spatial_summary, ramp = ramp_summary)
end

"""
    write_run_metrics_toml(path, losses, ts, run_meta, spatial_summary, ramp) -> path

Write a compact scalar summary of a completed run to `path` as TOML: the final
(and best) values of the per-epoch training history, run metadata (`run_meta`),
the median of each aggregated spatial-error metric per state variable, and the
overprediction-vs-ramp correlations. Non-finite values are omitted so the file
stays a valid, parser-friendly TOML of plain numbers — a token-cheap single-file
entry point for downstream evaluation agents.
"""
function write_run_metrics_toml(path::AbstractString, losses, ts::TrainSettings,
                                run_meta, spatial_summary, ramp)
    function putf!(d, k, v)
        v === nothing && return
        if v isa Integer
            d[k] = v
        elseif v isa Real && isfinite(v)
            d[k] = round(Float64(v); sigdigits = 6)
        end
    end
    lastf(v) = (v === nothing || isempty(v)) ? nothing : last(v)
    function bestf(v, red)
        (v === nothing || isempty(v)) && return nothing
        fv = filter(isfinite, v)
        isempty(fv) ? nothing : red(fv)
    end

    root = Dict{String, Any}()

    run_t = Dict{String, Any}()
    putf!(run_t, "n_params",               run_meta.n_params)
    putf!(run_t, "train_duration_s",       run_meta.train_duration_s)
    putf!(run_t, "val_rollout_duration_s", run_meta.val_rollout_duration_s)
    putf!(run_t, "val_n_timesteps",        run_meta.val_n_timesteps)
    putf!(run_t, "epochs_run",             losses.stopped_epoch)
    putf!(run_t, "best_epoch",             losses.best_epoch)
    root["run"] = run_t

    loss_t = Dict{String, Any}()
    putf!(loss_t, "final_train_rollout", lastf(losses.train_rollout))
    putf!(loss_t, "final_val_rollout",   lastf(losses.val_rollout))
    putf!(loss_t, "best_val_rollout",    bestf(losses.val_rollout, minimum))
    putf!(loss_t, "final_train_1step",   lastf(losses.train_1step))
    putf!(loss_t, "final_val_1step",     lastf(losses.val_1step))
    putf!(loss_t, "final_train_q_1step", lastf(losses.train_q_1step))
    putf!(loss_t, "final_val_q_1step",   lastf(losses.val_q_1step))
    putf!(loss_t, "final_train_h_1step", lastf(losses.train_h_1step))
    putf!(loss_t, "final_val_h_1step",   lastf(losses.val_h_1step))
    root["loss"] = loss_t

    stab_t = Dict{String, Any}()
    putf!(stab_t, "final_grad_norm", lastf(losses.grad_norm))
    putf!(stab_t, "max_grad_norm",   bestf(losses.grad_norm, maximum))
    putf!(stab_t, "final_lr",        lastf(losses.lr))
    putf!(stab_t, "final_amp",       lastf(losses.val_amp))
    putf!(stab_t, "final_mb_gain",   lastf(losses.mb_gain))
    putf!(stab_t, "n_nonfinite_skips", get(losses, :n_nonfinite_skips, nothing))
    putf!(stab_t, "n_backoffs",        get(losses, :n_backoffs, nothing))
    stopped_early = get(losses, :stopped_early, nothing)
    stopped_early === nothing || (stab_t["stopped_early"] = stopped_early)
    root["training_stability"] = stab_t

    fh_t = Dict{String, Any}()
    putf!(fh_t, "horizon",          ts.eval_horizon)
    putf!(fh_t, "final_val_rmse",   lastf(losses.val_fixed_rmse))
    putf!(fh_t, "best_val_rmse",    bestf(losses.val_fixed_rmse, minimum))
    putf!(fh_t, "final_peak_ratio", lastf(losses.val_peak_ratio))
    root["fixed_horizon"] = fh_t

    if spatial_summary !== nothing
        sp_t = Dict{String, Any}()
        for (vname, mdict) in spatial_summary
            var_t = Dict{String, Any}()
            for (m, s) in mdict
                putf!(var_t, m, s.median)
            end
            sp_t[vname] = var_t
        end
        root["spatial_median"] = sp_t
    end

    if ramp !== nothing
        ramp_t = Dict{String, Any}()
        putf!(ramp_t, "n",            ramp.n)
        putf!(ramp_t, "pearson_e_g",  ramp.pearson_e_g)
        putf!(ramp_t, "pearson_op_g", ramp.pearson_op_g)
        putf!(ramp_t, "spearman_e_g", ramp.spearman_e_g)
        root["ramp"] = ramp_t
    end

    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        TOML.print(io, root)
    end
    return path
end

"""
    run_wflow_gnn(ds, ms, ts) -> model

Execute the full training workflow and save all artefacts.

Steps:
1. Build the `GNNGraph` time series from `ds.wflow_model_path`
   (`staticmaps.nc` + `run_default/output.nc`) for domain `ms.domain`.
2. Create sliding-window samples with horizon `maximum(ts.strategy.steps) + 1`
   and split into train / val / test using `ds.train_frac` / `ds.val_frac`.
3. Build a `WflowGNN` from `ms` and train it with `train_model!`.
4. Save all artefacts under `<ds.runs_dir>/<ds.run_name>/`:
       model/
           data_settings.toml
           model_settings.toml
           train_settings.toml
           norm_stats.toml
           model.jld2
       metrics/
           metrics.toml
           *.csv
       plots/
           *.png
           *.mp4
       output/
           train_pred.nc, train_true.nc, val_pred.nc, val_true.nc
           spatial_metrics.nc
           data/
               grid.jld2
               train.jld2
               val.jld2
               test.jld2

Returns `(model, metrics)` where `metrics` is a `NamedTuple` with fields:
- `final_train_loss`            : rollout train loss at the last epoch
- `final_val_loss`              : rollout val loss at the last epoch
- `n_params`                    : total number of trainable model parameters
- `train_duration_s`            : wall-clock seconds spent in `train_model!`
- `val_rollout_duration_s`      : wall-clock seconds spent on the val trajectory rollout
- `val_n_timesteps`             : number of timesteps in the val trajectory
"""
function run_wflow_gnn(ds::DataSettings, ms::ModelSettings, ts::TrainSettings)

    # --- 1. Build time-series graphs ---
    staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
    output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")

    @info "Building Graph"
    schema = load_schema(ds.wflow_schema)
    graphs, norm_stats, grid, postscale, static_arr = build_wflow_graph(staticmaps_file, output_file, ms.domain; schema)

    g0        = graphs[1]
    n_nodes   = g0.num_nodes
    n_edges   = g0.num_edges
    n_times   = length(graphs)
    @info @sprintf("Graph: %d nodes  %d edges  %d timesteps", n_nodes, n_edges, n_times)

    # --- 2. Sliding-window horizon dataset ---
    @info "Building datasets"
    nhorizon = maximum(ts.strategy.steps) + 1
    dataset  = make_horizon_dataset(graphs, nhorizon; at = (ds.train_frac, ds.val_frac))

    n_train_windows = length(dataset.train)
    n_val_windows   = length(dataset.val)
    n_test_windows  = length(dataset.test)
    n_state   = size(g0.ndata.state,   1)
    n_forcing = size(g0.ndata.forcing, 1)
    n_static  = size(static_arr, 1)
    bytes_per_window = nhorizon * n_nodes * (n_state + n_forcing) * sizeof(Float32)
    @info @sprintf("Dataset: %d train windows  %d val windows  %d test windows  (horizon=%d)",
                   n_train_windows, n_val_windows, n_test_windows, nhorizon)
    @info @sprintf("Window size: %d steps × %d nodes × (%d state + %d forcing) = %.1f KB  [%d static features shared]",
                   nhorizon, n_nodes, n_state, n_forcing, bytes_per_window / 1024, n_static)

    # --- 3. DataLoaders ---
    train_loader = DataLoader(dataset.train;
                              batchsize = min(ts.batch_size, length(dataset.train)),
                              shuffle   = true,
                              collate   = true,
                              parallel  = true)
    val_loader   = DataLoader(dataset.val;
                              batchsize = min(ts.batch_size, length(dataset.val)),
                              shuffle   = false,
                              collate   = true,
                              parallel  = true)

    # --- 4. Build model and train ---
    @info "Training model"
    dev_fn  = ts.device == :gpu ? Flux.gpu : identity

    train_batch_size = min(ts.batch_size, length(dataset.train))
    model = build_gnn_model(ms, graphs, norm_stats, postscale, output_file,
                            train_batch_size; strategy = ts.strategy,
                            h_loss_scale = ts.h_loss_scale)

    n_params = sum(length, Flux.params(model))
    @info @sprintf("Model: %d MP layers  hidden_dim=%d  mlp_layers=%d  trainable params=%s",
                   ms.nlayers, ms.hidden_dim, ms.mlp_layers,
                   replace(string(n_params), r"(?<=\d)(?=(\d{3})+$)" => "_"))

    model = dev_fn(model)

    if CUDA.functional()
        mi = CUDA.MemoryInfo()
        used_b  = mi.total_bytes - mi.free_bytes
        pool_str = isnothing(mi.pool_used_bytes) ? "" :
            @sprintf("  |  pool: %.3f GiB used  %.3f GiB reserved",
                     mi.pool_used_bytes    / 2^30,
                     mi.pool_reserved_bytes / 2^30)
        @info @sprintf("GPU memory after model load: %.3f GiB used / %.3f GiB total%s",
                       used_b / 2^30, mi.total_bytes / 2^30, pool_str)
    end

    # Timestamps of the full output series (needed both for per-checkpoint eval
    # during training and the final evaluation).
    all_times = NCDataset(output_file, "r") do dsx; dsx["time"][:]; end

    run_dir = joinpath(ds.runs_dir, ds.run_name)
    mkpath(run_dir)
    model_dir   = joinpath(run_dir, "model")
    metrics_dir = joinpath(run_dir, "metrics")
    plots_dir   = joinpath(run_dir, "plots")
    output_dir  = joinpath(run_dir, "output")
    mkpath(model_dir)
    mkpath(metrics_dir)
    mkpath(plots_dir)
    mkpath(output_dir)

    # Fixed-horizon validation metric (constant-length rollout from anchors),
    # comparable epoch-to-epoch and used for early stopping when enabled.
    fixed_eval = build_fixed_horizon_eval(dataset.val, static_arr, norm_stats,
                                          ms.domain, postscale;
                                          horizon   = ts.eval_horizon,
                                          n_anchors = ts.eval_anchors)

    # Periodic checkpoint hook: save weights every `checkpoint_every` epochs and,
    # when requested, run the same full evaluation used at the end of training.
    checkpoint_callback = ts.checkpoint_every > 0 ?
        function (m, epoch)
            ckpt_dir     = joinpath(model_dir, "checkpoints", @sprintf("epoch_%04d", epoch))
            ckpt_model   = joinpath(ckpt_dir, "model")
            ckpt_metrics = joinpath(ckpt_dir, "metrics")
            ckpt_plots   = joinpath(ckpt_dir, "plots")
            ckpt_output  = joinpath(ckpt_dir, "output")
            mkpath(ckpt_model)
            mkpath(ckpt_metrics)
            mkpath(ckpt_plots)
            mkpath(ckpt_output)
            cpu_ckpt = Flux.cpu(m)
            JLD2.jldsave(joinpath(ckpt_model, "model.jld2");
                         model_state = Flux.state(cpu_ckpt))
            @info "Saved checkpoint (epoch $epoch) → $ckpt_dir"
            if ts.checkpoint_full_eval
                @info "Running full evaluation for checkpoint epoch $epoch"
                evaluate_and_write(m, dataset, norm_stats, grid, postscale,
                                   static_arr, ms, ts, output_file, staticmaps_file,
                                   all_times, schema, ckpt_output, ckpt_plots, ckpt_metrics)
            end
        end : nothing

    train_duration = @elapsed begin
        losses = train_model!(model, train_loader, val_loader, ts, static_arr;
                              fixed_eval          = fixed_eval,
                              checkpoint_callback = checkpoint_callback)
    end

    if CUDA.functional()
        mi = CUDA.MemoryInfo()
        used_b  = mi.total_bytes - mi.free_bytes
        pool_str = isnothing(mi.pool_used_bytes) ? "" :
            @sprintf("  |  pool: %.3f GiB used  %.3f GiB reserved",
                     mi.pool_used_bytes    / 2^30,
                     mi.pool_reserved_bytes / 2^30)
        @info @sprintf("GPU memory after training:   %.3f GiB used / %.3f GiB total%s",
                       used_b / 2^30, mi.total_bytes / 2^30, pool_str)
    end
    train_rollout = losses.train_rollout
    val_rollout   = losses.val_rollout
    train_1step   = losses.train_1step
    val_1step     = losses.val_1step

    # --- 5. Persist artefacts ---
    @info "Saving artefacts to $(run_dir)"
    data_dir = joinpath(output_dir, "data")
    mkpath(data_dir)

    save_data_settings( joinpath(model_dir, "data_settings.toml"),  ds)
    save_model_settings(joinpath(model_dir, "model_settings.toml"), ms)
    save_train_settings(joinpath(model_dir, "train_settings.toml"), ts)

    # Normalisation statistics
    stats_dict = Dict(
        var => Dict("mean" => Float64(s.mean), "std" => Float64(s.std))
        for (var, s) in norm_stats
    )
    open(joinpath(model_dir, "norm_stats.toml"), "w") do io
        TOML.print(io, stats_dict)
    end

    # Model weights (always saved on CPU)
    JLD2.jldsave(joinpath(model_dir, "model.jld2");
                 model_state = Flux.state(Flux.cpu(model)))

    # Training loss curves
    plot_losses(train_rollout, val_rollout, train_1step, val_1step;
                train_q_1step = losses.train_q_1step,
                val_q_1step   = losses.val_q_1step,
                train_h_1step = losses.train_h_1step,
                val_h_1step   = losses.val_h_1step,
                grad_norm = get(losses, :grad_norm, nothing),
                lr        = get(losses, :lr, nothing),
                steps     = get(losses, :steps, nothing),
                path     = joinpath(plots_dir, "losses.png"),
                csv_path = joinpath(metrics_dir, "losses.csv"))

    # Q→H error amplification diagnostic (mass-balance runs only)
    if haskey(losses, :train_amp) && any(isfinite, losses.train_amp)
        plot_amplification(losses.train_amp, losses.val_amp;
                           mb_gain = get(losses, :mb_gain, nothing),
                           path     = joinpath(plots_dir, "amplification.png"),
                           csv_path = joinpath(metrics_dir, "amplification.csv"))
    end

    # Fixed-horizon validation metric (discharge RMSE + peak ratio per epoch)
    if haskey(losses, :val_fixed_rmse) && any(isfinite, losses.val_fixed_rmse)
        plot_fixed_horizon(losses.val_fixed_rmse, losses.val_peak_ratio;
                           horizon = ts.eval_horizon,
                           path    = joinpath(plots_dir, "fixed_horizon.png"),
                           csv_path = joinpath(metrics_dir, "fixed_horizon.csv"))
    end

    # Grid lookup table (node index → raster position)
    JLD2.jldsave(joinpath(data_dir, "grid.jld2");
                 rows  = grid.rows,
                 cols  = grid.cols,
                 nrows = grid.nrows,
                 ncols = grid.ncols)

    # Dataset splits
    JLD2.jldsave(joinpath(data_dir, "train.jld2"); data = dataset.train)
    JLD2.jldsave(joinpath(data_dir, "val.jld2");   data = dataset.val)
    JLD2.jldsave(joinpath(data_dir, "test.jld2");  data = dataset.test)

    # --- 6. Evaluate train and val trajectories (each once) ---
    @info "Evaluating train and val trajectories"
    cpu_model = Flux.cpu(model)
    n_params  = sum(length, Flux.trainables(cpu_model))

    eval_out = evaluate_and_write(
        model, dataset, norm_stats, grid, postscale, static_arr, ms, ts,
        output_file, staticmaps_file, all_times, schema, output_dir, plots_dir, metrics_dir)
    val_rollout_duration = eval_out.val_rollout_duration
    val_n_timesteps      = eval_out.val_n_timesteps

    # Scalar per-run summary (token-cheap single file for evaluation agents):
    # final/best training-history metrics + aggregated spatial-error and
    # overprediction-vs-ramp diagnostics, so no CSV/NetCDF parsing is needed.
    write_run_metrics_toml(joinpath(metrics_dir, "metrics.toml"), losses, ts,
                           (; n_params,
                              train_duration_s       = train_duration,
                              val_rollout_duration_s = val_rollout_duration,
                              val_n_timesteps        = val_n_timesteps),
                           eval_out.spatial_summary, eval_out.ramp)

    metrics = (
        final_train_loss           = last(train_rollout),
        final_val_loss             = last(val_rollout),
        n_params                   = n_params,
        train_duration_s           = train_duration,
        val_rollout_duration_s     = val_rollout_duration,
        val_n_timesteps            = val_n_timesteps,
    )
    return model, metrics
end
