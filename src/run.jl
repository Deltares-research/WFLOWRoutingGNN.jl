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
    )

    md = d["model"]
    ms = ModelSettings(
        domain          = md["domain"],
        hidden_dim      = get(md, "hidden_dim",      64),
        nlayers         = get(md, "nlayers",          3),
        enc_activation  = ACTIVATIONS[get(md, "enc_activation",  "swish")],
        proc_activation = ACTIVATIONS[get(md, "proc_activation", "swish")],
    )

    td = d["train"]
    sd = get(td, "strategy", Dict{String,Any}())
    strategy = TrainingStrategy(
        get(sd, "steps",       [1]),
        get(sd, "durations",   [td["epochs"]]),
        get(sd, "noise_scale", 0.0),
    )
    ts = TrainSettings(
        epochs        = td["epochs"],
        batch_size    = td["batch_size"],
        lr_start      = td["lr_start"],
        lr_final      = td["lr_final"],
        lr_steps      = td["lr_steps"],
        strategy      = strategy,
        device        = Symbol(get(td, "device", "cpu")),
        val_daterange = if haskey(td, "val_daterange")
            r = td["val_daterange"]
            (Dates.DateTime(r[1]), Dates.DateTime(r[2]))
        else
            nothing
        end,
    )

    return first(run_wflow_gnn(ds, ms, ts))
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
       data_settings.toml
       model_settings.toml
       train_settings.toml
       norm_stats.toml
       model.jld2
       data/
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
    output_file     = joinpath(ds.wflow_model_path, "run_default", "output.nc")

    @info "Building Graph"
    graphs, norm_stats, grid, postscale = build_wflow_graph(staticmaps_file, output_file, ms.domain)

    # --- 2. Sliding-window horizon dataset ---
    @info "Building datasets"
    nhorizon = maximum(ts.strategy.steps) + 1
    dataset  = make_horizon_dataset(graphs, nhorizon; at = (ds.train_frac, ds.val_frac))

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

    # Build the sparse adjacency matrix from the graph topology.
    # A[i,j] = 1 means node j is an upstream neighbour of node i.
    # Self-loops are included so each node also aggregates its own state,
    # matching the default GraphConv behaviour.
    g0 = graphs[1]
    n_nodes   = g0.num_nodes
    src_edges, tgt_edges = edge_index(g0)
    all_src   = vcat(src_edges, collect(1:n_nodes))
    all_tgt   = vcat(tgt_edges, collect(1:n_nodes))
    A_sparse  = sparse(all_tgt, all_src, ones(Float32, length(all_src)), n_nodes, n_nodes)

    if ms.domain == "river"
        dt = get_timestep(output_file)
        mb = MassBalanceLayer(
            postscale["river_q"],
            postscale["river_h"],
            Float32(norm_stats["river_q"].mean),
            Float32(norm_stats["river_q"].std),
            Float32(norm_stats["river_h"].mean),
            Float32(norm_stats["river_h"].std),
            Float32(norm_stats["river_inwater"].mean),
            Float32(norm_stats["river_inwater"].std),
            dt,
        )
        h_weight = mb.σ_h / (mb.dt * mb.σ_q)
        @info "Mass balance h_loss_weight = $(round(h_weight; sigdigits=3)) " *
              "(σ_h=$(round(mb.σ_h; sigdigits=3)), σ_q=$(round(mb.σ_q; sigdigits=3)), dt=$(mb.dt) s)"
        ts.strategy.h_loss_weight = h_weight
        # @info "Mass balance enabled: h derived from physics constraint, h_loss_weight=0"
        # ts.strategy.h_loss_weight = 0f0
        model = dev_fn(WflowGNN(ms, mb, A_sparse))
    else
        model = dev_fn(WflowGNN(ms, A_sparse))
    end
    train_duration = @elapsed begin
        losses = train_model!(model, train_loader, val_loader, ts)
    end
    train_rollout = losses.train_rollout
    val_rollout   = losses.val_rollout
    train_1step   = losses.train_1step
    val_1step     = losses.val_1step

    # --- 5. Persist artefacts ---
    @info "Saving artefacts to $(joinpath(ds.runs_dir, ds.run_name))"
    run_dir  = joinpath(ds.runs_dir, ds.run_name)
    data_dir = joinpath(run_dir, "data")
    mkpath(data_dir)

    save_data_settings( joinpath(run_dir, "data_settings.toml"),  ds)
    save_model_settings(joinpath(run_dir, "model_settings.toml"), ms)
    save_train_settings(joinpath(run_dir, "train_settings.toml"), ts)

    # Normalisation statistics
    stats_dict = Dict(
        var => Dict("mean" => Float64(s.mean), "std" => Float64(s.std))
        for (var, s) in norm_stats
    )
    open(joinpath(run_dir, "norm_stats.toml"), "w") do io
        TOML.print(io, stats_dict)
    end

    # Model weights (always saved on CPU)
    JLD2.jldsave(joinpath(run_dir, "model.jld2");
                 model_state = Flux.state(Flux.cpu(model)))

    # Training loss curves
    plot_losses(train_rollout, val_rollout, train_1step, val_1step;
                train_q_1step = losses.train_q_1step,
                val_q_1step   = losses.val_q_1step,
                train_h_1step = losses.train_h_1step,
                val_h_1step   = losses.val_h_1step,
                path = joinpath(run_dir, "losses.png"))

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
    all_times = NCDataset(output_file, "r") do ds; ds["time"][:]; end
    cpu_model = Flux.cpu(model)
    n_params  = sum(length, Flux.trainables(cpu_model))

    val_rollout_duration = 0.0
    val_n_timesteps      = 0

    for (split_name, split_data, t_offset) in (
            ("train", dataset.train, 0),
            ("val",   dataset.val,   length(dataset.train)))

        t0 = time_ns()
        p_states, t_states = evaluate_trajectory(
            cpu_model, split_data, norm_stats, ms.domain;
            device = :cpu, postscale)
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
                               joinpath(run_dir, "$(split_name)_pred.nc"))
        write_regrid_to_netcdf(t_grids, staticmaps_file, split_times,
                               joinpath(run_dir, "$(split_name)_true.nc"))

        if split_name == "val"
            plot_validation_movie(p_grids, t_grids, ms.domain;
                                  path       = joinpath(run_dir, "validation.mp4"),
                                  framerate  = 10,
                                  timestamps = split_times)

            plot_downstream_timeseries(p_grids, t_grids, ms.domain, grid,
                                       postscale["river_q"];  # upstream area per node
                                       path       = joinpath(run_dir, "downstream_timeseries.png"),
                                       timestamps = split_times)

            if !isnothing(cpu_model.mass_balance)
                @info "Computing mass balance diagnostics on validation split"
                mb_diags = rollout_mb_diagnostics(cpu_model, split_data)
                plot_mb_diagnostics(mb_diags;
                                    path       = joinpath(run_dir, "mb_diagnostics.png"),
                                    timestamps = split_times)
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
                        cpu_model, dr_split, norm_stats, ms.domain;
                        device = :cpu, postscale)
                    dr_p_grids = regrid(dr_p_states, grid, ms.domain)
                    dr_t_grids = regrid(dr_t_states, grid, ms.domain)

                    # evaluate_trajectory flattens windows into a consecutive sequence
                    # and returns nhorizon + n_windows - 2 frames — more than n_windows.
                    # Derive timestamps directly from all_times for the exact frame count.
                    n_dr_frames   = size(dr_p_states, 3)
                    dr_pred_times = [all_times[clamp(n_train_w + w_start + i, 1, length(all_times))]
                                     for i in 1:n_dr_frames]

                    plot_validation_movie(dr_p_grids, dr_t_grids, ms.domain;
                                          path       = joinpath(run_dir, "validation_daterange.mp4"),
                                          framerate  = 10,
                                          timestamps = dr_pred_times)

                    plot_downstream_timeseries(dr_p_grids, dr_t_grids, ms.domain, grid,
                                               postscale["river_q"];
                                               path       = joinpath(run_dir, "downstream_timeseries_daterange.png"),
                                               timestamps = dr_pred_times)
                end
            end
        end
    end

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
