import JLD2
import Dates
using Random
using CUDA

# ---------------------------------------------------------------------------
# Pipeline stages 1 (preprocess) and 2 (train)
#
#   preprocess_run(ds, ms; nhorizon) -> PrepBundle      writes <run>/prep/ (+ output/data/)
#   train_run(prep, ds, ms, ts)      -> (; model, ...)  reads only the PrepBundle,
#                                                       writes <run>/model/ (+ losses)
#
# `run_wflow_gnn` (src/run.jl) composes the stages; each can also be run on its
# own, `train_run(load_prep(run_dir), ...)` resuming from a stored `prep/`.
# ---------------------------------------------------------------------------

"""
Version of the on-disk [`PrepBundle`](@ref) layout (`prep/preprocessed.jld2`),
stored under the key `"prep_schema"` alongside the `wflow_schema` id. Bump when
fields are added, removed or change meaning; [`load_prep`](@ref) refuses files
with a different version.
"""
const PREP_SCHEMA_VERSION = 1

"""
    PrepBundle

Everything stage 2 (training) and later stages need from preprocessing, so that
no raw wflow output is read after [`preprocess_run`](@ref).

Fields:
- `prep_schema`  : [`PREP_SCHEMA_VERSION`](@ref) the bundle was written with.
- `wflow_schema` : id of the wflow variable schema (`DataSettings.wflow_schema`).
- `domain`       : model domain (`ModelSettings.domain`).
- `staticmaps_file`, `output_file` : source wflow files (provenance; still read by
  the fused evaluation until inference/analysis are split from training).
- `train_frac`, `val_frac` : temporal split fractions used.
- `nhorizon`     : window length of the horizon dataset (`max(strategy.steps) + 1`).
- `dt`           : model timestep in seconds.
- `times`        : timestamps of the full output series.
- `n_nodes`, `edge_src`, `edge_tgt` : graph topology (edge index of the
  unbatched graph; the routing adjacency is derived from it in
  [`build_gnn_model`](@ref)).
- `grid`         : node↔cell mapping `(rows, cols, nrows, ncols)` — node `i` sits at
  raster cell `(rows[i], cols[i])`; the inverse-gather key for gridded outputs.
- `norm_stats`, `postscale`, `static_arr` : normalisation statistics, the
  per-node physical-unit multipliers and the shared static feature matrix.
- `peak_stats`   : per-node peak-loss statistics `(q = (u, s), h = (u, s))` from the
  training period, or `nothing` when not computed.
- `dataset`      : `(train, val, test)` sliding-window splits.
"""
struct PrepBundle
    prep_schema     :: Int
    wflow_schema    :: String
    domain          :: String
    staticmaps_file :: String
    output_file     :: String
    train_frac      :: Float64
    val_frac        :: Float64
    nhorizon        :: Int
    dt              :: Float32
    times           :: Vector
    n_nodes         :: Int
    edge_src        :: Vector{Int}
    edge_tgt        :: Vector{Int}
    grid            :: NamedTuple
    norm_stats      :: Dict{String,NamedTuple}
    postscale       :: Dict{String,Vector{Float32}}
    static_arr      :: Matrix{Float32}
    peak_stats      :: Union{Nothing,NamedTuple}
    dataset         :: NamedTuple
end

# Directory layout of a run (`runs_dir/run_name/…`).
function _run_dirs(root::AbstractString)
    return (; root,
            model   = joinpath(root, "model"),
            metrics = joinpath(root, "metrics"),
            plots   = joinpath(root, "plots"),
            output  = joinpath(root, "output"),
            data    = joinpath(root, "output", "data"),
            prep    = joinpath(root, "prep"))
end
_run_dirs(ds::DataSettings) = _run_dirs(joinpath(ds.runs_dir, ds.run_name))

"""
    save_prep(prep, run_dir) -> prep_file

Persist `prep` under `run_dir`:

    prep/preprocessed.jld2        everything except the dataset (see [`PrepBundle`](@ref))
    output/data/grid.jld2         node → raster lookup (rows, cols, nrows, ncols)
    output/data/{train,val,test}.jld2   dataset splits (key `data`)

The dataset splits and `grid.jld2` keep their historical location under
`output/data/` so existing consumers are unaffected.
"""
function save_prep(prep::PrepBundle, run_dir::AbstractString)
    dirs = _run_dirs(run_dir)
    mkpath(dirs.prep)
    mkpath(dirs.data)

    prep_file = joinpath(dirs.prep, "preprocessed.jld2")
    JLD2.jldsave(prep_file;
                 prep_schema     = prep.prep_schema,
                 wflow_schema    = prep.wflow_schema,
                 domain          = prep.domain,
                 staticmaps_file = prep.staticmaps_file,
                 output_file     = prep.output_file,
                 train_frac      = prep.train_frac,
                 val_frac        = prep.val_frac,
                 nhorizon        = prep.nhorizon,
                 dt              = prep.dt,
                 times           = prep.times,
                 n_nodes         = prep.n_nodes,
                 edge_src        = prep.edge_src,
                 edge_tgt        = prep.edge_tgt,
                 grid            = prep.grid,
                 norm_stats      = prep.norm_stats,
                 postscale       = prep.postscale,
                 static_arr      = prep.static_arr,
                 peak_stats      = prep.peak_stats)

    # Grid lookup table (node index → raster position)
    JLD2.jldsave(joinpath(dirs.data, "grid.jld2");
                 rows  = prep.grid.rows,
                 cols  = prep.grid.cols,
                 nrows = prep.grid.nrows,
                 ncols = prep.grid.ncols)

    # Dataset splits
    JLD2.jldsave(joinpath(dirs.data, "train.jld2"); data = prep.dataset.train)
    JLD2.jldsave(joinpath(dirs.data, "val.jld2");   data = prep.dataset.val)
    JLD2.jldsave(joinpath(dirs.data, "test.jld2");  data = prep.dataset.test)
    return prep_file
end

"""
    load_prep(run_dir) -> PrepBundle

Load the [`PrepBundle`](@ref) written by [`save_prep`](@ref) /
[`preprocess_run`](@ref) from `run_dir`. Throws an `ArgumentError` when
`prep/preprocessed.jld2` is missing or was written with a different
`prep_schema` version.
"""
function load_prep(run_dir::AbstractString)
    dirs      = _run_dirs(run_dir)
    prep_file = joinpath(dirs.prep, "preprocessed.jld2")
    isfile(prep_file) ||
        throw(ArgumentError("no preprocessed bundle at $prep_file (run preprocess_run first)"))
    d = JLD2.load(prep_file)
    version = get(d, "prep_schema", nothing)
    version == PREP_SCHEMA_VERSION ||
        throw(ArgumentError("$prep_file has prep_schema $(repr(version)); " *
                            "this code reads prep_schema $PREP_SCHEMA_VERSION — re-run preprocess_run"))
    dataset = (train = JLD2.load(joinpath(dirs.data, "train.jld2"), "data"),
               val   = JLD2.load(joinpath(dirs.data, "val.jld2"),   "data"),
               test  = JLD2.load(joinpath(dirs.data, "test.jld2"),  "data"))
    return PrepBundle(d["prep_schema"], d["wflow_schema"], d["domain"],
                      d["staticmaps_file"], d["output_file"],
                      d["train_frac"], d["val_frac"], d["nhorizon"], d["dt"], d["times"],
                      d["n_nodes"], d["edge_src"], d["edge_tgt"], d["grid"],
                      d["norm_stats"], d["postscale"], d["static_arr"], d["peak_stats"],
                      dataset)
end

"""
    preprocess_run(ds, ms; nhorizon, peak_stats = true) -> PrepBundle
    preprocess_run(ds, ms, ts) -> PrepBundle

Stage 1 of the pipeline. Builds the graph time series
([`build_wflow_graph`](@ref)), the sliding-window horizon dataset
([`make_horizon_dataset`](@ref)), the per-node peak-loss statistics
([`peak_node_stats`](@ref)) and the normalised-tail diagnostics, then persists
the result with [`save_prep`](@ref) under `<ds.runs_dir>/<ds.run_name>/`:

    prep/preprocessed.jld2, output/data/{grid,train,val,test}.jld2
    metrics/normalized_tail_diagnostics.toml

Deterministic given the settings: no model, no device and no random numbers.

`nhorizon` is the window length of the dataset; the `ts` method derives it from
the curriculum (`maximum(ts.strategy.steps) + 1`) and computes the peak
statistics only when the loss needs them (`ts.strategy.loss_type == :huber`).
`peak_stats = false` skips that (potentially expensive) computation.
"""
function preprocess_run(ds::DataSettings, ms::ModelSettings;
                        nhorizon::Int, peak_stats::Bool = true)
    staticmaps_file = joinpath(ds.wflow_model_path, "staticmaps.nc")
    output_file     = joinpath(ds.wflow_model_path, ds.output_run_dir, "output.nc")

    # --- 1. Build time-series graphs ---
    @info "Building Graph"
    schema = load_schema(ds.wflow_schema)
    graphs, norm_stats, grid, postscale, static_arr = build_wflow_graph(
        staticmaps_file, output_file, ms.domain;
        schema,
        stats_frac = ds.train_frac,
        include_log_upstream_area = ms.include_log_upstream_area,
        sigma_floor_h = ms.mb_sigma_h_floor,
        sigma_floor_q = ms.mb_sigma_q_floor,
        sigma_floor_inwater = ms.mb_sigma_inwater_floor,
        normalization = ms.normalization)

    g0        = graphs[1]
    n_nodes   = g0.num_nodes
    n_edges   = g0.num_edges
    n_times   = length(graphs)
    @info @sprintf("Graph: %d nodes  %d edges  %d timesteps", n_nodes, n_edges, n_times)

    # --- 2. Sliding-window horizon dataset ---
    @info "Building datasets"
    dataset = make_horizon_dataset(graphs, nhorizon; at = (ds.train_frac, ds.val_frac))

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

    # --- 2b. Per-node peak-loss stats (Huber loss only) ---
    # Precomputed once from the training-period slice of `graphs` only (never
    # val/test) — the `u_i`/`s_i` design notes call for. `loss_function`
    # hardcodes state row 1 = q, row 2 = h (same convention already used by the
    # mass-balance decoder), so only domains with >= 2 state variables are
    # supported; other domains keep the coarse per-batch fallback threshold.
    peak = nothing
    state_vars = DOMAIN_VARS[ms.domain]["state"]
    if peak_stats && length(state_vars) >= 2
        node_stats = peak_node_stats(graphs, ms.domain; frac_train = ds.train_frac)
        peak = (q = node_stats[state_vars[1]], h = node_stats[state_vars[2]])
    end

    # Timestamps of the full output series and the model timestep.
    all_times = NCDataset(output_file, "r") do dsx; dsx["time"][:]; end
    dt        = get_timestep(output_file)

    dirs = _run_dirs(ds)
    foreach(mkpath, (dirs.root, dirs.model, dirs.metrics, dirs.plots, dirs.output))

    # Lightweight normalized-tail dump for train/val/test/static z-scored
    # features. Helps detect heavy-tail outliers after feature scaling.
    tail_diag = normalized_tail_diagnostics(
        graphs, static_arr, ms.domain;
        frac_train = ds.train_frac,
        frac_val = ds.val_frac)
    write_normalized_tail_diagnostics(
        joinpath(dirs.metrics, "normalized_tail_diagnostics.toml"), tail_diag)

    edge_src, edge_tgt = edge_index(g0)
    prep = PrepBundle(PREP_SCHEMA_VERSION, ds.wflow_schema, ms.domain,
                      staticmaps_file, output_file,
                      ds.train_frac, ds.val_frac, nhorizon, dt, all_times,
                      n_nodes, edge_src, edge_tgt, grid,
                      norm_stats, postscale, static_arr, peak, dataset)
    save_prep(prep, dirs.root)
    return prep
end

preprocess_run(ds::DataSettings, ms::ModelSettings, ts::TrainSettings) =
    preprocess_run(ds, ms;
                   nhorizon   = maximum(ts.strategy.steps) + 1,
                   peak_stats = ts.strategy.loss_type == :huber)

# Reproducibility toggle: when a seed is configured, seed both CPU and GPU RNGs
# before any stochastic run component (init, shuffling, injected noise).
function _seed_rngs(ts::TrainSettings)
    isnothing(ts.seed) && return nothing
    Random.seed!(ts.seed)
    if CUDA.functional()
        CUDA.seed!(UInt64(ts.seed))
    end
    @info "Using fixed RNG seed $(ts.seed)"
    return nothing
end

function _log_gpu_memory(label::AbstractString)
    CUDA.functional() || return nothing
    mi = CUDA.MemoryInfo()
    used_b  = mi.total_bytes - mi.free_bytes
    pool_str = isnothing(mi.pool_used_bytes) ? "" :
        @sprintf("  |  pool: %.3f GiB used  %.3f GiB reserved",
                 mi.pool_used_bytes    / 2^30,
                 mi.pool_reserved_bytes / 2^30)
    @info @sprintf("GPU memory %s: %.3f GiB used / %.3f GiB total%s",
                   label, used_b / 2^30, mi.total_bytes / 2^30, pool_str)
    return nothing
end

"""
    train_run(prep, ds, ms, ts) -> (; model, losses, train_duration_s, n_params)

Stage 2 of the pipeline. Builds the model from `prep` ([`build_gnn_model`](@ref)),
trains it with [`train_model!`](@ref) and writes the training artefacts under
`<ds.runs_dir>/<ds.run_name>/`:

    model/    data_settings.toml, model_settings.toml, train_settings.toml,
              norm_stats.toml, model.jld2, train_summary.jld2
    metrics/  losses.csv, amplification.csv, fixed_horizon*.csv
    plots/    losses.png, amplification.png, fixed_horizon.png

`train_summary.jld2` (keys `losses`, `train_duration_s`, `n_params`) lets
[`analyze_run`](@ref) write `metrics.toml` without the in-memory training state.

Reads only `prep` — never the raw wflow outputs — except that the periodic
full-evaluation checkpoint hook (`ts.checkpoint_full_eval`) runs
[`infer_run`](@ref) + [`analyze_run`](@ref), which read the raw wflow file paths
stored in `prep` (coordinates only). `prep` must have been built with the
curriculum's window length (`prep.nhorizon == maximum(ts.strategy.steps) + 1`).

Returns the trained `model` (on `ts.device`), the `losses` history, the
wall-clock seconds spent in `train_model!` and the trainable parameter count.
"""
function train_run(prep::PrepBundle, ds::DataSettings, ms::ModelSettings, ts::TrainSettings)
    prep.domain == ms.domain ||
        throw(ArgumentError("prep was built for domain \"$(prep.domain)\" but ms.domain is \"$(ms.domain)\""))
    nhorizon = maximum(ts.strategy.steps) + 1
    prep.nhorizon == nhorizon ||
        throw(ArgumentError("prep has horizon $(prep.nhorizon) but the training strategy needs " *
                            "$nhorizon (maximum(steps) + 1) — re-run preprocess_run"))
    if ts.strategy.loss_type == :huber && length(DOMAIN_VARS[ms.domain]["state"]) >= 2 &&
       isnothing(prep.peak_stats)
        throw(ArgumentError("loss_type = :huber needs peak_stats; re-run preprocess_run " *
                            "with peak_stats = true"))
    end

    _seed_rngs(ts)

    dataset    = prep.dataset
    norm_stats = prep.norm_stats
    postscale  = prep.postscale
    static_arr = prep.static_arr
    peak_stats = ts.strategy.loss_type == :huber ? prep.peak_stats : nothing

    dirs = _run_dirs(ds)
    foreach(mkpath, (dirs.root, dirs.model, dirs.metrics, dirs.plots, dirs.output))

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
    model = build_gnn_model(ms, prep, train_batch_size; strategy = ts.strategy,
                            h_loss_scale = ts.h_loss_scale)

    n_params = sum(length, Flux.params(model))
    @info @sprintf("Model: %d MP layers  hidden_dim=%d  mlp_layers=%d  trainable params=%s",
                   ms.nlayers, ms.hidden_dim, ms.mlp_layers,
                   replace(string(n_params), r"(?<=\d)(?=(\d{3})+$)" => "_"))

    model = dev_fn(model)
    _log_gpu_memory("after model load")

    # Fixed-horizon validation metric (constant-length rollout from anchors),
    # comparable epoch-to-epoch and used for early stopping when enabled.
    fixed_eval = build_fixed_horizon_eval(dataset.val, static_arr, norm_stats,
                                          ms.domain, postscale;
                                          horizon   = ts.eval_horizon,
                                          n_anchors = ts.eval_anchors)

    # Periodic checkpoint hook: save weights every `checkpoint_every` epochs and,
    # when requested, run the same inference + analysis stages used at the end.
    checkpoint_callback = ts.checkpoint_every > 0 ?
        function (m, epoch)
            ckpt_dir     = joinpath(dirs.model, "checkpoints", @sprintf("epoch_%04d", epoch))
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
                ckpt_dirs = (; output = ckpt_output, plots = ckpt_plots, metrics = ckpt_metrics)
                infer_run(m, prep, ms, ts; dirs = ckpt_dirs)
                analyze_run(prep, ms, ts; dirs = ckpt_dirs, model = m, metrics_toml = false)
            end
        end : nothing

    train_duration = @elapsed begin
        losses = train_model!(model, train_loader, val_loader, ts, static_arr;
                              fixed_eval          = fixed_eval,
                              checkpoint_callback = checkpoint_callback,
                              peak_stats          = peak_stats)
    end

    _log_gpu_memory("after training")
    train_rollout = losses.train_rollout
    val_rollout   = losses.val_rollout
    train_1step   = losses.train_1step
    val_1step     = losses.val_1step

    # --- 5. Persist artefacts ---
    @info "Saving artefacts to $(dirs.root)"

    save_data_settings( joinpath(dirs.model, "data_settings.toml"),  ds)
    save_model_settings(joinpath(dirs.model, "model_settings.toml"), ms)
    save_train_settings(joinpath(dirs.model, "train_settings.toml"), ts)

    # Normalisation statistics
    encode_stat(x) = x isa AbstractVector ? Float64.(x) : Float64(x)
    stats_dict = Dict(
        var => Dict{String,Any}("mean" => encode_stat(s.mean), "std" => encode_stat(s.std))
        for (var, s) in norm_stats
    )
    # Nonlinear-scaling parameters (only present for non-zscore variables).
    for (var, s) in norm_stats
        haskey(s, :scheme) || continue
        merge!(stats_dict[var], Dict{String,Any}(
            "scheme" => String(s.scheme), "scale" => encode_stat(s.scale),
            "nl_mean" => encode_stat(s.nl_mean), "nl_std" => encode_stat(s.nl_std)))
    end
    open(joinpath(dirs.model, "norm_stats.toml"), "w") do io
        TOML.print(io, stats_dict)
    end

    # Model weights (always saved on CPU)
    JLD2.jldsave(joinpath(dirs.model, "model.jld2");
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
                path     = joinpath(dirs.plots, "losses.png"),
                csv_path = joinpath(dirs.metrics, "losses.csv"))

    # Stability diagnostic (mass-balance runs only): q→h amplification for theta,
    # backbone contraction for the hybrid, plus the measured carried-state gain ρ̂.
    if (haskey(losses, :train_amp) && any(isfinite, losses.train_amp)) ||
       (haskey(losses, :mb_gain) && any(isfinite, losses.mb_gain))
        plot_amplification(losses.train_amp, losses.val_amp;
                           mb_gain     = get(losses, :mb_gain, nothing),
                           contraction = get(losses, :val_contraction, nothing),
                           kind        = mb_formulation_kind(ms),
                           path     = joinpath(dirs.plots, "amplification.png"),
                           csv_path = joinpath(dirs.metrics, "amplification.csv"))
    end

    # Fixed-horizon validation metric (discharge RMSE + peak ratio per epoch)
    if haskey(losses, :val_fixed_rmse) && any(isfinite, losses.val_fixed_rmse)
        plot_fixed_horizon(losses.val_fixed_rmse, losses.val_peak_ratio;
                           val_peak_ratio_frac_gt2 = get(losses, :val_peak_ratio_frac_gt2, nothing),
                           val_fixed_rmse_highflow = get(losses, :val_fixed_rmse_highflow, nothing),
                           horizon  = ts.eval_horizon,
                           path     = joinpath(dirs.plots, "fixed_horizon.png"),
                           csv_path = joinpath(dirs.metrics, "fixed_horizon.csv"))

        if fixed_eval !== nothing
            fh_final = fixed_horizon_metrics(model, fixed_eval; device = ts.device)
            write_fixed_horizon_anchor_table(
                joinpath(dirs.metrics, "fixed_horizon_anchors.csv"),
                fixed_eval,
                fh_final.rmse_q_anchor,
                fh_final.peak_ratio_anchor)
        end
    end

    # Training summary for stage 4 (`metrics.toml`) when it runs without the
    # in-memory training state.
    n_params = sum(length, Flux.trainables(Flux.cpu(model)))
    JLD2.jldsave(joinpath(dirs.model, "train_summary.jld2");
                 losses = losses, train_duration_s = train_duration, n_params = n_params)

    return (; model, losses, train_duration_s = train_duration, n_params)
end

"""
    load_train_summary(model_dir) -> (; losses, train_duration_s, n_params)

Load the training summary written by [`train_run`](@ref) from
`<model_dir>/train_summary.jld2`.
"""
function load_train_summary(model_dir::AbstractString)
    file = joinpath(model_dir, "train_summary.jld2")
    isfile(file) ||
        throw(ArgumentError("no training summary at $file (run train_run first)"))
    d = JLD2.load(file)
    return (losses = d["losses"], train_duration_s = d["train_duration_s"],
            n_params = d["n_params"])
end

"""
    load_model(prep, ms, model_dir; batch_size = 1) -> WflowGNN

Rebuild the model for `ms` on the topology/statistics in `prep` (see
[`build_gnn_model`](@ref)) and load the trained weights from
`<model_dir>/model.jld2` (written by [`train_run`](@ref)). The returned model is
on the CPU; move it with `Flux.gpu` for GPU inference. This is how stages 3 and 4
run on an already-trained run without retraining.
"""
function load_model(prep::PrepBundle, ms::ModelSettings, model_dir::AbstractString;
                    batch_size::Int = 1)
    file = joinpath(model_dir, "model.jld2")
    isfile(file) || throw(ArgumentError("no model weights at $file (run train_run first)"))
    model = build_gnn_model(ms, prep, batch_size)
    Flux.loadmodel!(model, JLD2.load(file, "model_state"))
    return model
end

# Timestamps of the `n_frames` predicted frames of a split whose first graph sits
# at index `offset + 1` of the full series (frame i is graph `offset + 1 + i`).
_split_frame_times(all_times, offset::Int, n_frames::Int) =
    [all_times[clamp(offset + 1 + i, 1, length(all_times))] for i in 1:n_frames]

"""
    infer_run(model, prep, ms, ts; dirs, splits = (:train, :val)) -> (; val_rollout_duration_s, val_n_timesteps)

Stage 3 of the pipeline: pure forward model. For every split in `splits`, run the
autoregressive trajectory rollout ([`evaluate_trajectory`](@ref)) on `model`
(on `ts.device`), scatter it back onto the raster ([`regrid`](@ref)) and write

    <dirs.output>/<split>_pred.nc, <split>_true.nc

as gridded `(lon, lat, time)` NetCDF in the layout of wflow's `output.nc`
(`lon`/`lat` from `prep.staticmaps_file`, `time` axis from `prep.times`). No
metrics or plots are produced; [`analyze_run`](@ref) reads these files back.

When `:val ∈ splits` and `ts.val_daterange` is set, the validation windows that
fall inside the range are additionally rolled out and written as
`val_daterange_pred.nc` / `val_daterange_true.nc`.

`dirs` is any value with an `output` field (e.g. `_run_dirs(ds)`). Returns the
wall-clock seconds and frame count of the validation rollout (`0.0` / `0` when
`:val ∉ splits`).
"""
function infer_run(model, prep::PrepBundle, ms::ModelSettings, ts::TrainSettings;
                   dirs, splits = (:train, :val))
    mkpath(dirs.output)
    schema   = load_schema(prep.wflow_schema)
    dataset  = prep.dataset
    n_train  = length(dataset.train)
    offsets  = (train = 0, val = n_train, test = n_train + length(dataset.val))

    val_rollout_duration = 0.0
    val_n_timesteps      = 0

    for split in splits
        split in keys(offsets) ||
            throw(ArgumentError("unknown split $(repr(split)); expected one of $(keys(offsets))"))
        split_data = getfield(dataset, split)

        t0 = time_ns()
        p_states, t_states = evaluate_trajectory(
            model, split_data, prep.norm_stats, ms.domain, prep.static_arr;
            device = ts.device, postscale = prep.postscale)
        if split == :val
            val_rollout_duration = (time_ns() - t0) / 1e9
            val_n_timesteps      = size(p_states, 3)
        end
        p_grids = regrid(p_states, prep.grid, ms.domain)
        t_grids = regrid(t_states, prep.grid, ms.domain)

        split_times = _split_frame_times(prep.times, offsets[split], size(p_states, 3))

        write_regrid_to_netcdf(p_grids, prep.staticmaps_file, split_times,
                               joinpath(dirs.output, "$(split)_pred.nc"); schema)
        write_regrid_to_netcdf(t_grids, prep.staticmaps_file, split_times,
                               joinpath(dirs.output, "$(split)_true.nc"); schema)
    end

    if :val in splits && !isnothing(ts.val_daterange)
        _infer_val_daterange(model, prep, ms, ts, dirs.output, schema)
    end

    return (; val_rollout_duration_s = val_rollout_duration, val_n_timesteps)
end

# Optional date-range rollout on the validation split → `val_daterange_{pred,true}.nc`.
function _infer_val_daterange(model, prep::PrepBundle, ms::ModelSettings, ts::TrainSettings,
                              output_dir::AbstractString, schema)
    dr_start, dr_stop = ts.val_daterange
    all_times  = prep.times
    split_data = prep.dataset.val
    n_train_w  = length(prep.dataset.train)

    # Find which val-split windows fall within the date range. The window
    # starting at split index `w` has its initial-condition graph at
    # `val_graph_times[w]`; take the first window whose initial-state time is
    # ≥ dr_start and run until the last one ≤ dr_stop.
    val_graph_times = [all_times[clamp(n_train_w + i, 1, length(all_times))]
                       for i in 1:length(split_data)]
    w_start = findfirst(t -> t ≥ dr_start, val_graph_times)
    w_stop  = findlast( t -> t ≤ dr_stop,  val_graph_times)

    if isnothing(w_start) || isnothing(w_stop) || w_start > w_stop
        @warn "val_daterange $dr_start – $dr_stop does not overlap the val split; skipping"
        return false
    end

    dr_p_states, dr_t_states = evaluate_trajectory(
        model, split_data[w_start:w_stop], prep.norm_stats, ms.domain, prep.static_arr;
        device = ts.device, postscale = prep.postscale)

    # evaluate_trajectory flattens windows into a consecutive sequence and returns
    # nhorizon + n_windows - 2 frames — more than n_windows — so derive the
    # timestamps from the series for the exact frame count.
    dr_times = _split_frame_times(all_times, n_train_w + w_start - 1, size(dr_p_states, 3))

    write_regrid_to_netcdf(regrid(dr_p_states, prep.grid, ms.domain), prep.staticmaps_file,
                           dr_times, joinpath(output_dir, "val_daterange_pred.nc"); schema)
    write_regrid_to_netcdf(regrid(dr_t_states, prep.grid, ms.domain), prep.staticmaps_file,
                           dr_times, joinpath(output_dir, "val_daterange_true.nc"); schema)
    return true
end
