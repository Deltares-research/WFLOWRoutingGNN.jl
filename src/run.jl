import JLD2
import Dates
using Random
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

[eval]
upstream_points = 5
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
    settings_from_config(d, toml_dir) -> (ds, ms, ts)

Build `(DataSettings, ModelSettings, TrainSettings)` from an already-parsed
config `Dict` `d` (the four `[data]`/`[model]`/`[train]`/`[train.strategy]`
tables), resolving relative `[data]` paths against `toml_dir`.

This is the single source of truth for the dict→settings mapping, shared by
[`parse_run_config`](@ref) (path → loaded TOML → here) and the hyperparameter
search (`hpar_search`, per-combo mutated dict → here) so both interpret configs
identically and new config keys land in both paths automatically. The optional
`[train]` keys `lr_warmup_epochs` and `lr_peak_decay` feed the curriculum LR
schedule; `lr_steps` is retained for backward compatibility but unused by it.
"""
function settings_from_config(d::AbstractDict, toml_dir::AbstractString)
    toml_dir_abs = abspath(toml_dir)

    haskey(d, "data")  || throw(ArgumentError("TOML missing [data] table"))
    haskey(d, "model") || throw(ArgumentError("TOML missing [model] table"))
    haskey(d, "train") || throw(ArgumentError("TOML missing [train] table"))

    # Resolve relative paths against the directory of the TOML file
    resolve(p) = isabspath(p) ? p : normpath(joinpath(toml_dir_abs, p))

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
        mb_formulation  = String(get(md, "mb_formulation", "theta_kinematic_wave")),
        mb_theta        = Float32(get(md, "mb_theta", 1.0)),
        mb_augment_decoder = get(md, "mb_augment_decoder", false),
        include_log_upstream_area = get(md, "include_log_upstream_area", false),
        mb_smooth_h_floor = get(md, "mb_smooth_h_floor", false),
        mb_h_floor_softness = Float32(get(md, "mb_h_floor_softness", 0.05)),
        mb_sigma_h_floor = Float32(get(md, "mb_sigma_h_floor", 0.0)),
        mb_sigma_q_floor = Float32(get(md, "mb_sigma_q_floor", 0.0)),
        mb_sigma_inwater_floor = Float32(get(md, "mb_sigma_inwater_floor", 0.0)),
    )

    td = d["train"]
    ed = get(d, "eval", Dict{String,Any}())
    sd = get(td, "strategy", Dict{String,Any}())
    strategy = TrainingStrategy(
        get(sd, "steps",       [1]),
        get(sd, "durations",   [td["epochs"]]),
        get(sd, "noise_scale", 0.0);
        pushforward_tf_weight = get(sd, "pushforward_tf_weight", 0.0),
        loss_type = Symbol(get(sd, "loss_type", "mse")),
        peak_delta = get(sd, "peak_delta", 1.0),
        peak_lambda = get(sd, "peak_lambda", 0.0),
        peak_gamma = get(sd, "peak_gamma", 1.0),
        peak_w_max = get(sd, "peak_w_max", 4.0),
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
        rollout_grad     = Symbol(get(td, "rollout_grad", "full_bptt")),
        seed             = haskey(td, "seed") ? Int(td["seed"]) : nothing,
        eval_horizon     = get(td, "eval_horizon", 30),
        eval_anchors     = get(td, "eval_anchors", 32),
        upstream_points  = get(ed, "upstream_points", get(td, "upstream_points", 5)),
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
    parse_run_config(toml_path) -> (ds, ms, ts)

Load an experiment config TOML from `toml_path` and parse it into
`(DataSettings, ModelSettings, TrainSettings)` without running any training,
via [`settings_from_config`](@ref). Relative paths in `[data]` are resolved
relative to the directory containing the TOML file.

Shared by [`run_wflow_gnn_from_toml`](@ref) and standalone tuning scripts (e.g.
the LR range test) so they interpret configs identically.
"""
function parse_run_config(toml_path::String)
    isfile(toml_path) || throw(ArgumentError("TOML file not found: $toml_path"))
    toml_dir = dirname(abspath(toml_path))
    d        = TOML.parsefile(toml_path)
    return settings_from_config(d, toml_dir)
end

"""
    build_gnn_model(ms, graphs, norm_stats, postscale, output_file, batch_size;
                    strategy = nothing, h_loss_scale = :absolute) -> WflowGNN

Construct a `WflowGNN` (with block-diagonal adjacency pre-computed for
`batch_size`) from a built graph time series. For the `"river"` domain this also
builds the mass-balance formulation (see [`mb_formulation_kind`](@ref)) and, when
`strategy` is supplied, sets
`strategy.h_loss_weight` to the mass-balance-consistent value.

`h_loss_scale` selects how the water-depth term of the loss is normalised:
- `:absolute`  → weight `σ_h/(dt·σ_q)` (balances the q- and h-loss magnitudes).
- `:increment` → weight `(σ_h/(dt·σ_q))²`, i.e. h-error measured on the
  mass-balance increment scale `dt·σ_q`. This makes `∂h_norm/∂q_norm ≈ O(1)`
  and removes the stiff gradient amplification of the hard mass-balance decoder
  (at the cost of weaker h-supervision once discharge is well fit).

A second method, `build_gnn_model(ms, prep::PrepBundle, batch_size; strategy,
h_loss_scale)`, takes the topology, statistics and timestep from a
[`PrepBundle`](@ref) (stage 2 of the pipeline: no raw wflow files are read).

Shared by [`train_run`](@ref) and standalone tuning scripts so the model is
built identically everywhere.
"""
function build_gnn_model(ms::ModelSettings, graphs, norm_stats, postscale,
                         output_file::AbstractString, batch_size::Int;
                         strategy::Union{Nothing,TrainingStrategy} = nothing,
                         h_loss_scale::Symbol = :absolute)
    g0 = graphs[1]
    src_edges, tgt_edges = edge_index(g0)
    return _build_gnn_model(ms, g0.num_nodes, src_edges, tgt_edges, norm_stats,
                            postscale, get_timestep(output_file), batch_size;
                            strategy, h_loss_scale)
end

function build_gnn_model(ms::ModelSettings, prep::PrepBundle, batch_size::Int;
                         strategy::Union{Nothing,TrainingStrategy} = nothing,
                         h_loss_scale::Symbol = :absolute)
    return _build_gnn_model(ms, prep.n_nodes, prep.edge_src, prep.edge_tgt,
                            prep.norm_stats, prep.postscale, prep.dt, batch_size;
                            strategy, h_loss_scale)
end

function _build_gnn_model(ms::ModelSettings, n_nodes::Int, src_edges, tgt_edges,
                          norm_stats, postscale, dt::Real, batch_size::Int;
                          strategy::Union{Nothing,TrainingStrategy},
                          h_loss_scale::Symbol)
    # Sparse adjacency with self-loops: A[i,j]=1 means node j is an upstream
    # neighbour of node i (each node also aggregates its own state).
    all_src  = vcat(src_edges, collect(1:n_nodes))
    all_tgt  = vcat(tgt_edges, collect(1:n_nodes))
    A_sparse = sparse(all_tgt, all_src, ones(Float32, length(all_src)), n_nodes, n_nodes)

    mb = _build_mb_formulation(Val(mb_formulation_kind(ms)), ms, norm_stats, postscale,
                               src_edges, tgt_edges, dt)
    h_weight = mb_h_loss_weight(mb, h_loss_scale)
    (strategy === nothing || h_weight === nothing) || (strategy.h_loss_weight = h_weight)
    model = WflowGNN(ms, mb, A_sparse)

    # Block-diagonal adjacency for the batch size. Must be built on CPU
    # (blockdiag needs SparseMatrixCSC); device movement happens at the call site.
    return precompute_batched(model, batch_size)
end

# Build the mass-balance formulation selected by `mb_formulation_kind(ms)`.
_build_mb_formulation(::Val{:none}, ms, norm_stats, postscale,
                      src_edges, tgt_edges, dt) = begin
    if ms.domain == "river"
        @info "Mass balance DISABLED: river_q and river_h are predicted " *
              "independently (decoder out_dim = " *
              "$(length(DOMAIN_VARS["river"]["state"])))"
    end
    NoMassBalance()
end

_build_mb_formulation(::Val{:theta_kinematic_wave}, ms, norm_stats, postscale,
                      src_edges, tgt_edges, dt) =
    build_massbalance(norm_stats, postscale, src_edges, tgt_edges, dt;
                      θ = ms.mb_theta,
                      smooth_h_floor = ms.mb_smooth_h_floor,
                      h_floor_softness = ms.mb_h_floor_softness)

function _write_mb_floor_fraction_by_node(path::AbstractString,
                                          frac_h_raw_neg_node::AbstractVector,
                                          n_valid_h_raw_node::AbstractVector{<:Integer})
    n_nodes = length(frac_h_raw_neg_node)
    length(n_valid_h_raw_node) == n_nodes ||
        throw(ArgumentError("floor-fraction vectors must have matching lengths"))
    open(path, "w") do io
        println(io, "node_idx,frac_h_raw_neg,n_valid,n_neg")
        for i in 1:n_nodes
            n_valid = Int(n_valid_h_raw_node[i])
            if n_valid == 0
                println(io, string(i, ",NaN,0,0"))
            else
                frac  = Float32(frac_h_raw_neg_node[i])
                n_neg = round(Int, frac * n_valid)
                println(io, string(i, ",", frac, ",", n_valid, ",", n_neg))
            end
        end
    end
    return path
end

function _mb_floor_summary(diags)
    frac_node = Float32.(diags.frac_h_raw_neg_node)
    valid_node = Int.(diags.n_valid_h_raw_node)

    finite_mask = isfinite.(frac_node)
    frac_finite = frac_node[finite_mask]
    n_nodes = length(frac_node)
    n_nodes_finite = length(frac_finite)

    # Network-wide fraction over all valid node-time samples.
    hvals = filter(isfinite, vec(diags.h_raw))
    frac_network = isempty(hvals) ? NaN32 : Float32(mean(hvals .< 0f0))

    if n_nodes_finite == 0
        return (;
            n_nodes,
            n_nodes_finite,
            frac_h_raw_neg_network = frac_network,
            frac_h_raw_neg_node_mean = NaN32,
            frac_h_raw_neg_node_median = NaN32,
            frac_h_raw_neg_node_p90 = NaN32,
            frac_h_raw_neg_node_max = NaN32,
            frac_h_raw_neg_node_max_idx = 0,
            n_valid_h_raw_total = sum(valid_node),
        )
    end

    frac_sorted = sort(frac_finite)
    p90_idx = clamp(round(Int, 0.9 * length(frac_sorted)), 1, length(frac_sorted))
    frac_p90 = frac_sorted[p90_idx]

    max_frac = maximum(frac_node)
    max_idx = findfirst(==(max_frac), frac_node)

    return (;
        n_nodes,
        n_nodes_finite,
        frac_h_raw_neg_network = frac_network,
        frac_h_raw_neg_node_mean = Float32(mean(frac_finite)),
        frac_h_raw_neg_node_median = Float32(median(frac_finite)),
        frac_h_raw_neg_node_p90 = Float32(frac_p90),
        frac_h_raw_neg_node_max = Float32(max_frac),
        frac_h_raw_neg_node_max_idx = isnothing(max_idx) ? 0 : Int(max_idx),
        n_valid_h_raw_total = sum(valid_node),
    )
end

function _write_mb_stiffness_by_node(path::AbstractString, mb::ThetaKinematicWave)
    gain = mb.θ .* mb.dt .* mb.σ_q ./ mb.σ_h
    open(path, "w") do io
        println(io, "node_idx,mb_gain,mb_base_weight,sigma_q,sigma_h")
        for i in eachindex(gain)
            g = Float32(gain[i])
            b = isfinite(g) && g != 0f0 ? Float32(1f0 / g) : NaN32
            println(io, string(i, ",", g, ",", b, ",", Float32(mb.σ_q[i]), ",", Float32(mb.σ_h[i])))
        end
    end
    return path
end

function _mb_stiffness_summary(mb::ThetaKinematicWave)
    gain = Float32.(mb.θ .* mb.dt .* mb.σ_q ./ mb.σ_h)
    fg = sort(Float32[x for x in gain if isfinite(x)])
    n = length(gain)
    if isempty(fg)
        return (n_nodes = n, n_finite = 0,
                gain_median = NaN32, gain_p90 = NaN32, gain_max = NaN32,
                gain_max_idx = 0)
    end
    p90_idx = clamp(round(Int, 0.9 * length(fg)), 1, length(fg))
    gmax = maximum(gain)
    max_idx = findfirst(==(gmax), gain)
    return (n_nodes = n,
            n_finite = length(fg),
            gain_median = Float32(median(fg)),
            gain_p90 = Float32(fg[p90_idx]),
            gain_max = Float32(gmax),
            gain_max_idx = isnothing(max_idx) ? 0 : Int(max_idx))
end

"""
    write_run_metrics_toml(path, losses, ts, run_meta, spatial_summary, ramp,
                           river_q_perf, volume_budget, mb_floor, mb_stiffness) -> path

Write a compact scalar summary of a completed run to `path` as TOML: the final
(and best) values of the per-epoch training history, run metadata (`run_meta`),
the median of each aggregated spatial-error metric per state variable, the
overprediction-vs-ramp correlations, the Tier-2 river-discharge performance
metrics (`river_q_perf` from [`river_q_performance_metrics`](@ref)) and, when
available, the system volume-budget diagnostics (`volume_budget` from
[`volume_budget_diagnostics`](@ref)); the mass-balance floor diagnostics summary
(`mb_floor`, derived from validation `h_raw`); the per-node MB stiffness summary
(`mb_stiffness`, from `θ·dt·σ_q/σ_h`); and, when `losses.loss_type == :huber`, the Tier-1 peak-loss diagnostics from
[`peak_epoch_diagnostics`](@ref). Non-finite values are omitted so the file
stays a valid, parser-friendly TOML of plain numbers — a token-cheap single-file
entry point for downstream evaluation agents.
"""
function write_run_metrics_toml(path::AbstractString, losses, ts::TrainSettings,
                                run_meta, spatial_summary, ramp,
                                river_q_perf = nothing, volume_budget = nothing,
                                mb_floor = nothing, mb_stiffness = nothing)
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
    putf!(run_t, "seed",                   ts.seed)
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
    putf!(fh_t, "final_peak_ratio_frac_gt2", lastf(get(losses, :val_peak_ratio_frac_gt2, nothing)))
    putf!(fh_t, "final_val_rmse_highflow",   lastf(get(losses, :val_fixed_rmse_highflow, nothing)))
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

    if river_q_perf !== nothing
        function _gauge_metrics!(d, m)
            putf!(d, "kge",        m.kge)
            putf!(d, "r",          m.r)
            putf!(d, "alpha",      m.alpha)
            putf!(d, "beta",       m.beta)
            putf!(d, "mae",        m.mae)
            putf!(d, "pbias",      m.pbias)
            putf!(d, "peak_error", m.peak_error)
            putf!(d, "peak_ratio", m.peak_ratio)
            putf!(d, "fhv",        m.fhv)
        end
        rq_t     = Dict{String, Any}()
        pooled_t = Dict{String, Any}()
        _gauge_metrics!(pooled_t, river_q_perf.pooled)
        rq_t["pooled"] = pooled_t
        gauge_t = Dict{String, Any}()
        putf!(gauge_t, "outlet_idx", river_q_perf.gauge.outlet_idx)
        _gauge_metrics!(gauge_t, river_q_perf.gauge)
        rq_t["gauge"] = gauge_t
        root["river_q_performance"] = rq_t
    end

    if volume_budget !== nothing
        vb_t = Dict{String, Any}()
        putf!(vb_t, "outlet_idx", volume_budget.outlet_idx)
        putf!(vb_t, "volume_pbias", volume_budget.volume_pbias)
        putf!(vb_t, "volume_drift", volume_budget.volume_drift)
        putf!(vb_t, "budget_residual_rms", volume_budget.budget_residual_rms)
        root["volume_budget"] = vb_t
    end

    if mb_floor !== nothing
        mbf_t = Dict{String, Any}()
        putf!(mbf_t, "n_nodes", mb_floor.n_nodes)
        putf!(mbf_t, "n_nodes_finite", mb_floor.n_nodes_finite)
        putf!(mbf_t, "n_valid_h_raw_total", mb_floor.n_valid_h_raw_total)
        putf!(mbf_t, "frac_h_raw_neg_network", mb_floor.frac_h_raw_neg_network)
        putf!(mbf_t, "frac_h_raw_neg_node_mean", mb_floor.frac_h_raw_neg_node_mean)
        putf!(mbf_t, "frac_h_raw_neg_node_median", mb_floor.frac_h_raw_neg_node_median)
        putf!(mbf_t, "frac_h_raw_neg_node_p90", mb_floor.frac_h_raw_neg_node_p90)
        putf!(mbf_t, "frac_h_raw_neg_node_max", mb_floor.frac_h_raw_neg_node_max)
        putf!(mbf_t, "frac_h_raw_neg_node_max_idx", mb_floor.frac_h_raw_neg_node_max_idx)
        root["mass_balance_floor"] = mbf_t
    end

    if mb_stiffness !== nothing
        mbs_t = Dict{String, Any}()
        putf!(mbs_t, "n_nodes", mb_stiffness.n_nodes)
        putf!(mbs_t, "n_finite", mb_stiffness.n_finite)
        putf!(mbs_t, "gain_median", mb_stiffness.gain_median)
        putf!(mbs_t, "gain_p90", mb_stiffness.gain_p90)
        putf!(mbs_t, "gain_max", mb_stiffness.gain_max)
        putf!(mbs_t, "gain_max_idx", mb_stiffness.gain_max_idx)
        root["mass_balance_stiffness"] = mbs_t
    end

    if any(isfinite, get(losses, :peak_c_peak, Float32[]))
        peak_t = Dict{String, Any}()
        putf!(peak_t, "final_c_peak",        lastf(losses.peak_c_peak))
        putf!(peak_t, "final_rmse_high",     lastf(losses.peak_rmse_high))
        putf!(peak_t, "final_mae_high",      lastf(losses.peak_mae_high))
        putf!(peak_t, "final_w_mean",        lastf(losses.peak_w_mean))
        putf!(peak_t, "final_w_max",         lastf(losses.peak_w_max))
        putf!(peak_t, "final_w_min",         lastf(losses.peak_w_min))
        putf!(peak_t, "final_q_grad_norm",   lastf(losses.peak_q_grad_norm))
        putf!(peak_t, "final_h_grad_norm",   lastf(losses.peak_h_grad_norm))
        putf!(peak_t, "final_peak_grad_frac", lastf(losses.peak_grad_frac))
        root["peak_loss"] = peak_t
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
1. [`preprocess_run`](@ref): build the `GNNGraph` time series from
   `ds.wflow_model_path` (`staticmaps.nc` + `run_default/output.nc`) for domain
   `ms.domain`, create sliding-window samples with horizon
   `maximum(ts.strategy.steps) + 1`, split into train / val / test using
   `ds.train_frac` / `ds.val_frac`, and persist the [`PrepBundle`](@ref).
2. [`train_run`](@ref): build a `WflowGNN` from `ms` and the bundle and train it
   with `train_model!`.
3. [`infer_run`](@ref): roll the final model out on train/val and write the
   gridded `<split>_pred.nc` / `<split>_true.nc`.
4. [`analyze_run`](@ref): metrics, plots and `metrics.toml` from those NetCDFs.

All artefacts go under `<ds.runs_dir>/<ds.run_name>/`:
       prep/
           preprocessed.jld2      (versioned `PrepBundle`, see `prep_schema`)
       model/
           data_settings.toml
           model_settings.toml
           train_settings.toml
           norm_stats.toml
           model.jld2
           train_summary.jld2
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

    # --- Stage 1: preprocess → prep/ ---
    prep = preprocess_run(ds, ms, ts)

    # --- Stage 2: train → model/ ---
    tr = train_run(prep, ds, ms, ts)
    model          = tr.model
    losses         = tr.losses
    train_duration = tr.train_duration_s
    train_rollout  = losses.train_rollout
    val_rollout    = losses.val_rollout

    dirs = _run_dirs(ds)

    # --- Stage 3: infer → output/<split>_{pred,true}.nc ---
    @info "Evaluating train and val trajectories"
    inf = infer_run(model, prep, ms, ts; dirs)

    # --- Stage 4: analyze → metrics/, plots/, output/spatial_metrics.nc ---
    analyze_run(prep, ms, ts; dirs, model, train_summary = tr, infer = inf)

    metrics = (
        final_train_loss           = last(train_rollout),
        final_val_loss             = last(val_rollout),
        n_params                   = tr.n_params,
        train_duration_s           = train_duration,
        val_rollout_duration_s     = inf.val_rollout_duration_s,
        val_n_timesteps            = inf.val_n_timesteps,
    )
    return model, metrics
end
