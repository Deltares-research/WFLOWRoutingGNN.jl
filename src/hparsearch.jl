import TOML
import Dates
using Statistics

# ---------------------------------------------------------------------------
# HParSearchSettings
# ---------------------------------------------------------------------------

"""
    HParSearchSettings

Configuration for a hyperparameter search.

Fields:
- `search_type`  : search strategy. Currently only `"box"` is supported, which
                   performs an exhaustive grid search (Cartesian product) over
                   all values in `search_space`.
- `search_space` : `Dict{String, Vector{Any}}` mapping parameter names to the
                   list of values to try. Parameter names use dot-separated
                   paths that mirror the TOML table structure, e.g.
                   `"model.hidden_dim"`, `"train.lr_start"`,
                   `"train.strategy.steps"`.
"""
struct HParSearchSettings
    search_type  :: String
    search_space :: Dict{String, Vector{Any}}
    seeds        :: Vector{Int}
end

const SUPPORTED_SEARCH_TYPES = ("box",)

"""
    HParSearchSettings(; search_type, search_space) -> HParSearchSettings
"""
function HParSearchSettings(;
    search_type  :: String,
    search_space :: Dict,
    seeds        :: AbstractVector{<:Integer} = Int[])

    search_type in SUPPORTED_SEARCH_TYPES ||
        throw(ArgumentError("search_type must be one of " *
                            join(SUPPORTED_SEARCH_TYPES, ", ") *
                            "; got \"$search_type\""))
    seeds_v = Int[seeds...]
    all(>=(0), seeds_v) ||
        throw(ArgumentError("all seeds must be non-negative"))

    HParSearchSettings(
        search_type,
        Dict{String, Vector{Any}}(k => collect(Any, v) for (k, v) in search_space),
        seeds_v,
    )
end

function Base.show(io::IO, s::HParSearchSettings)
    println(io, "HParSearchSettings:")
    println(io, "  search_type : ", s.search_type)
    println(io, "  search_space:")
    for (k, v) in sort(collect(s.search_space); by = first)
        println(io, "    ", k, " : ", v)
    end
    print(io, "  seeds       : ", isempty(s.seeds) ? "(default: single unseeded run)" : string(s.seeds))
end

# ---------------------------------------------------------------------------
# TOML IO
# ---------------------------------------------------------------------------

"""
    save_hpar_search_settings(path, settings)

Write `settings` to a TOML file at `path`.
"""
function save_hpar_search_settings(path::String, s::HParSearchSettings)
    dict = Dict(
        "search_type"  => s.search_type,
        "search_space" => Dict(k => v for (k, v) in s.search_space),
    )
    isempty(s.seeds) || (dict["seeds"] = s.seeds)
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_hpar_search_settings(path) -> HParSearchSettings

Read a `HParSearchSettings` from the TOML file at `path`.
"""
function load_hpar_search_settings(path::String)
    d  = TOML.parsefile(path)
    ss = d["search_space"]
    return HParSearchSettings(
        search_type  = d["search_type"],
        search_space = Dict{String, Vector{Any}}(
            k => collect(Any, v) for (k, v) in ss),
        seeds = haskey(d, "seeds") ? Int.(d["seeds"]) : Int[],
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Apply a dot-path override to a nested Dict parsed from TOML.
# e.g. _set!(d, "model.hidden_dim", 128) sets d["model"]["hidden_dim"] = 128
function _set!(d::Dict, path::String, value)
    parts = split(path, ".")
    node  = d
    for p in parts[1:end-1]
        node = get!(node, p, Dict{String,Any}())
    end
    node[parts[end]] = value
end

# `missing` / `nothing` mark a metric that does not apply to the run's formulation
# (e.g. a theta-only companion): written as an empty CSV cell, never as `NaN`.
_hps_cell(v) = (v === missing || v === nothing) ? "" : string(v)
_hps_present(x) = !(x === missing || x === nothing) && isfinite(x)

# Write a Vector of NamedTuples as a CSV file.
function _write_hps_csv(path::String, rows::Vector)
    isempty(rows) && return
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows
            println(io, join(_hps_cell.(values(row)), ","))
        end
    end
end

# Per-formulation stability companions and secondary conservation numbers: only
# written to the summary TOML for combinations where they exist.
const _HPS_OPTIONAL_METRICS = (:volume_pbias, :mb_residual_rms, :analytic_gain,
                               :q_to_h_amp, :backbone_rho_median, :backbone_rho_max,
                               :resid_frac_cap_pos, :resid_frac_cap_neg)

function _hps_add_optional!(entry::Dict, r)
    for k in _HPS_OPTIONAL_METRICS
        m = getfield(r, Symbol(k, "_mean"))
        _hps_present(m) && (entry[String(k)] = Dict("mean" => m))
    end
    return entry
end

function _write_hps_summary_toml(path::String, rows::Vector)
    mkpath(dirname(path))
    payload = Dict(
        "generated_at" => string(Dates.now()),
        "n_combinations" => length(rows),
        "combinations" => [
            _hps_add_optional!(Dict(
                "combo_index" => r.combo_index,
                "run_name_base" => r.run_name_base,
                "n_repetitions" => r.n_repetitions,
                "final_val_loss" => Dict(
                    "mean" => r.final_val_loss_mean,
                    "std"  => r.final_val_loss_std,
                    "min"  => r.final_val_loss_min,
                    "max"  => r.final_val_loss_max,
                ),
                "fixed_val_rmse" => Dict(
                    "mean" => r.fixed_val_rmse_mean,
                    "std"  => r.fixed_val_rmse_std,
                    "min"  => r.fixed_val_rmse_min,
                    "max"  => r.fixed_val_rmse_max,
                ),
                "peak_ratio_frac_gt2" => Dict(
                    "mean" => r.peak_ratio_frac_gt2_mean,
                    "std"  => r.peak_ratio_frac_gt2_std,
                ),
                "pooled_kge" => Dict(
                    "mean" => r.pooled_kge_mean,
                    "std"  => r.pooled_kge_std,
                ),
                "river_h_nse" => Dict(
                    "mean" => r.river_h_nse_mean,
                    "std"  => r.river_h_nse_std,
                ),
                "rollout_contraction" => Dict(
                    "mean" => r.rollout_contraction_mean,
                    "std"  => r.rollout_contraction_std,
                ),
            ), r) for r in rows
        ],
    )
    open(path, "w") do io
        TOML.print(io, payload)
    end
end

# Generate all Cartesian-product combinations for a box search.
# Returns a Vector of Dicts mapping parameter name to value.
function _box_combinations(search_space::Dict{String, Vector{Any}})
    keys_   = collect(keys(search_space))
    values_ = [search_space[k] for k in keys_]
    combos  = Dict{String, Any}[]
    function recurse(idx, current)
        if idx > length(keys_)
            push!(combos, copy(current))
            return
        end
        for v in values_[idx]
            current[keys_[idx]] = v
            recurse(idx + 1, current)
        end
    end
    recurse(1, Dict{String, Any}())
    return combos
end

# Resolve optional multi-seed replication from [hparsearch].
# - seeds = [..] takes precedence and must be non-empty if supplied.
# - repetitions = R expands to seeds 1:R.
# - if neither is set, returns Int[] (single run, no explicit seeding override).
function _resolve_replication_seeds(hps_d::AbstractDict)
    has_seeds = haskey(hps_d, "seeds")
    has_reps  = haskey(hps_d, "repetitions")
    if has_seeds && has_reps
        throw(ArgumentError("Use either [hparsearch].seeds or repetitions, not both"))
    end
    if has_seeds
        seeds = Int.(hps_d["seeds"])
        isempty(seeds) && throw(ArgumentError("[hparsearch].seeds must not be empty"))
        all(>=(0), seeds) || throw(ArgumentError("all [hparsearch].seeds must be non-negative"))
        return seeds
    end
    if has_reps
        reps = Int(hps_d["repetitions"])
        reps > 0 || throw(ArgumentError("[hparsearch].repetitions must be positive"))
        return collect(1:reps)
    end
    return Int[]
end

function _read_run_scalar_metrics(run_dir::AbstractString)
    path = joinpath(run_dir, "metrics", "metrics.toml")
    isfile(path) || return (fixed_val_rmse = NaN,
                            peak_ratio_frac_gt2 = NaN,
                            pooled_kge = NaN,
                            river_h_nse = NaN,
                            rollout_contraction = NaN,
                            mb_residual_rms = NaN,
                            volume_pbias = NaN,
                            volume_drift = NaN,
                            h_loss_weight_effective = NaN,
                            analytic_gain = missing,
                            q_to_h_amp = missing,
                            backbone_rho_median = missing,
                            backbone_rho_max = missing,
                            resid_frac_cap_pos = missing,
                            resid_frac_cap_neg = missing)
    d = TOML.parsefile(path)
    fh = get(d, "fixed_horizon", Dict{String,Any}())
    rp = get(d, "river_q_performance", Dict{String,Any}())
    rp_pooled = get(rp, "pooled", Dict{String,Any}())
    sp = get(d, "spatial_median", Dict{String,Any}())
    sp_h = get(sp, "river_h", Dict{String,Any}())
    stab = get(d, "stability", Dict{String,Any}())
    form = get(d, "formulation", Dict{String,Any}())
    mbr  = get(d, "mass_balance_residual", Dict{String,Any}())
    mbr_pred = get(mbr, "pred", Dict{String,Any}())
    rsat = get(d, "residual_saturation", Dict{String,Any}())
    # `missing` when the key is absent: the metric does not apply to this formulation
    opt(t, k) = haskey(t, k) ? Float64(t[k]) : missing
    return (
        fixed_val_rmse = Float64(get(fh, "final_val_rmse", NaN)),
        peak_ratio_frac_gt2 = Float64(get(fh, "final_peak_ratio_frac_gt2", NaN)),
        pooled_kge = Float64(get(rp_pooled, "kge", NaN)),
        river_h_nse = Float64(get(sp_h, "nse", NaN)),
        # headline, formulation-agnostic
        rollout_contraction = Float64(get(stab, "rollout_contraction", NaN)),
        mb_residual_rms = Float64(get(mbr_pred, "rms", NaN)),
        volume_pbias = Float64(get(mbr, "volume_pbias", NaN)),
        volume_drift = Float64(get(mbr, "volume_drift", NaN)),
        h_loss_weight_effective = Float64(get(form, "h_loss_weight", NaN)),
        # per-formulation companions
        analytic_gain = opt(stab, "analytic_gain"),
        q_to_h_amp = opt(stab, "q_to_h_amp"),
        backbone_rho_median = opt(stab, "backbone_rho_median"),
        backbone_rho_max = opt(stab, "backbone_rho_max"),
        resid_frac_cap_pos = opt(rsat, "frac_cap_pos"),
        resid_frac_cap_neg = opt(rsat, "frac_cap_neg"),
    )
end

function _aggregate_repetition_rows(rows::Vector)
    isempty(rows) && return NamedTuple[]
    combo_ids = sort(unique(getfield.(rows, :combo_index)))
    out = NamedTuple[]
    for combo_idx in combo_ids
        grp = filter(r -> r.combo_index == combo_idx, rows)
        function _stats(v)
            fv = Float64[x for x in v if _hps_present(x)]
            isempty(fv) && return (mean = NaN, std = NaN, min = NaN, max = NaN)
            return (mean = mean(fv),
                    std  = length(fv) > 1 ? std(fv) : 0.0,
                    min  = minimum(fv),
                    max  = maximum(fv))
        end

        st_val = _stats(getfield.(grp, :final_val_loss))
        st_fh  = _stats(getfield.(grp, :fixed_val_rmse))
        st_pg2 = _stats(getfield.(grp, :peak_ratio_frac_gt2))
        st_kge = _stats(getfield.(grp, :pooled_kge))
        st_hn  = _stats(getfield.(grp, :river_h_nse))
        st_rho = _stats(getfield.(grp, :rollout_contraction))
        # mean over the repetitions where the metric applies; `missing` if none does
        opt_mean(k) = (s = _stats(getfield.(grp, k)); isfinite(s.mean) ? s.mean : missing)

        push!(out, (
            combo_index = combo_idx,
            run_name_base = first(grp).run_name_base,
            n_repetitions = length(grp),
            final_val_loss_mean = st_val.mean,
            final_val_loss_std  = st_val.std,
            final_val_loss_min  = st_val.min,
            final_val_loss_max  = st_val.max,
            fixed_val_rmse_mean = st_fh.mean,
            fixed_val_rmse_std  = st_fh.std,
            fixed_val_rmse_min  = st_fh.min,
            fixed_val_rmse_max  = st_fh.max,
            peak_ratio_frac_gt2_mean = st_pg2.mean,
            peak_ratio_frac_gt2_std  = st_pg2.std,
            pooled_kge_mean = st_kge.mean,
            pooled_kge_std  = st_kge.std,
            river_h_nse_mean = st_hn.mean,
            river_h_nse_std  = st_hn.std,
            rollout_contraction_mean = st_rho.mean,
            rollout_contraction_std  = st_rho.std,
            mb_residual_rms_mean     = opt_mean(:mb_residual_rms),
            volume_pbias_mean        = opt_mean(:volume_pbias),
            analytic_gain_mean       = opt_mean(:analytic_gain),
            q_to_h_amp_mean          = opt_mean(:q_to_h_amp),
            backbone_rho_median_mean = opt_mean(:backbone_rho_median),
            backbone_rho_max_mean    = opt_mean(:backbone_rho_max),
            resid_frac_cap_pos_mean  = opt_mean(:resid_frac_cap_pos),
            resid_frac_cap_neg_mean  = opt_mean(:resid_frac_cap_neg),
        ))
    end
    return out
end

# ---------------------------------------------------------------------------
# Resume / failure handling helpers
# ---------------------------------------------------------------------------

# Core per-run numbers (the `metrics` NamedTuple of `run_wflow_gnn`) recovered from
# a finished run's `metrics/metrics.toml`. Non-finite values are absent from that
# file and come back as `NaN` / `0`.
function _read_run_core_metrics(run_dir::AbstractString)
    d = TOML.parsefile(joinpath(run_dir, "metrics", "metrics.toml"))
    r = get(d, "run",  Dict{String,Any}())
    l = get(d, "loss", Dict{String,Any}())
    return (
        final_train_loss       = Float64(get(l, "final_train_rollout", NaN)),
        final_val_loss         = Float64(get(l, "final_val_rollout",   NaN)),
        n_params               = Int(get(r, "n_params", 0)),
        train_duration_s       = Float64(get(r, "train_duration_s",       NaN)),
        val_rollout_duration_s = Float64(get(r, "val_rollout_duration_s", NaN)),
        val_n_timesteps        = Int(get(r, "val_n_timesteps", 0)),
    )
end

# A cell counts as finished when its terminal artifacts (`metrics/metrics.toml`, written
# last by the analysis stage, and the gridded `output/<split>_{pred,true}.nc`) exist AND
# the settings it was produced with equal the settings this search would use now. The
# latter guards against silently reusing a cell left behind by an earlier, different
# configuration (stricter than the stage-level `resume` of `run_wflow_gnn`).
function _hps_run_is_complete(run_dir::AbstractString, ds, ms, ts)
    dirs = _run_dirs(run_dir)
    isfile(joinpath(dirs.metrics, "metrics.toml")) || return false
    all(f -> isfile(joinpath(dirs.output, f)),
        ("train_pred.nc", "train_true.nc", "val_pred.nc", "val_true.nc")) || return false
    mdir = dirs.model
    return mktempdir() do tmp
        for (name, save!, s) in (("data_settings.toml",  save_data_settings,  ds),
                                 ("model_settings.toml", save_model_settings, ms),
                                 ("train_settings.toml", save_train_settings, ts))
            saved = joinpath(mdir, name)
            isfile(saved) || return false
            fresh = joinpath(tmp, name)
            save!(fresh, s)
            TOML.parsefile(saved) == TOML.parsefile(fresh) || return false
        end
        return true
    end
end

# Release the Julia-side and (if present) CUDA-side memory of the finished run so
# successive runs do not accumulate GPU allocations.
function _hps_free_memory()
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()
    return nothing
end

_hps_oneline(e) = strip(replace(first(split(sprint(showerror, e), '\n')), r"[,\"\r]" => " "))

# (Re)write the results CSV, the failure list and, for replicated searches, the
# per-combination summaries. Called after every run so partial results survive a
# crash or kill.
function _hps_write_outputs(runs_dir::AbstractString, base_name::AbstractString,
                            rows::Vector, failures::Vector, replicated::Bool)
    _write_hps_csv(joinpath(runs_dir, base_name * "_hps_results.csv"), rows)

    fail_path = joinpath(runs_dir, base_name * "_hps_failures.csv")
    if isempty(failures)
        isfile(fail_path) && rm(fail_path)
    else
        _write_hps_csv(fail_path, failures)
    end

    if replicated && !isempty(rows)
        summary_rows = _aggregate_repetition_rows(rows)
        _write_hps_csv(joinpath(runs_dir, base_name * "_hps_summary.csv"), summary_rows)
        _write_hps_summary_toml(joinpath(runs_dir, base_name * "_hps_summary.toml"), summary_rows)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

"""
    hpar_search(toml_path; resume = false) -> Vector

Read a TOML file containing `[data]`, `[model]`, `[train]`, and `[hparsearch]`
tables, then perform a hyperparameter search by iterating over all parameter
combinations defined in `[hparsearch.search_space]` and calling
`run_wflow_gnn` for each.

Search-space parameter names use dot-separated paths that mirror the TOML
table structure (e.g. `"model.hidden_dim"`, `"train.lr_start"`). Each entry
is an array of values to try. For `search_type = "box"` every Cartesian-
product combination is evaluated.

For each combination, `[data].run_name` is suffixed with a zero-padded run
index so results are written to separate output folders.

Robustness:
- `resume = true` (CLI: `--resume`; default `false`): a cell whose `metrics/metrics.toml`
  and `output/<split>_{pred,true}.nc` exist, and whose saved `model/*_settings.toml`
  equal the settings of this search, is not retrained; its result row is rebuilt from
  disk. An incomplete (or differently configured) cell is rerun from scratch. Without
  the flag every cell is run and overwritten.
- Optional `[hparsearch] continue_on_error = true` (default): a failing run is logged
  (`@error` with backtrace), listed in `<run_name>_hps_failures.csv` and skipped; the
  search goes on. `InterruptException` always aborts. `false` stops at the first failure.
- `<run_name>_hps_results.csv` (and, for replicated searches, the summary CSV / TOML)
  is rewritten after every run, so partial results survive a crash.
- After each run the GPU model is dropped and GPU memory reclaimed.

Returns a `Vector` of the models trained in this call (on the CPU; resumed and
failed runs contribute none).

# Example TOML

```toml
[hparsearch]
search_type = "box"

[hparsearch.search_space]
"model.hidden_dim" = [32, 64, 128]
"train.lr_start"   = [1e-3, 5e-4]
```
"""
function hpar_search(toml_path::String; resume::Bool = false)
    isfile(toml_path) || throw(ArgumentError("TOML file not found: $toml_path"))
    toml_dir = dirname(abspath(toml_path))
    d_orig   = TOML.parsefile(toml_path)

    haskey(d_orig, "data")       || throw(ArgumentError("TOML missing [data] table"))
    haskey(d_orig, "model")      || throw(ArgumentError("TOML missing [model] table"))
    haskey(d_orig, "train")      || throw(ArgumentError("TOML missing [train] table"))
    haskey(d_orig, "hparsearch") || throw(ArgumentError("TOML missing [hparsearch] table"))

    hps_d = d_orig["hparsearch"]
    seeds = _resolve_replication_seeds(hps_d)
    hps   = HParSearchSettings(
        search_type  = hps_d["search_type"],
        search_space = Dict{String, Vector{Any}}(
            k => collect(Any, v) for (k, v) in hps_d["search_space"]),
        seeds = seeds,
    )

    combos    = hps.search_type == "box" ? _box_combinations(hps.search_space) :
                error("Unsupported search_type: $(hps.search_type)")
    n_combos  = length(combos)
    seed_values = isempty(hps.seeds) ? Union{Nothing,Int}[nothing] : Union{Nothing,Int}[hps.seeds...]
    n_runs    = n_combos * length(seed_values)
    base_name = d_orig["data"]["run_name"]
    models    = []
    rows      = NamedTuple[]
    failures  = NamedTuple[]

    continue_on_error = get(hps_d, "continue_on_error", true)
    continue_on_error isa Bool || throw(ArgumentError("[hparsearch].continue_on_error must be a Bool"))

    @info "HParSearch: $(hps.search_type) search, $n_combos combinations, $(length(seed_values)) repetition(s), $n_runs total runs"

    resolve(p) = isabspath(p) ? p : normpath(joinpath(toml_dir, p))
    runs_dir_resolved = resolve(d_orig["data"]["runs_dir"])

    combo_digits = max(2, ndigits(max(1, n_combos)))
    seed_digits = isempty(hps.seeds) ? 1 : max(2, ndigits(maximum(hps.seeds)))
    run_idx = 0
    for (combo_idx, combo) in enumerate(combos)
        run_base = base_name * "_hps$(lpad(combo_idx, combo_digits, '0'))"
        for seed in seed_values
            run_idx += 1
            @info "Run $run_idx / $n_runs" combo seed

            # Deep-copy raw TOML dict, apply parameter overrides, set unique run name
            d = deepcopy(d_orig)
            for (path, value) in combo
                _set!(d, path, value)
            end
            d["data"]["run_name"] = isnothing(seed) ? run_base : run_base * "_s$(lpad(seed, seed_digits, '0'))"
            if !isnothing(seed)
                d_train = get!(d, "train", Dict{String,Any}())
                d_train["seed"] = seed
            end

            try
                ds, ms, ts = settings_from_config(d, toml_dir)
                run_dir    = joinpath(ds.runs_dir, ds.run_name)

                if resume && _hps_run_is_complete(run_dir, ds, ms, ts)
                    @info "Run $run_idx / $n_runs already finished with identical settings; " *
                          "reusing $(ds.run_name)"
                    metrics = _read_run_core_metrics(run_dir)
                else
                    # never let a stale `metrics.toml` make an interrupted rerun look finished
                    rm(joinpath(run_dir, "metrics", "metrics.toml"); force = true)
                    model, metrics = run_wflow_gnn(ds, ms, ts)
                    push!(models, Flux.cpu(model))
                    model = nothing
                end
                push!(rows, _hps_result_row(ds, ms, ts, run_base, combo_idx, metrics))
            catch e
                e isa InterruptException && rethrow()
                continue_on_error || rethrow()
                run_name = d["data"]["run_name"]
                @error "HParSearch run $run_idx / $n_runs ($run_name) failed; continuing" exception = (e, catch_backtrace())
                push!(failures, (run_name = run_name, run_name_base = run_base,
                                 combo_index = combo_idx,
                                 seed = isnothing(seed) ? -1 : seed,
                                 error = _hps_oneline(e)))
            finally
                _hps_free_memory()
            end
            _hps_write_outputs(runs_dir_resolved, base_name, rows, failures, length(seed_values) > 1)
        end
    end

    results_path  = joinpath(runs_dir_resolved, base_name * "_hps_results.csv")
    failures_path = joinpath(runs_dir_resolved, base_name * "_hps_failures.csv")
    if isempty(failures)
        @info "HParSearch finished: $(length(rows)) / $n_runs runs, results in $results_path"
    else
        @warn "HParSearch finished with $(length(failures)) failed run(s) of $n_runs; " *
              "see $failures_path. Re-run the same command with --resume to retry only " *
              "the failed runs." failed = [f.run_name for f in failures]
    end

    return models
end

# One results-CSV row from the run's settings, the core metrics and the per-run
# `metrics.toml` scalars.
function _hps_result_row(ds, ms, ts, run_base, combo_idx, metrics)
    msc = _read_run_scalar_metrics(joinpath(ds.runs_dir, ds.run_name))
    return (
        run_name                        = ds.run_name,
        run_name_base                   = run_base,
        combo_index                     = combo_idx,
        seed                            = isnothing(ts.seed) ? -1 : ts.seed,
        hidden_dim                      = ms.hidden_dim,
        nlayers                         = ms.nlayers,
        enforce_mass_balance            = ms.enforce_mass_balance,
        mb_formulation                  = String(mb_formulation_kind(ms)),
        batch_size                      = ts.batch_size,
        epochs                          = ts.epochs,
        n_params                        = metrics.n_params,
        max_train_steps                 = maximum(ts.strategy.steps),
        final_train_loss                = metrics.final_train_loss,
        final_val_loss                  = metrics.final_val_loss,
        fixed_val_rmse                  = msc.fixed_val_rmse,
        peak_ratio_frac_gt2             = msc.peak_ratio_frac_gt2,
        pooled_kge                      = msc.pooled_kge,
        river_h_nse                     = msc.river_h_nse,
        rollout_contraction             = msc.rollout_contraction,
        mb_residual_rms                 = msc.mb_residual_rms,
        volume_pbias                    = msc.volume_pbias,
        volume_drift                    = msc.volume_drift,
        h_loss_weight_effective         = msc.h_loss_weight_effective,
        analytic_gain                   = msc.analytic_gain,
        q_to_h_amp                      = msc.q_to_h_amp,
        backbone_rho_median             = msc.backbone_rho_median,
        backbone_rho_max                = msc.backbone_rho_max,
        resid_frac_cap_pos              = msc.resid_frac_cap_pos,
        resid_frac_cap_neg              = msc.resid_frac_cap_neg,
        train_duration_s                = metrics.train_duration_s,
        val_rollout_duration_s          = metrics.val_rollout_duration_s,
        val_rollout_duration_per_step_s = metrics.val_rollout_duration_s /
                                         max(1, metrics.val_n_timesteps),
    )
end
