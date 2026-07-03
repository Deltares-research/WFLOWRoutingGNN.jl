import TOML

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
end

const SUPPORTED_SEARCH_TYPES = ("box",)

"""
    HParSearchSettings(; search_type, search_space) -> HParSearchSettings
"""
function HParSearchSettings(;
        search_type  :: String,
        search_space :: Dict)

    search_type in SUPPORTED_SEARCH_TYPES ||
        throw(ArgumentError("search_type must be one of " *
                            join(SUPPORTED_SEARCH_TYPES, ", ") *
                            "; got \"$search_type\""))
    isempty(search_space) &&
        throw(ArgumentError("search_space must not be empty"))

    HParSearchSettings(
        search_type,
        Dict{String, Vector{Any}}(k => collect(Any, v) for (k, v) in search_space),
    )
end

function Base.show(io::IO, s::HParSearchSettings)
    println(io, "HParSearchSettings:")
    println(io, "  search_type : ", s.search_type)
    println(io, "  search_space:")
    for (k, v) in sort(collect(s.search_space); by = first)
        println(io, "    ", k, " : ", v)
    end
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
        node = node[p]
    end
    node[parts[end]] = value
end

# Write a Vector of NamedTuples as a CSV file.
function _write_hps_csv(path::String, rows::Vector)
    isempty(rows) && return
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ","))
        for row in rows
            println(io, join(string.(values(row)), ","))
        end
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

# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

"""
    hpar_search(toml_path) -> Vector

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

Returns a `Vector` of trained models (one per combination).

# Example TOML

```toml
[hparsearch]
search_type = "box"

[hparsearch.search_space]
"model.hidden_dim" = [32, 64, 128]
"train.lr_start"   = [1e-3, 5e-4]
```
"""
function hpar_search(toml_path::String)
    isfile(toml_path) || throw(ArgumentError("TOML file not found: $toml_path"))
    toml_dir = dirname(abspath(toml_path))
    d_orig   = TOML.parsefile(toml_path)

    haskey(d_orig, "data")       || throw(ArgumentError("TOML missing [data] table"))
    haskey(d_orig, "model")      || throw(ArgumentError("TOML missing [model] table"))
    haskey(d_orig, "train")      || throw(ArgumentError("TOML missing [train] table"))
    haskey(d_orig, "hparsearch") || throw(ArgumentError("TOML missing [hparsearch] table"))

    hps_d = d_orig["hparsearch"]
    hps   = HParSearchSettings(
        search_type  = hps_d["search_type"],
        search_space = Dict{String, Vector{Any}}(
            k => collect(Any, v) for (k, v) in hps_d["search_space"]),
    )

    combos    = hps.search_type == "box" ? _box_combinations(hps.search_space) :
                error("Unsupported search_type: $(hps.search_type)")
    n_combos  = length(combos)
    base_name = d_orig["data"]["run_name"]
    models    = []
    rows      = NamedTuple[]

    @info "HParSearch: $(hps.search_type) search, $n_combos combinations"

    resolve(p) = isabspath(p) ? p : normpath(joinpath(toml_dir, p))
    runs_dir_resolved = resolve(d_orig["data"]["runs_dir"])

    for (idx, combo) in enumerate(combos)
        @info "Run $idx / $n_combos" combo

        # Deep-copy raw TOML dict, apply parameter overrides, set unique run name
        d = deepcopy(d_orig)
        for (path, value) in combo
            _set!(d, path, value)
        end
        d["data"]["run_name"] = base_name * "_hps$(lpad(idx, ndigits(n_combos), '0'))"

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

        model, metrics = run_wflow_gnn(ds, ms, ts)
        push!(models, model)
        push!(rows, (
            run_name                        = ds.run_name,
            hidden_dim                      = ms.hidden_dim,
            nlayers                         = ms.nlayers,
            batch_size                      = ts.batch_size,
            epochs                          = ts.epochs,
            n_params                        = metrics.n_params,
            max_train_steps                 = maximum(ts.strategy.steps),
            final_train_loss                = metrics.final_train_loss,
            final_val_loss                  = metrics.final_val_loss,
            train_duration_s                = metrics.train_duration_s,
            val_rollout_duration_s          = metrics.val_rollout_duration_s,
            val_rollout_duration_per_step_s = metrics.val_rollout_duration_s /
                                             max(1, metrics.val_n_timesteps),
        ))
    end

    csv_path = joinpath(runs_dir_resolved, base_name * "_hps_results.csv")
    _write_hps_csv(csv_path, rows)
    @info "HParSearch results written to $csv_path"

    return models
end
