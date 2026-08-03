import TOML
import Dates
using NCDatasets
using GraphNeuralNetworks
using MLUtils
using Statistics

const DOMAIN_VARS = Dict(
    "river" => Dict(
        "state"   => ["river_q", "river_h"],
        "forcing" => ["river_inwater"],
        "static"  => ["river_length", "river_slope", "river_width", "river_depth", "river_manning_n"],
    ),
    "land" => Dict(
        "state"   => String[],
        "forcing" => String[],
        "static"  => String[],
    ),
    "subsurface" => Dict(
        "state"   => String[],
        "forcing" => String[],
        "static"  => String[],
    ),
)

# LDD numpad direction → (Δlon_idx, Δlat_idx) offset
# The LDD raster is stored as (lon, lat): dim1 = longitude, dim2 = latitude.
# Latitude DECREASES with lat_idx (north-at-top raster convention), so
# moving south means lat_idx+1, north means lat_idx-1.
# Numpad layout:
#   7 8 9
#   4 5 6
#   1 2 3
const LDD_OFFSETS = Dict(
    1 => (-1, +1),  # Southwest: lon-1, lat_idx+1 (lat decreases → south)
    2 => ( 0, +1),  # South:     lon+0, lat_idx+1
    3 => (+1, +1),  # Southeast: lon+1, lat_idx+1
    4 => (-1,  0),  # West:      lon-1, lat_idx+0
    6 => (+1,  0),  # East:      lon+1, lat_idx+0
    7 => (-1, -1),  # Northwest: lon-1, lat_idx-1
    8 => ( 0, -1),  # North:     lon+0, lat_idx-1
    9 => (+1, -1),  # Northeast: lon+1, lat_idx-1
    # 5 = pit/outlet — no outflow, excluded
)

"""
    ldd_to_graph(ncfile::String, domain::String) -> (sources::Vector{Int}, targets::Vector{Int})

Read the `ldd` (local drainage direction) variable from a NetCDF file and return
a directed edge list as two parallel integer vectors.

`domain` must be `"land"` or `"river"`. When `"river"` is specified, only cells
where the `river_mask` variable is non-zero are included.

Each element `i` represents a directed edge from node `sources[i]` to node
`targets[i]`, meaning cell `sources[i]` drains into cell `targets[i]`.

Node indices are column-major linear indices into the 2-D LDD raster (Julia default).
Cells with `missing` values (nodata) and cells with value `5` (pit/outlet) are
excluded — they have no downstream neighbour. Out-of-bounds neighbours (raster
edges) are silently skipped.
"""
function ldd_to_graph(ncfile::String, domain::String, schema::WflowSchema = SCHEMA_V1)
    domain in keys(DOMAIN_VARS) || throw(ArgumentError("domain must be one of $(join(sort(collect(keys(DOMAIN_VARS))), ", ")), got \"$domain\""))

    NCDataset(ncfile, "r") do ds
        ldd_raw = ds[schema.ldd_var][:, :]  # Array{Union{Missing, Int}, 2}  (nrows × ncols)

        # river uses river_mask to define active cells; land and subsurface use all non-nodata cells
        river_mask = domain == "river" ? ds[schema.mask_var][:, :] : nothing

        is_active(r, c) = domain == "river" ?
            (!ismissing(river_mask[r, c]) && river_mask[r, c] != 0) :
            !ismissing(ldd_raw[r, c])

        nrows, ncols = size(ldd_raw)

        sources = Int[]
        targets = Int[]

        for col in 1:ncols, row in 1:nrows
            val = ldd_raw[row, col]
            ismissing(val) && continue
            val == 5 && continue  # pit / outlet

            is_active(row, col) || continue

            offset = get(LDD_OFFSETS, Int(val), nothing)
            isnothing(offset) && continue  # unknown value, skip

            Δrow, Δcol = offset
            nrow = row + Δrow
            ncol = col + Δcol

            # Skip if neighbour falls outside the raster
            (nrow < 1 || nrow > nrows || ncol < 1 || ncol > ncols) && continue

            # Skip if target cell is inactive
            is_active(nrow, ncol) || continue

            # Column-major linear index: (col - 1) * nrows + row
            src = (col  - 1) * nrows + row
            tgt = (ncol - 1) * nrows + nrow

            push!(sources, src)
            push!(targets, tgt)
        end

        return sources, targets
    end
end

"""
    check_and_correct_grid_alignment(staticmaps_file, output_file, domain)
        -> (dim1_flip::Bool, dim2_flip::Bool)

Verify that the spatial grids in `staticmaps_file` and `output_file` cover
the same geographic domain, and detect whether either spatial axis is
reversed between the two files.

The check is performed on the `local_drain_direction` variable in
`staticmaps_file` and the first state (or forcing) variable for `domain` in
`output_file`.

Throws `ArgumentError` if:
- The number of spatial dimensions differs.
- Any spatial axis has a different number of coordinate values.
- Any spatial axis does not span the same coordinate range in both files
  (compared after sorting, so direction is ignored for the extent check).

Returns a `NamedTuple` `(dim1_flip, dim2_flip)` where a `true` flag means
the corresponding axis in `output_file` runs in the **opposite** direction
to the same axis in `staticmaps_file`. `build_wflow_graph` uses these flags
to remap node indices when reading from `output_file`.
"""
function check_and_correct_grid_alignment(
        staticmaps_file :: String,
        output_file     :: String,
        domain          :: String,
        schema          :: WflowSchema = SCHEMA_V1)

    # Find the first output-sourced variable to use as the spatial grid reference
    all_out = filter(p -> p.second.source == :output,
                     vcat(schema.state_vars, schema.forcing_vars, schema.static_vars))
    isempty(all_out) && throw(ArgumentError(
        "schema has no output-sourced variables; cannot check grid alignment"))
    ref_vname = first(all_out).second.ncdf_name

    sm_coords, out_coords = NCDataset(staticmaps_file, "r") do sm_ds
        NCDataset(output_file, "r") do out_ds
            sm_sdims  = [d for d in dimnames(sm_ds[schema.ldd_var])
                         if d ∉ ("time", "layer")]
            out_sdims = [d for d in dimnames(out_ds[ref_vname])
                         if d ∉ ("time", "layer")]

            length(sm_sdims) == length(out_sdims) ||
                throw(ArgumentError(
                    "$(basename(staticmaps_file)) has $(length(sm_sdims)) spatial " *
                    "dimension(s) but $(basename(output_file)) has $(length(out_sdims))"))

            sc = [Float64.(sm_ds[d][:])  for d in sm_sdims]
            oc = [Float64.(out_ds[d][:]) for d in out_sdims]
            sc, oc
        end
    end

    flips = Bool[]
    for i in eachindex(sm_coords)
        sc = sm_coords[i]
        oc = out_coords[i]

        length(sc) == length(oc) ||
            throw(ArgumentError(
                "spatial axis $i length mismatch: " *
                "$(basename(staticmaps_file)) has $(length(sc)) values, " *
                "$(basename(output_file)) has $(length(oc))"))

        isapprox(sort(sc), sort(oc); rtol = 1e-5) ||
            throw(ArgumentError(
                "spatial axis $i does not cover the same coordinate range: " *
                "$(basename(staticmaps_file)) spans [$(minimum(sc)), $(maximum(sc))], " *
                "$(basename(output_file)) spans [$(minimum(oc)), $(maximum(oc))]"))

        # Reversed when first value of one ≈ last value of the other
        push!(flips, isapprox(sc[1], oc[end]; rtol = 1e-5) &&
                     isapprox(sc[end], oc[1]; rtol = 1e-5))
    end

    return (dim1_flip = flips[1], dim2_flip = flips[2])
end

# ── Per-variable preprocessing scalers ───────────────────────────────────────
# Each function scales a `(n_nodes × ntimes)` Float32 slice in-place and
# returns a `Vector{Float32}` of per-node multipliers that **invert** the
# applied transform (multiply z-score-denormalised values by the returned
# vector to recover physical units).
#
# Signature: f!(slice, rows, cols, staticmaps_file) -> inv_scales

function scale_river_q!(slice           :: AbstractMatrix{Float32},
                        rows            :: Vector{Int},
                        cols            :: Vector{Int},
                        staticmaps_file :: String,
                        schema          :: WflowSchema = SCHEMA_V1)
    NCDataset(staticmaps_file, "r") do ds
        coerce(x) = ismissing(x) ? NaN32 : Float32(x)
        area = ds[schema.upstream_area_var][:, :]
        inv  = Vector{Float32}(undef, length(rows))
        for i in eachindex(rows)
            a = coerce(area[rows[i], cols[i]])
            if !isnan(a) && a > 0f0
                slice[i, :] ./= a
                inv[i] = a          # postscale: multiply by area to recover m³/s
            else
                inv[i] = NaN32
            end
        end
        inv
    end
end

function scale_river_h!(slice           :: AbstractMatrix{Float32},
                        rows            :: Vector{Int},
                        cols            :: Vector{Int},
                        staticmaps_file :: String,
                        schema          :: WflowSchema = SCHEMA_V1)
    NCDataset(staticmaps_file, "r") do ds
        coerce(x) = ismissing(x) ? NaN32 : Float32(x)
        area  = ds[schema.upstream_area_var][:, :]
        width = ds[schema.river_width_var][:, :]
        len   = ds[schema.river_length_var][:, :]
        inv   = Vector{Float32}(undef, length(rows))
        for i in eachindex(rows)
            r, c = rows[i], cols[i]
            w = coerce(width[r, c])
            l = coerce(len[r, c])
            a = coerce(area[r, c])
            if !isnan(w) && !isnan(l) && !isnan(a) && a > 0f0
                slice[i, :] .*= (w * l / a)
                inv[i] = a / (w * l)  # postscale: multiply by a/(w*l) to recover m
            else
                inv[i] = NaN32
            end
        end
        inv
    end
end

"""
Dict mapping `(domain, varname)` to a preprocessing scaler function.

Each entry has signature:
    f!(slice::AbstractMatrix{Float32}, rows, cols, staticmaps_file) -> Vector{Float32}
where `slice` is `(n_nodes × ntimes)` and the returned vector is the per-node
inverse-scale multiplier (postscale) needed to recover physical units after
z-score denormalisation.
"""
const VAR_SCALERS = Dict(
    "river" => Dict(
        "river_q" => scale_river_q!,
        "river_h" => scale_river_h!,
    ),
    "land"        => Dict{String, Function}(),
    "subsurface"  => Dict{String, Function}(),
)

"""
    build_wflow_graph(staticmaps_file, output_file, domain)
        -> (graphs, stats, grid, postscale, static)

Construct a `GNNGraph` for the wflow routing domain with standardized node features.

- Topology is derived from the `local_drain_direction` variable in `staticmaps_file`
  via `ldd_to_graph`.
- Node indices are compacted to a contiguous 1-based range.
- Variable names for each feature group are taken from `DOMAIN_VARS[domain]`.
- Node features extracted from `output_file` (all simulated timesteps):
  - `ndata.state`:   `Float32` array of shape `(n_state,   n_nodes)` per graph
  - `ndata.forcing`: `Float32` array of shape `(n_forcing, n_nodes)` per graph
- Static node features extracted from `staticmaps_file`:
  - `static`:  `Float32` array of shape `(n_static, n_nodes)`, returned as the
    fifth value (shared across all timesteps, not stored in GNNGraph ndata).
- All features are z-score standardized per variable (mean 0, std 1).
- `missing` values are coerced to `NaN32` before standardization.
- The second return value maps each variable name to its `(mean, std)` named tuple,
  for use when standardizing new data or inverting the transform.
- The third return value is a `NamedTuple` `(rows, cols, nrows, ncols)` giving the
  1-based row/column position of each compacted node and the full raster dimensions.
  Pass this directly to `ungrid` to scatter rollout output back to the grid.
- The fourth return value is a `Dict{String, Vector{Float32}}` mapping each
  preprocessed state variable to a per-node multiplier that inverts the
  variable-specific transform applied before z-score normalisation (e.g. area
  normalisation for `river_q`, storage-proxy scaling for `river_h`).
  Pass this to `evaluate_trajectory` as `postscale` to recover physical units.
- The fifth return value is the shared `static` feature matrix.
"""
function build_wflow_graph(staticmaps_file::String, output_file::String, domain::String;
                           schema::WflowSchema = SCHEMA_V1)
    # ── 0. Check grid alignment; detect reversed axes ────────────────────────
    alignment  = check_and_correct_grid_alignment(staticmaps_file, output_file, domain, schema)
    dim1_flip  = alignment.dim1_flip
    dim2_flip  = alignment.dim2_flip

    # ── 1. Get sparse linear-index edge list ────────────────────────────────
    src_raw, tgt_raw = ldd_to_graph(staticmaps_file, domain, schema)

    # ── 2. Build compact node index mapping (all active domain cells) ────────
    # Derive node_ids from the mask directly so that isolated/outlet cells
    # (e.g. LDD pit cells with value 5) are still included as graph nodes.
    node_ids = NCDataset(staticmaps_file, "r") do ds
        if domain == "river"
            mask = ds[schema.mask_var][:, :]
            nr, nc = size(mask)
            sort([((c - 1) * nr + r)
                  for c in 1:nc, r in 1:nr
                  if !ismissing(mask[r, c]) && mask[r, c] != 0])
        else
            ldd = ds[schema.ldd_var][:, :]
            nr, nc = size(ldd)
            sort([((c - 1) * nr + r)
                  for c in 1:nc, r in 1:nr
                  if !ismissing(ldd[r, c])])
        end
    end
    n_nodes   = length(node_ids)
    id_to_idx = Dict(id => i for (i, id) in enumerate(node_ids))

    sources = [id_to_idx[s] for s in src_raw]
    targets = [id_to_idx[t] for t in tgt_raw]

    # ── 3. Recover (row, col) for each node ─────────────────────────────────
    nrows, ncols = NCDataset(staticmaps_file, "r") do ds
        size(ds[schema.ldd_var])
    end

    rows = [(id - 1) % nrows + 1 for id in node_ids]
    cols = [(id - 1) ÷ nrows + 1 for id in node_ids]

    coerce(x) = ismissing(x) ? NaN32 : Float32(x)

    # Standardize a feature slice in-place, ignoring NaNs.
    # Returns (mean, std) for the stats dict.
    function standardize!(slice::AbstractArray{Float32})
        vals = filter(!isnan, vec(slice))
        μ    = isempty(vals) ? 0f0 : mean(vals)
        σ    = (isempty(vals) || length(vals) == 1) ? 1f0 : std(vals)
        σ    = σ == 0f0 ? 1f0 : σ   # guard against constant variables
        slice .= (slice .- μ) ./ σ
        μ, σ
    end

    stats     = Dict{String, @NamedTuple{mean::Float32, std::Float32}}()
    postscale = Dict{String, Vector{Float32}}()

    # ── 4 & 5. Extract all features, routing each variable to its source file ──
    # Both files are opened once; each VarSpec carries the ncdf_name and source.
    state, forcing, static = NCDataset(staticmaps_file, "r") do sm_ds
        NCDataset(output_file, "r") do out_ds

            # ntimes from the first output-sourced state/forcing variable
            all_tv  = vcat(schema.state_vars, schema.forcing_vars)
            ref_idx = findfirst(p -> p.second.source == :output, all_tv)
            isnothing(ref_idx) && throw(ArgumentError(
                "schema has no output-sourced state/forcing variables"))
            ntimes = size(out_ds[all_tv[ref_idx].second.ncdf_name], 3)

            # Extract time-varying variables; VarSpec carries ncdf_name and source.
            function extract_timeseries(var_pairs)
                n   = length(var_pairs)
                arr = Array{Float32}(undef, n, n_nodes, ntimes)
                for (vi, (lname, spec)) in enumerate(var_pairs)
                    if spec.source == :output
                        data = out_ds[spec.ncdf_name][:, :, :]
                        for (i, (r, c)) in enumerate(zip(rows, cols))
                            r_out = dim1_flip ? nrows - r + 1 : r
                            c_out = dim2_flip ? ncols - c + 1 : c
                            for t in 1:ntimes
                                arr[vi, i, t] = coerce(data[r_out, c_out, t])
                            end
                        end
                    else  # :staticmaps — time-invariant; broadcast across timesteps
                        data = sm_ds[spec.ncdf_name][:, :]
                        for (i, (r, c)) in enumerate(zip(rows, cols))
                            val = coerce(data[r, c])
                            for t in 1:ntimes
                                arr[vi, i, t] = val
                            end
                        end
                    end
                    # Apply domain/variable-specific preprocessing via VAR_SCALERS
                    domain_scalers = get(VAR_SCALERS, domain, Dict{String,Function}())
                    if haskey(domain_scalers, lname)
                        postscale[lname] = domain_scalers[lname](
                            @view(arr[vi, :, :]), rows, cols, staticmaps_file, schema)
                    end
                    μ, σ = standardize!(@view arr[vi, :, :])
                    stats[lname] = (mean = μ, std = σ)
                end
                arr
            end

            # Extract static variables; source-routed per variable.
            function extract_static(var_pairs)
                n   = length(var_pairs)
                arr = Array{Float32}(undef, n, n_nodes)
                for (vi, (lname, spec)) in enumerate(var_pairs)
                    if spec.source == :staticmaps
                        data = sm_ds[spec.ncdf_name][:, :]
                        for (i, (r, c)) in enumerate(zip(rows, cols))
                            arr[vi, i] = coerce(data[r, c])
                        end
                    else  # :output — apply flip correction; take first timestep (static in time)
                        data = out_ds[spec.ncdf_name][:, :, 1]
                        for (i, (r, c)) in enumerate(zip(rows, cols))
                            r_out = dim1_flip ? nrows - r + 1 : r
                            c_out = dim2_flip ? ncols - c + 1 : c
                            arr[vi, i] = coerce(data[r_out, c_out])
                        end
                    end
                    μ, σ = standardize!(@view arr[vi, :])
                    stats[lname] = (mean = μ, std = σ)
                end
                arr
            end

            extract_timeseries(schema.state_vars),
            extract_timeseries(schema.forcing_vars),
            extract_static(schema.static_vars)
        end
    end

    # ── 6. Construct one GNNGraph per timestep ───────────────────────────────
    ntimes = size(state, 3)
    graphs = [
        GNNGraph(sources, targets;
                 num_nodes = n_nodes,
                 ndata = (state   = state[:, :, t],
                          forcing = forcing[:, :, t]))
        for t in 1:ntimes
    ]
    grid = (rows  = rows,
            cols  = cols,
            nrows = nrows,
            ncols = ncols)
    return graphs, stats, grid, postscale, static
end

"""
    make_horizon_dataset(graphs, nhorizon; at) -> (train, val, test)

Slide a window of `nhorizon` consecutive `GNNGraph`s over `graphs` to produce
a `Vector{Vector{GNNGraph}}`, then partition it into train / validation / test
subsets using `MLUtils.splitobs`.

- `graphs`   — `Vector{GNNGraph}` as returned by `build_wflow_graph`.
- `nhorizon` — number of consecutive timesteps per sample window.
- `at`       — passed directly to `MLUtils.splitobs` as its `at` keyword.
               Use a 2-tuple of per-split fractions for three splits,
               e.g. `at = (0.6, 0.2)` gives 60 % train, 20 % validation,
               20 % test (remainder).

Returns a named tuple `(train, val, test)`.
"""
function make_horizon_dataset(
        graphs  ::Vector{<:GNNGraph},
        nhorizon::Int;
        at)

    length(graphs) >= nhorizon ||
        throw(ArgumentError("nhorizon ($nhorizon) exceeds the number of timesteps ($(length(graphs)))"))

    nwindows = length(graphs) - nhorizon + 1
    windows  = [graphs[t:t+nhorizon-1] for t in 1:nwindows]

    train, val, test = splitobs(windows; at)
    return (train = train, val = val, test = test)
end

# Collate a vector of windows (each a Vector{GNNGraph}) into a single
# Vector{GNNGraph} where each element is a batched GNNGraph.
MLUtils.batch(windows::AbstractVector{<:Vector{<:GNNGraph}}) =
    [GNNGraphs.batch([w[t] for w in windows]) for t in 1:length(windows[1])]

"""
    get_timestep(output_file) -> Float32

Return the model timestep in seconds by reading two consecutive timestamps
from the `time` variable in `output_file` (a CF-convention NetCDF file).
"""
function get_timestep(output_file::String)
    NCDataset(output_file, "r") do ds
        times = ds["time"][:]
        length(times) >= 2 ||
            throw(ArgumentError("output file must contain at least 2 timesteps"))
        # DateTime subtraction returns Dates.Millisecond on all Julia versions
        Float32(Dates.value(times[2] - times[1]) / 1000)
    end
end

# ---------------------------------------------------------------------------
# DataSettings
# ---------------------------------------------------------------------------

"""
    DataSettings

Configuration for data generation and splits.

Fields:
- `run_name`         : name of this run (used for output folder naming).
- `runs_dir`         : root directory where run folders are saved.
- `wflow_model_path` : path to the wflow model directory.
- `train_frac`       : fraction of windows used for training.
- `val_frac`         : fraction of windows used for validation.
- `output_run_dir`   : subdirectory inside `wflow_model_path` containing `output.nc`
                       (default `"run_default"`).
- `wflow_schema`     : schema preset name (e.g. `"v1"`) or path to a schema TOML file
                       that maps logical variable names to NetCDF names and source files
                       (default `"v1"`).
"""
struct DataSettings
    run_name         :: String
    runs_dir         :: String
    wflow_model_path :: String
    train_frac       :: Float64
    val_frac         :: Float64
    output_run_dir   :: String
    wflow_schema     :: String
end

"""
    DataSettings(; run_name, runs_dir, wflow_model_path, train_frac, val_frac,
                   output_run_dir = "run_default", wflow_schema = "v1") -> DataSettings
"""
function DataSettings(;
        run_name         :: String,
        runs_dir         :: String,
        wflow_model_path :: String,
        train_frac       :: Real,
        val_frac         :: Real,
        output_run_dir   :: String = "run_default",
        wflow_schema     :: String = "v1")

    isempty(run_name)       && throw(ArgumentError("run_name must not be empty"))
    isempty(output_run_dir) && throw(ArgumentError("output_run_dir must not be empty"))
    train_frac > 0      || throw(ArgumentError("train_frac must be positive"))
    val_frac   > 0      || throw(ArgumentError("val_frac must be positive"))
    train_frac + val_frac < 1 ||
        throw(ArgumentError("train_frac + val_frac must be < 1"))

    DataSettings(run_name, runs_dir, wflow_model_path,
                 Float64(train_frac), Float64(val_frac),
                 output_run_dir, wflow_schema)
end

function Base.show(io::IO, s::DataSettings)
    println(io, "DataSettings:")
    println(io, "  run_name         : ", s.run_name)
    println(io, "  runs_dir         : ", s.runs_dir)
    println(io, "  wflow_model_path : ", s.wflow_model_path)
    println(io, "  train_frac       : ", s.train_frac)
    println(io, "  val_frac         : ", s.val_frac)
    println(io, "  output_run_dir   : ", s.output_run_dir)
    print(  io, "  wflow_schema     : ", s.wflow_schema)
end

"""
    save_data_settings(path, settings)

Write `settings` to a TOML file at `path`.
"""
function save_data_settings(path::String, s::DataSettings)
    dict = Dict(
        "run_name"         => s.run_name,
        "runs_dir"         => s.runs_dir,
        "wflow_model_path" => s.wflow_model_path,
        "train_frac"       => s.train_frac,
        "val_frac"         => s.val_frac,
        "output_run_dir"   => s.output_run_dir,
        "wflow_schema"     => s.wflow_schema,
    )
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_data_settings(path) -> DataSettings

Read a `DataSettings` from the TOML file at `path`.
"""
function load_data_settings(path::String)
    d = TOML.parsefile(path)
    return DataSettings(
        run_name         = d["run_name"],
        runs_dir         = d["runs_dir"],
        wflow_model_path = d["wflow_model_path"],
        train_frac       = d["train_frac"],
        val_frac         = d["val_frac"],
        output_run_dir   = get(d, "output_run_dir", "run_default"),
        wflow_schema     = get(d, "wflow_schema",   "v1"),
    )
end
