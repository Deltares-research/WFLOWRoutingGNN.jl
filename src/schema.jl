import TOML

"""
    VarSpec

Descriptor for one logical variable: its NetCDF name and the source file it
lives in.

Fields:
- `ncdf_name` : variable name in the NetCDF file.
- `source`    : `:staticmaps` (read from `staticmaps.nc`) or
                `:output`     (read from `output.nc`).
"""
struct VarSpec
    ncdf_name :: String
    source    :: Symbol   # :staticmaps | :output
end

"""
    WflowSchema

Maps the logical variable names used by the GNN to the actual NetCDF variable
names and source files for a specific wflow output version.

The logical names (the keys of each variable list) must match the corresponding
entries in `DOMAIN_VARS` for the target domain; only the NetCDF names and
source locations may differ between wflow versions.

Fields:
- `state_vars`        : ordered `(logical_name => VarSpec)` pairs for state variables.
- `forcing_vars`      : ordered `(logical_name => VarSpec)` pairs for forcing variables.
- `static_vars`       : ordered `(logical_name => VarSpec)` pairs for static variables.
- `ldd_var`           : NetCDF name of the local drain direction raster in `staticmaps.nc`.
- `mask_var`          : NetCDF name of the river mask raster in `staticmaps.nc`.
- `upstream_area_var` : NetCDF name of upstream catchment area in `staticmaps.nc`
                        (used by the `river_q` and `river_h` preprocessing scalers).
- `river_width_var`   : NetCDF name of river width in `staticmaps.nc` (river_h scaler).
- `river_length_var`  : NetCDF name of river length in `staticmaps.nc` (river_h scaler).
"""
struct WflowSchema
    state_vars        :: Vector{Pair{String, VarSpec}}
    forcing_vars      :: Vector{Pair{String, VarSpec}}
    static_vars       :: Vector{Pair{String, VarSpec}}
    ldd_var           :: String
    mask_var          :: String
    upstream_area_var :: String
    river_width_var   :: String
    river_length_var  :: String
end

# ── Convenience helpers ──────────────────────────────────────────────────────

"""Return the ordered list of logical variable names for a variable group."""
logical_names(vars::Vector{Pair{String, VarSpec}}) = first.(vars)

"""Return the ordered list of NetCDF variable names for a variable group."""
ncdf_names(vars::Vector{Pair{String, VarSpec}}) = map(p -> p.second.ncdf_name, vars)

# ── Predefined schemas ───────────────────────────────────────────────────────

"""
    SCHEMA_V1

Default schema matching the current wflow output format (wflow_sbm v0.x).
All state and forcing variables are in `output.nc`; all static variables are
in `staticmaps.nc`.
"""
const SCHEMA_V1 = WflowSchema(
    # state_vars (logical => VarSpec)
    ["river_q" => VarSpec("river_q", :output),
     "river_h" => VarSpec("river_h", :output)],
    # forcing_vars
    ["river_inwater" => VarSpec("river_inwater", :output)],
    # static_vars
    ["river_length"    => VarSpec("river_length",    :staticmaps),
     "river_slope"     => VarSpec("river_slope",     :staticmaps),
     "river_width"     => VarSpec("river_width",     :staticmaps),
     "river_depth"     => VarSpec("river_depth",     :staticmaps),
     "river_manning_n" => VarSpec("river_manning_n", :staticmaps)],
    # infrastructure variable names in staticmaps.nc
    "local_drain_direction",   # ldd_var
    "river_mask",              # mask_var
    "meta_upstream_area",      # upstream_area_var (river_q / river_h scalers)
    "river_width",             # river_width_var   (river_h scaler)
    "river_length",            # river_length_var  (river_h scaler)
)

const SCHEMA_V081 = WflowSchema(
    ["river_q" => VarSpec("river_q", :output),
     "river_h" => VarSpec("river_h", :output)],
    ["river_inwater" => VarSpec("river_inwater", :output)],
    ["river_length"    => VarSpec("wflow_riverlength",    :staticmaps),
     "river_slope"     => VarSpec("RiverSlope",     :staticmaps),
     "river_width"     => VarSpec("wflow_riverwidth",     :staticmaps),
     "river_depth"     => VarSpec("river_depth",     :output),
     "river_manning_n" => VarSpec("river_manning_n", :output)],
    "wflow_ldd",   # ldd_var
    "wflow_river",              # mask_var
    "wflow_uparea",      # upstream_area_var (river_q / river_h scalers)
    "wflow_riverwidth",             # river_width_var   (river_h scaler)
    "wflow_riverlength",            # river_length_var  (river_h scaler)
)

"""
Registry of named schemas.  Add entries here to register new wflow versions.
Keys are the strings accepted by `load_schema` and by the `wflow_schema` field
of `DataSettings`.
"""
const SCHEMAS = Dict{String, WflowSchema}(
    "v1" => SCHEMA_V1,
    "v0.8.1" => SCHEMA_V081
    )

# ── TOML serialization ───────────────────────────────────────────────────────

function _varlist_to_dicts(vars::Vector{Pair{String, VarSpec}})
    [Dict("logical"   => name,
          "ncdf_name" => spec.ncdf_name,
          "source"    => String(spec.source))
     for (name, spec) in vars]
end

function _dicts_to_varlist(dicts::Vector)
    [d["logical"] => VarSpec(d["ncdf_name"], Symbol(d["source"])) for d in dicts]
end

"""
    save_schema(path, schema)

Serialise `schema` to a TOML file at `path`.  The variable lists are stored as
TOML arrays of inline tables so that insertion order is preserved on reload.
"""
function save_schema(path::String, s::WflowSchema)
    dict = Dict(
        "ldd_var"           => s.ldd_var,
        "mask_var"          => s.mask_var,
        "upstream_area_var" => s.upstream_area_var,
        "river_width_var"   => s.river_width_var,
        "river_length_var"  => s.river_length_var,
        "state_vars"        => _varlist_to_dicts(s.state_vars),
        "forcing_vars"      => _varlist_to_dicts(s.forcing_vars),
        "static_vars"       => _varlist_to_dicts(s.static_vars),
    )
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_schema(name_or_path) -> WflowSchema

Return a `WflowSchema` by preset name (e.g. `"v1"`) or by reading a TOML file
at the given absolute or relative path.

Preset names currently available: $(join(sort(collect(keys(SCHEMAS))), ", ")).
"""
function load_schema(name_or_path::String)
    haskey(SCHEMAS, name_or_path) && return SCHEMAS[name_or_path]
    isfile(name_or_path) || throw(ArgumentError(
        "\"$name_or_path\" is neither a known schema preset " *
        "($(join(sort(collect(keys(SCHEMAS))), ", "))) " *
        "nor a path to an existing file"))
    d = TOML.parsefile(name_or_path)
    WflowSchema(
        _dicts_to_varlist(d["state_vars"]),
        _dicts_to_varlist(d["forcing_vars"]),
        _dicts_to_varlist(d["static_vars"]),
        d["ldd_var"],
        d["mask_var"],
        d["upstream_area_var"],
        d["river_width_var"],
        d["river_length_var"],
    )
end
