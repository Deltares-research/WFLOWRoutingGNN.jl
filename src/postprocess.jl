"""
    regrid(states, grid, domain) -> Dict{String, Array{Float32,3}}

Convert the `(n_state, n_nodes, T)` rollout output back to the full 2-D grid
format used by wflow (nrows × ncols × T), with `NaN32` at non-active cells.

`grid` is the `NamedTuple` returned as the third value of `build_wflow_graph`.
It holds the pre-computed `(row, col)` position of every compacted node and the
full raster dimensions — so no file I/O is needed here.

Arguments:
- `states` : `Array{Float32,3}` of shape `(n_state, n_nodes, T)` as returned by
             `rollout` or `evaluate_trajectory`.
- `grid`   : `NamedTuple` with fields `rows`, `cols`, `nrows`, `ncols` as
             returned by `build_wflow_graph`.
- `domain` : routing domain string (key of `DOMAIN_VARS`), used to look up
             variable names for the output dictionary.

Returns a `Dict{String, Array{Float32,3}}` mapping each state variable name to
an array of shape `(nrows, ncols, T)` filled with `NaN32` everywhere except at
active domain nodes.
"""
function regrid(states ::AbstractArray{Float32,3},
                grid   ::NamedTuple,
                domain ::String)

    state_vars = DOMAIN_VARS[domain]["state"]
    n_state    = size(states, 1)
    n_nodes    = size(states, 2)
    T          = size(states, 3)

    n_state == length(state_vars) ||
        throw(ArgumentError("states has $(n_state) state dimensions but domain " *
                            "\"$domain\" has $(length(state_vars)) state variables"))

    rows  = grid.rows
    cols  = grid.cols
    nrows = grid.nrows
    ncols = grid.ncols

    length(rows) == n_nodes ||
        throw(ArgumentError("grid has $(length(rows)) nodes but states n_nodes is $n_nodes"))

    # Scatter each variable back onto the full grid
    result = Dict{String, Array{Float32,3}}()
    for (vi, vname) in enumerate(state_vars)
        raster = fill(NaN32, nrows, ncols, T)
        for (ni, (r, c)) in enumerate(zip(rows, cols))
            raster[r, c, :] = states[vi, ni, :]
        end
        result[vname] = raster
    end

    return result
end

"""
    write_regrid_to_netcdf(grids, staticmaps_file, timestamps, path)

Write the gridded output from `regrid` to a NetCDF file with `lon`, `lat`, and
`time` as dimensions.

Spatial coordinates are read from `staticmaps_file`.  `timestamps` must be a
`Vector{DateTime}`; NCDatasets encodes it automatically using CF conventions.
`NaN32` values (inactive cells) are written as `_FillValue`.

Arguments:
- `grids`           : `Dict{String, Array{Float32,3}}` as returned by `regrid`.
- `staticmaps_file` : path to the wflow `staticmaps.nc` used during training.
- `timestamps`      : `Vector{DateTime}` of length `T`.
- `path`            : output file path (`.nc`).
"""
function write_regrid_to_netcdf(
        grids           :: Dict{String, Array{Float32,3}},
        staticmaps_file :: String,
        timestamps      :: AbstractVector,
        path            :: String)

    isempty(grids) && throw(ArgumentError("grids must not be empty"))

    # Read lon/lat coordinate vectors from staticmaps (dim1=lon, dim2=lat)
    lon_vals, lat_vals = NCDataset(staticmaps_file, "r") do ds
        dn = dimnames(ds["local_drain_direction"])
        Float64.(ds[dn[1]][:]), Float64.(ds[dn[2]][:])
    end

    T = length(timestamps)

    NCDataset(path, "c") do ds
        defDim(ds, "lon",  length(lon_vals))
        defDim(ds, "lat",  length(lat_vals))
        defDim(ds, "time", T)

        defVar(ds, "lon", lon_vals, ("lon",);
               attrib = ["units"         => "degrees_east",
                         "long_name"     => "longitude",
                         "standard_name" => "longitude"])

        defVar(ds, "lat", lat_vals, ("lat",);
               attrib = ["units"         => "degrees_north",
                         "long_name"     => "latitude",
                         "standard_name" => "latitude"])

        # Pass the DateTime array directly as the third arg so NCDatasets
        # infers the element type and handles CF encoding automatically
        defVar(ds, "time", timestamps, ("time",))

        fill_val = NaN32
        for (vname, raster) in grids
            v = defVar(ds, vname, Float32, ("lon", "lat", "time");
                       fillvalue = fill_val,
                       attrib    = ["long_name" => vname])
            for t in 1:T
                v[:, :, t] = replace(raster[:, :, t], NaN32 => missing)
            end
        end
    end
end
