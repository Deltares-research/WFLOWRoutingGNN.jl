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
        path            :: String;
        schema          :: WflowSchema = SCHEMA_V1)

    isempty(grids) && throw(ArgumentError("grids must not be empty"))

    # Read lon/lat coordinate vectors from staticmaps (dim1=lon, dim2=lat)
    lon_vals, lat_vals = NCDataset(staticmaps_file, "r") do ds
        dn = dimnames(ds[schema.ldd_var])
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

# List of per-cell metric names produced by `spatial_error_metrics`, in a
# stable order (used for NetCDF variable naming and plot panel layout).
const SPATIAL_METRIC_NAMES = ["rmse", "bias", "relbias", "overpred_freq",
                              "peak_err", "peak_lag", "nbias", "nse"]

# Cross-correlation timing lag (in timesteps) between a predicted and true
# series. Returns the integer lag `k` in `−max_lag:max_lag` that maximises
# `corr(pred(t), true(t−k))`. Positive ⇒ predicted signal (peaks) arrives late
# (behind); negative ⇒ early (ahead). `NaN32` if too few overlapping points or
# no variability. Timing is dominated by the large excursions, i.e. the peaks.
function _xcorr_lag(pv::AbstractVector, tv::AbstractVector, max_lag::Int, eps::Float32)
    T  = length(pv)
    ml = min(max_lag, T - 2)
    ml < 1 && return NaN32
    best_lag = 0
    best_c   = -Inf
    found    = false
    @inbounds for k in -ml:ml
        tlo = max(1, 1 + k)
        thi = min(T, T + k)
        n = 0
        sp = 0.0; st = 0.0; spp = 0.0; stt = 0.0; spt = 0.0
        for t in tlo:thi
            p = pv[t]; q = tv[t - k]
            (isfinite(p) && isfinite(q)) || continue
            n += 1
            sp += p; st += q; spp += p * p; stt += q * q; spt += p * q
        end
        n < 3 && continue
        cov = spt / n - (sp / n) * (st / n)
        vp  = spp / n - (sp / n)^2
        vt  = stt / n - (st / n)^2
        (vp > eps && vt > eps) || continue
        c = cov / sqrt(vp * vt)
        if c > best_c
            best_c = c; best_lag = k; found = true
        end
    end
    return found ? Float32(best_lag) : NaN32
end

"""
    spatial_error_metrics(pred_grids, true_grids, domain; eps = 1f-6, max_lag = 10)
        -> Dict{String, Dict{String, Matrix{Float32}}}

Reduce the `(nrows, ncols, T)` prediction/truth rasters over **time** into a set
of per-cell error maps, one nested `Dict` per state variable. Inactive cells
(all-`NaN` in the truth) are `NaN32`.

Metrics per cell (error `e = pred − truth`, positive `e` = **overprediction**):

- `rmse`          : `sqrt(mean(e²))`
- `bias`          : `mean(e)` — signed; the over/under-prediction map
- `relbias`       : `bias / mean(truth)` (`NaN` where `|mean(truth)| ≤ eps`)
- `overpred_freq` : fraction of timesteps with `e > 0` (0.5 ⇒ sign-unbiased)
- `peak_err`      : `maximum(pred) − maximum(truth)` — the flood-overshoot map
- `peak_lag`      : cross-correlation timing lag in timesteps (searched over
                    `±max_lag`); **positive ⇒ predicted peaks are behind (late),
                    negative ⇒ ahead (early)**
- `nbias`         : `bias / std(truth)` — bias in units of the cell's variability
- `nse`           : Nash–Sutcliffe efficiency `1 − Σe² / Σ(truth − mean(truth))²`

Only timesteps finite in **both** pred and truth contribute.
"""
function spatial_error_metrics(pred_grids ::Dict{String, Array{Float32,3}},
                               true_grids ::Dict{String, Array{Float32,3}},
                               domain     ::String;
                               eps        ::Float32 = 1f-6,
                               max_lag    ::Int     = 10)

    state_vars = DOMAIN_VARS[domain]["state"]
    out = Dict{String, Dict{String, Matrix{Float32}}}()

    for vname in state_vars
        haskey(pred_grids, vname) && haskey(true_grids, vname) || continue
        pg = pred_grids[vname]
        tg = true_grids[vname]
        nrows, ncols, T = size(tg)

        maps = Dict(m => fill(NaN32, nrows, ncols) for m in SPATIAL_METRIC_NAMES)

        @inbounds for j in 1:ncols, i in 1:nrows
            # Fast skip of inactive cells (truth all NaN).
            isnan(tg[i, j, 1]) && all(isnan, view(tg, i, j, :)) && continue

            n = 0
            se = 0.0        # Σ e²
            sb = 0.0        # Σ e
            st = 0.0        # Σ truth
            st2 = 0.0       # Σ truth²
            pmax = -Inf32
            tmax = -Inf32
            for t in 1:T
                tt = tg[i, j, t]
                pp = pg[i, j, t]
                (isfinite(tt) && isfinite(pp)) || continue
                e = pp - tt
                n += 1
                se += e * e
                sb += e
                st += tt
                st2 += tt * tt
                pp > pmax && (pmax = pp)
                tt > tmax && (tmax = tt)
                e > 0f0 && (maps["overpred_freq"][i, j] =
                    (isnan(maps["overpred_freq"][i, j]) ? 0f0 : maps["overpred_freq"][i, j]) + 1f0)
            end
            n == 0 && continue

            meant = st / n
            vart  = max(st2 / n - meant^2, 0.0)
            stdt  = sqrt(vart)
            bias  = sb / n

            maps["rmse"][i, j]  = Float32(sqrt(se / n))
            maps["bias"][i, j]  = Float32(bias)
            maps["peak_err"][i, j] = Float32(pmax - tmax)
            maps["peak_lag"][i, j] = _xcorr_lag(view(pg, i, j, :), view(tg, i, j, :), max_lag, eps)
            maps["relbias"][i, j]  = abs(meant) > eps ? Float32(bias / meant) : NaN32
            maps["nbias"][i, j]    = stdt > eps       ? Float32(bias / stdt)  : NaN32
            # overpred_freq currently holds the count of e>0; normalise to a fraction.
            cnt = maps["overpred_freq"][i, j]
            maps["overpred_freq"][i, j] = isnan(cnt) ? 0f0 : Float32(cnt / n)
            # NSE (undefined for a flat truth series).
            denom = st2 - n * meant^2
            maps["nse"][i, j] = denom > eps ? Float32(1 - se / denom) : NaN32
        end

        out[vname] = maps
    end

    return out
end

"""
    write_spatial_metrics_to_netcdf(metrics, staticmaps_file, path; schema)

Write the per-cell maps from [`spatial_error_metrics`](@ref) to a NetCDF with
`lon`/`lat` dimensions (no time axis). Each variable is named `"{var}_{metric}"`
(e.g. `river_q_rmse`). `NaN32` is stored as `_FillValue`.
"""
function write_spatial_metrics_to_netcdf(
        metrics         :: Dict{String, Dict{String, Matrix{Float32}}},
        staticmaps_file :: String,
        path            :: String;
        schema          :: WflowSchema = SCHEMA_V1)

    isempty(metrics) && throw(ArgumentError("metrics must not be empty"))

    lon_vals, lat_vals = NCDataset(staticmaps_file, "r") do ds
        dn = dimnames(ds[schema.ldd_var])
        Float64.(ds[dn[1]][:]), Float64.(ds[dn[2]][:])
    end

    NCDataset(path, "c") do ds
        defDim(ds, "lon", length(lon_vals))
        defDim(ds, "lat", length(lat_vals))
        defVar(ds, "lon", lon_vals, ("lon",);
               attrib = ["units" => "degrees_east",  "long_name" => "longitude",
                         "standard_name" => "longitude"])
        defVar(ds, "lat", lat_vals, ("lat",);
               attrib = ["units" => "degrees_north", "long_name" => "latitude",
                         "standard_name" => "latitude"])

        for (vname, maps) in metrics, m in SPATIAL_METRIC_NAMES
            haskey(maps, m) || continue
            v = defVar(ds, "$(vname)_$(m)", Float32, ("lon", "lat");
                       fillvalue = NaN32, attrib = ["long_name" => "$(vname) $(m)"])
            v[:, :] = replace(maps[m], NaN32 => missing)
        end
    end
    return path
end

"""
    spatial_metric_summary(metrics) -> Dict{String, Dict{String, NamedTuple}}

Aggregate the per-cell maps from [`spatial_error_metrics`](@ref) over all active
(finite) cells into compact summary statistics per `(variable, metric)`:
`(; n, mean, median, p10, p90, std)`. `NaN`/inactive cells are excluded. This is
the shared reduction backing both [`write_spatial_metrics_to_csv`](@ref) and the
per-run `metrics.toml` summary, so a text-only agent never needs to open the
`spatial_metrics.nc` grid.
"""
function spatial_metric_summary(metrics::Dict{String, Dict{String, Matrix{Float32}}})
    out = Dict{String, Dict{String, NamedTuple}}()
    for (vname, maps) in metrics
        mdict = Dict{String, NamedTuple}()
        for m in SPATIAL_METRIC_NAMES
            haskey(maps, m) || continue
            vals = filter(isfinite, vec(maps[m]))
            mdict[m] = isempty(vals) ?
                (; n = 0, mean = NaN, median = NaN, p10 = NaN, p90 = NaN, std = NaN) :
                (; n      = length(vals),
                   mean   = Float64(mean(vals)),
                   median = Float64(median(vals)),
                   p10    = Float64(quantile(vals, 0.10)),
                   p90    = Float64(quantile(vals, 0.90)),
                   std    = length(vals) > 1 ? Float64(std(vals)) : 0.0)
        end
        out[vname] = mdict
    end
    return out
end

"""
    write_spatial_metrics_to_csv(metrics, path) -> path

Write the aggregated [`spatial_metric_summary`](@ref) of the per-cell error maps
to a CSV with one row per `(variable, metric)` and columns
`variable,metric,n,mean,median,p10,p90,std`. A token-cheap, text-readable
companion to the full `spatial_metrics.nc` grid.
"""
function write_spatial_metrics_to_csv(metrics::Dict{String, Dict{String, Matrix{Float32}}},
                                      path::AbstractString)
    isempty(metrics) && throw(ArgumentError("metrics must not be empty"))
    summary = spatial_metric_summary(metrics)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "variable,metric,n,mean,median,p10,p90,std")
        for vname in sort(collect(keys(summary)))
            mdict = summary[vname]
            for m in SPATIAL_METRIC_NAMES
                haskey(mdict, m) || continue
                s = mdict[m]
                println(io, join((vname, m, s.n, s.mean, s.median, s.p10, s.p90, s.std), ","))
            end
        end
    end
    return path
end

# Ramp-rate class edges on g = (Qt − Qt-1)/max(Qt, eps): falling → sharp rise.
const RAMP_EDGES  = Float32[-Inf, -0.05, 0.05, 0.2, 0.4, 0.8, Inf]
const RAMP_LABELS = ["falling", "steady", "mild rise", "moderate rise",
                     "strong rise", "sharp rise (g≥0.8)"]

"""
    overprediction_vs_ramp(pred_grids, true_grids; qvar = "river_q",
                           eps = 1f-6, max_scatter = 200_000, seed = 1)
        -> NamedTuple

Test whether **discharge overprediction correlates with steep rises in Q**.

For every active river cell and timestep `t ≥ 2` it pairs the prediction error
`e = pred − truth` with the *true* normalised ramp rate
`g = (Qₜ − Qₜ₋₁) / max(Qₜ, eps)` (the same rise metric as
`scripts/regime_analysis.jl`). Computed in a single streaming pass.

Returns a `NamedTuple`:
- `n`               : number of (cell, timestep) pairs
- `pearson_e_g`     : Pearson corr of `e` vs `g`
- `pearson_op_g`    : Pearson corr of overprediction `max(e,0)` vs `g`
- `spearman_e_g`    : Spearman corr of `e` vs `g` (on the scatter subsample)
- `bin_labels`, `bin_n`, `bin_mean_err`, `bin_mean_overpred`, `bin_frac_over`
                    : per ramp-class summaries (`RAMP_LABELS`)
- `scatter_g`, `scatter_e` : a random subsample (≤ `max_scatter`) for plotting
- `corr_map`        : `(nrows, ncols)` per-cell Pearson corr of `e` vs `g`
"""
function overprediction_vs_ramp(pred_grids ::Dict{String, Array{Float32,3}},
                                true_grids ::Dict{String, Array{Float32,3}};
                                qvar        ::String  = "river_q",
                                eps         ::Float32 = 1f-6,
                                max_scatter ::Int     = 200_000,
                                seed        ::Int     = 1)

    haskey(pred_grids, qvar) && haskey(true_grids, qvar) ||
        throw(ArgumentError("both grids must contain \"$qvar\""))
    pg = pred_grids[qvar]
    tg = true_grids[qvar]
    nrows, ncols, T = size(tg)

    nb = length(RAMP_LABELS)
    bin_n     = zeros(Int, nb)
    bin_serr  = zeros(Float64, nb)
    bin_sop   = zeros(Float64, nb)
    bin_nover = zeros(Int, nb)

    # Pearson accumulators (single pass).
    n = 0
    Se = 0.0; Sg = 0.0; See = 0.0; Sgg = 0.0; Seg = 0.0
    So = 0.0; Soo = 0.0; Sog = 0.0

    rng = Random.MersenneTwister(seed)
    scat_g = Float32[]; scat_e = Float32[]
    sizehint!(scat_g, min(max_scatter, 1024))
    sizehint!(scat_e, min(max_scatter, 1024))
    seen = 0  # for reservoir sampling

    corr_map = fill(NaN32, nrows, ncols)

    _bin(g) = begin
        b = 1
        @inbounds for k in 2:length(RAMP_EDGES)
            if g < RAMP_EDGES[k]; b = k - 1; break; end
        end
        b
    end

    @inbounds for j in 1:ncols, i in 1:nrows
        isnan(tg[i, j, 1]) && all(isnan, view(tg, i, j, :)) && continue

        # Per-cell Pearson accumulators for the corr map.
        cn = 0; ce = 0.0; cg = 0.0; cee = 0.0; cgg = 0.0; ceg = 0.0

        for t in 2:T
            tt  = tg[i, j, t];   pp  = pg[i, j, t]
            tp  = tg[i, j, t-1]
            (isfinite(tt) && isfinite(pp) && isfinite(tp)) || continue
            g = (tt - tp) / max(tt, eps)
            e = pp - tt
            o = e > 0f0 ? e : 0f0

            n += 1
            Se += e; Sg += g; See += e*e; Sgg += g*g; Seg += e*g
            So += o; Soo += o*o; Sog += o*g

            b = _bin(g)
            bin_n[b]    += 1
            bin_serr[b] += e
            bin_sop[b]  += o
            e > 0f0 && (bin_nover[b] += 1)

            cn += 1; ce += e; cg += g; cee += e*e; cgg += g*g; ceg += e*g

            # Reservoir sampling for the scatter subset.
            seen += 1
            if length(scat_g) < max_scatter
                push!(scat_g, Float32(g)); push!(scat_e, Float32(e))
            else
                r = rand(rng, 1:seen)
                if r <= max_scatter
                    scat_g[r] = Float32(g); scat_e[r] = Float32(e)
                end
            end
        end

        if cn >= 3
            cov  = ceg/cn - (ce/cn)*(cg/cn)
            ve   = cee/cn - (ce/cn)^2
            vg   = cgg/cn - (cg/cn)^2
            corr_map[i, j] = (ve > 0 && vg > 0) ? Float32(cov / sqrt(ve*vg)) : NaN32
        end
    end

    _pearson(sx, sy, sxx, syy, sxy, m) = begin
        m < 2 && return NaN32
        cov = sxy/m - (sx/m)*(sy/m)
        vx  = sxx/m - (sx/m)^2
        vy  = syy/m - (sy/m)^2
        (vx > 0 && vy > 0) ? Float32(cov / sqrt(vx*vy)) : NaN32
    end

    pearson_e_g  = _pearson(Se, Sg, See, Sgg, Seg, n)
    pearson_op_g = _pearson(So, Sg, Soo, Sgg, Sog, n)

    # Spearman on the subsample: Pearson of the rank-transformed vectors.
    spearman_e_g = if length(scat_g) >= 3
        rg = _tiedranks(scat_g); re = _tiedranks(scat_e)
        cor(rg, re) |> Float32
    else
        NaN32
    end

    bin_mean_err      = Float32[bin_n[b] > 0 ? Float32(bin_serr[b]/bin_n[b]) : NaN32 for b in 1:nb]
    bin_mean_overpred = Float32[bin_n[b] > 0 ? Float32(bin_sop[b]/bin_n[b])  : NaN32 for b in 1:nb]
    bin_frac_over     = Float32[bin_n[b] > 0 ? Float32(bin_nover[b]/bin_n[b]) : NaN32 for b in 1:nb]

    return (; n, pearson_e_g, pearson_op_g, spearman_e_g,
              bin_labels = RAMP_LABELS, bin_n, bin_mean_err, bin_mean_overpred,
              bin_frac_over, scatter_g = scat_g, scatter_e = scat_e, corr_map)
end

# Average (fractional) ranks with ties, for Spearman correlation.
function _tiedranks(x::AbstractVector)
    n = length(x)
    p = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    @inbounds while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end
