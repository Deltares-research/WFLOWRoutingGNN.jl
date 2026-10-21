# Subcatchment error attribution (TODO 3e, docs/notes/subcatchment_error_attribution.md).
#
# Post-hoc diagnostic: which subcatchments *inject* discharge error, as opposed to
# merely carrying error that was advected in from upstream?
#
# A subcatchment partition is a set of *outlet cells* on the drainage network. Every
# cell is assigned to its first downstream outlet (the cell itself if it is one), so the
# regions tessellate the (drained) domain by construction. The outlets form a tree: the
# parent of outlet `o` is the first outlet strictly downstream of `o`. With
# ΔQ = Q_pred − Q_true, the error that subcatchment `o` adds on top of what its
# upstream neighbours `u_i` deliver is
#
#     e_local(o, t) = ΔQ(o, t) − Σ_i ΔQ(u_i, t)           (same time step, no lag)
#
# Both ways of choosing outlets (observed gauges = Route A; LDD-derived confluences =
# Route B) reduce to an outlet cell set, so everything below is route-agnostic.

const EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

"""
    cell_areas_km2(lat, dlon_deg, dlat_deg, nrows) -> Matrix{Float64}

Spherical cell area (km²) of a regular lon/lat raster with `nrows` longitude cells.
`lat` holds the cell-centre latitudes (degrees); the result is `(nrows, length(lat))`.
"""
function cell_areas_km2(lat::AbstractVector, dlon_deg::Real, dlat_deg::Real, nrows::Integer)
    dλ = deg2rad(abs(dlon_deg))
    h  = deg2rad(abs(dlat_deg)) / 2
    A  = Matrix{Float64}(undef, nrows, length(lat))
    for (j, φ) in enumerate(deg2rad.(lat))
        a = EARTH_RADIUS_KM^2 * dλ * (sin(φ + h) - sin(φ - h))
        A[:, j] .= abs(a)
    end
    return A
end

# ---------------------------------------------------------------------------
# Outlet selection
# ---------------------------------------------------------------------------

"""
    downstream_map(sources, targets, n) -> Vector{Int}

`down[c]` = linear index of the cell `c` drains into, or `0` for pits / cells without
an active downstream neighbour. `sources`/`targets` are the edge lists of
[`ldd_to_graph`](@ref).
"""
function downstream_map(sources::AbstractVector{<:Integer}, targets::AbstractVector{<:Integer},
                        n::Integer)
    down = zeros(Int, n)
    down[sources] .= targets
    return down
end

"""
    gauge_outlets(gauge_raster, river) -> (cells, ids)

Route A: outlets are the cells of a gauge raster (`wflow_gauges*`; non-missing,
non-zero values are gauge ids) that lie on the river network. Each cell appears once
(first id wins). `gauge_raster` is a `(nrows, ncols)` matrix with `NaN` for missing and
`river` a `Bool` vector over linear indices. Gauges off the river network are dropped.
"""
function gauge_outlets(gauge_raster::AbstractMatrix, river::AbstractVector{Bool})
    cells = Int[]
    ids   = Int[]
    seen  = Set{Int}()
    for c in eachindex(gauge_raster)
        g = gauge_raster[c]
        (isfinite(g) && g != 0 && river[c] && !(c in seen)) || continue
        push!(seen, c); push!(cells, c); push!(ids, round(Int, g))
    end
    return cells, ids
end

"""
    network_outlets(down, river, uparea_km2; min_area_km2) -> Vector{Int}

Route B: outlets derived from the river network alone, for models without
a usable gauge set. Outlets are

- river sinks (no downstream river cell) with upstream area ≥ `min_area_km2`, and
- the cell just upstream of every *significant confluence*, i.e. every upstream
  neighbour `u` with `uparea(u) ≥ min_area_km2` of a river cell that has ≥ 2 such
  neighbours.

Each outlet is thus the end of a branch that is large enough to matter, and the
subcatchment below a confluence is the main-stem stretch down to the next outlet.
"""
function network_outlets(down::AbstractVector{<:Integer}, river::AbstractVector{Bool},
                         uparea_km2::AbstractVector; min_area_km2::Real)
    big(c) = isfinite(uparea_km2[c]) && uparea_km2[c] >= min_area_km2
    n_big_up = zeros(Int, length(down))
    for c in findall(river)
        d = down[c]
        (d != 0 && river[d] && big(c)) && (n_big_up[d] += 1)
    end
    outlets = Int[]
    for c in findall(river)
        big(c) || continue
        d = down[c]
        if d == 0 || !river[d]
            push!(outlets, c)                       # basin / reach sink
        elseif n_big_up[d] >= 2
            push!(outlets, c)                       # branch end above a significant confluence
        end
    end
    return outlets
end

# ---------------------------------------------------------------------------
# Tessellation and tree
# ---------------------------------------------------------------------------

"""
    assign_subcatchments(down, outlets, active) -> Vector{Int}

`label[c]` = index `k` into `outlets` of the first outlet on the downstream path of
cell `c` (including `c` itself), or `0` if the path leaves the domain without meeting
one (or `c` is inactive). Throws on a drainage cycle.
"""
function assign_subcatchments(down::AbstractVector{<:Integer}, outlets::AbstractVector{<:Integer},
                              active::AbstractVector{Bool})
    n = length(down)
    length(unique(outlets)) == length(outlets) ||
        throw(ArgumentError("outlets must be unique cells"))
    all(o -> 1 <= o <= n && active[o], outlets) ||
        throw(ArgumentError("every outlet must be an active cell"))

    label = fill(-1, n)                              # -1 = not yet visited
    for (k, o) in enumerate(outlets)
        label[o] = k
    end
    path = Int[]
    for c in 1:n
        (active[c] && label[c] == -1) || continue
        empty!(path)
        cur = c
        res = 0
        while true
            if label[cur] != -1
                res = label[cur]; break
            end
            push!(path, cur)
            length(path) > n && error("drainage cycle encountered while tracing downstream from cell $c")
            nxt = down[cur]
            if nxt == 0 || !active[nxt]
                res = 0; break
            end
            cur = nxt
        end
        for p in path
            label[p] = res
        end
    end
    replace!(label, -1 => 0)
    return label
end

"""
    build_subcatchment_tree(down, active, outlets, cell_area_km2; uparea_km2 = nothing)
        -> NamedTuple

Tessellate the domain by `outlets` and build the outlet tree. Fields:

- `outlets`, `label` : see [`assign_subcatchments`](@ref)
- `parent[k]`        : tree parent of outlet `k` (`0` for a root)
- `children[k]`      : outlets whose parent is `k` (the *upstream inlets* of region `k`)
- `n_cells[k]`, `inc_area_km2[k]` : cells / area of region `k` itself (incremental)
- `cum_area_km2[k]`  : own area plus all upstream regions
- `check`            : verification summary (see below)

`check` verifies the structural properties the attribution relies on: every
region boundary crossing happens at an outlet and goes to its tree parent (so all
flow entering a region passes through its tracked inlets), the regions plus the
unassigned cells cover every active cell exactly once, and, when `uparea_km2` is
given, that the cumulative region area reproduces the model's own upstream area at
each outlet (`area_rel_err_max`/`_median`).
"""
function build_subcatchment_tree(down::AbstractVector{<:Integer}, active::AbstractVector{Bool},
                                 outlets::AbstractVector{<:Integer},
                                 cell_area_km2::AbstractVector;
                                 uparea_km2::Union{Nothing,AbstractVector} = nothing)
    K = length(outlets)
    label = assign_subcatchments(down, outlets, active)

    parent = zeros(Int, K)
    for k in 1:K
        d = down[outlets[k]]
        parent[k] = (d == 0 || !active[d]) ? 0 : label[d]
    end
    children = [Int[] for _ in 1:K]
    for k in 1:K
        parent[k] > 0 && push!(children[parent[k]], k)
    end

    n_cells = zeros(Int, K)
    inc_area = zeros(Float64, K)
    for c in findall(active)
        k = label[c]
        k > 0 || continue
        n_cells[k] += 1
        inc_area[k] += cell_area_km2[c]
    end

    cum_area = copy(inc_area)
    cum_done = falses(K)
    function cum!(k)
        cum_done[k] && return cum_area[k]
        cum_area[k] = inc_area[k] + sum((cum!(u) for u in children[k]); init = 0.0)
        cum_done[k] = true
        return cum_area[k]
    end
    foreach(cum!, 1:K)

    # Verification -------------------------------------------------------
    n_active     = count(active)
    n_unassigned = count(c -> active[c] && label[c] == 0, 1:length(label))
    boundary_ok = true
    for c in findall(active)
        k = label[c]
        k > 0 || continue
        d = down[c]
        m = (d == 0 || !active[d]) ? 0 : label[d]
        m == k && continue
        if !(c == outlets[k] && m == parent[k])
            boundary_ok = false
            break
        end
    end
    covered_ok = (sum(n_cells) + n_unassigned == n_active)
    rel = Float64[]
    if uparea_km2 !== nothing
        for k in 1:K
            ua = uparea_km2[outlets[k]]
            (isfinite(ua) && ua > 0) && push!(rel, abs(cum_area[k] - ua) / ua)
        end
    end
    check = (; boundary_ok, covered_ok, n_active, n_unassigned,
               area_rel_err_max    = isempty(rel) ? NaN : maximum(rel),
               area_rel_err_median = isempty(rel) ? NaN : median(rel))

    return (; outlets = collect(Int, outlets), label, parent, children,
              n_cells, inc_area_km2 = inc_area, cum_area_km2 = cum_area, check)
end

"""
    mainstem(tree) -> Vector{Int}

Outlet indices along the main stem, upstream to downstream: from the root with the
largest cumulative area, repeatedly follow the child with the largest cumulative area.
"""
function mainstem(tree)
    K = length(tree.outlets)
    K == 0 && return Int[]
    roots = findall(==(0), tree.parent)
    k = roots[argmax(tree.cum_area_km2[roots])]
    path = [k]
    while !isempty(tree.children[k])
        k = tree.children[k][argmax(tree.cum_area_km2[tree.children[k]])]
        push!(path, k)
    end
    return reverse(path)
end

# ---------------------------------------------------------------------------
# Error decomposition and skill
# ---------------------------------------------------------------------------

_rms(x) = isempty(x) ? NaN : sqrt(mean(abs2, x))

function _outlet_skill(p::AbstractVector, t::AbstractVector)
    m = isfinite.(p) .& isfinite.(t)
    any(m) || return (; nse = NaN, kge = NaN, r = NaN, alpha = NaN, beta = NaN,
                        pbias = NaN, peak_ratio = NaN, rmse = NaN)
    pm = Float64.(p[m]); tm = Float64.(t[m])
    den = sum(abs2, tm .- mean(tm))
    nse = den > 0 ? 1 - sum(abs2, pm .- tm) / den : NaN
    k  = kge_metrics(pm, tm)
    pk = event_peak_metrics(pm, tm)
    return (; nse, kge = Float64(k.kge), r = Float64(k.r), alpha = Float64(k.alpha),
              beta = Float64(k.beta), pbias = Float64(pbias(pm, tm)),
              peak_ratio = Float64(pk.peak_ratio), rmse = sqrt(mean(abs2, pm .- tm)))
end

"""
    subcatch_error_decomposition(tree, pred_q, true_q; qbankfull = nothing) -> Vector{NamedTuple}

Per-outlet skill and error decomposition. `pred_q`/`true_q` are `(n_cells, T)` matrices
indexed by linear cell index (e.g. `reshape(grid, nrows*ncols, T)`). For outlet `o`
with inlets `u_i` (its tree children) and `ΔQ = Q_pred − Q_true`:

- `rms_total`      : `rms(ΔQ(o))`
- `rms_adv`        : `rms(Σ ΔQ(u_i))`, the error delivered by the inlets (0 for a headwater)
- `rms_local`      : `rms(e_local)`, `e_local = ΔQ(o) − Σ ΔQ(u_i)` (same time step)
- `mean_local`     : signed mean of `e_local` (positive = region creates water in the model)
- `local_over_total` : `rms_local / rms_total`
- `mean_true_local`  : mean of `Q_true(o) − Σ Q_true(u_i)` (true water generated in the region)
- `rel_local`        : `rms_local / mean_true_local` (NaN if the latter is ≤ 0)
- `frac_overbank`    : fraction of steps with `Q_true(o) > qbankfull(o)` (needs `qbankfull`,
                       a vector over linear cell index)
- `n_steps`          : steps where the outlet and all its inlets are finite

plus the skill of the outlet discharge itself (`nse, kge, r, alpha, beta, pbias,
peak_ratio, rmse`). `e_local` differs from the true local injection by the routing
lag between inlets and outlet, so it is a same-time (no-lag) attribution.
"""
function subcatch_error_decomposition(tree, pred_q::AbstractMatrix, true_q::AbstractMatrix;
                                      qbankfull::Union{Nothing,AbstractVector} = nothing)
    size(pred_q) == size(true_q) || throw(ArgumentError("pred_q and true_q must have the same shape"))
    T = size(pred_q, 2)
    out = NamedTuple[]
    for (k, o) in enumerate(tree.outlets)
        ins = tree.outlets[tree.children[k]]
        qp = Float64.(pred_q[o, :]);  qt = Float64.(true_q[o, :])
        adv   = zeros(T)
        tl    = copy(qt)
        ok    = isfinite.(qp) .& isfinite.(qt)
        for u in ins
            up = Float64.(pred_q[u, :]); ut = Float64.(true_q[u, :])
            ok .&= isfinite.(up) .& isfinite.(ut)
            adv .+= up .- ut
            tl  .-= ut
        end
        dq    = qp .- qt
        eloc  = (dq .- adv)[ok]
        advk  = adv[ok]
        mtl   = isempty(eloc) ? NaN : mean(tl[ok])
        rl    = _rms(eloc)
        fo    = (qbankfull !== nothing && isfinite(qbankfull[o]) && qbankfull[o] > 0 && any(ok)) ?
                mean(qt[ok] .> qbankfull[o]) : NaN
        push!(out, merge(_outlet_skill(qp, qt),
            (; rms_total = _rms(dq[ok]), rms_adv = _rms(advk), rms_local = rl,
               mean_local = isempty(eloc) ? NaN : mean(eloc),
               local_over_total = _rms(dq[ok]) > 0 ? rl / _rms(dq[ok]) : NaN,
               mean_true_local = mtl,
               rel_local = (isfinite(mtl) && mtl > 0) ? rl / mtl : NaN,
               frac_overbank = fo, n_steps = count(ok))))
    end
    return out
end

# ---------------------------------------------------------------------------
# Static attributes
# ---------------------------------------------------------------------------

"""
Attribute table: `(name, var, reduce)` where `reduce` is one of

- `:outlet` : value at the outlet cell,
- `:mean`   : mean over the finite cells of the region (land + river),
- `:river_mean` : mean over the region's river cells,
- `:presence`   : fraction of region cells with a finite value `> 0`.

Variables missing from `staticmaps.nc` (or not 2-D) are skipped.
"""
const SUBCATCH_ATTRIBUTES = [
    (name = "uparea_km2",          var = "wflow_uparea",         reduce = :outlet),
    (name = "streamorder",         var = "wflow_streamorder",    reduce = :outlet),
    (name = "river_slope_outlet",  var = "RiverSlope",           reduce = :outlet),
    (name = "ksat_ver",            var = "KsatVer",              reduce = :mean),
    (name = "soil_thickness",      var = "SoilThickness",        reduce = :mean),
    (name = "M",                   var = "M",                    reduce = :mean),
    (name = "f",                   var = "f",                    reduce = :mean),
    (name = "path_frac",           var = "PathFrac",             reduce = :mean),
    (name = "water_frac",          var = "WaterFrac",            reduce = :mean),
    (name = "land_slope",          var = "Slope",                reduce = :mean),
    (name = "glacier_frac",        var = "wflow_glacierfrac",    reduce = :mean),
    (name = "n_river",             var = "N_River",              reduce = :river_mean),
    (name = "river_slope_mean",    var = "RiverSlope",           reduce = :river_mean),
    (name = "reservoir_frac",      var = "wflow_reservoirareas", reduce = :presence),
    (name = "lake_frac",           var = "wflow_lakeareas",      reduce = :presence),
]

"""
    subcatch_attributes(tree, rasters, river; attributes = SUBCATCH_ATTRIBUTES)
        -> OrderedPairs (Vector{Pair{String,Vector{Float64}}})

Aggregate static attributes over each region of `tree`. `rasters` maps a staticmaps
variable name to a `(nrows, ncols)` `Float64` matrix with `NaN` for missing; attributes
whose variable is absent from `rasters` are skipped. `river` is a `Bool` vector over
linear cell index.
"""
function subcatch_attributes(tree, rasters::AbstractDict, river::AbstractVector{Bool};
                             attributes = SUBCATCH_ATTRIBUTES)
    K = length(tree.outlets)
    cells_of = [Int[] for _ in 1:K]
    for c in eachindex(tree.label)
        k = tree.label[c]
        k > 0 && push!(cells_of[k], c)
    end
    out = Pair{String,Vector{Float64}}[]
    for a in attributes
        haskey(rasters, a.var) || continue
        R = rasters[a.var]
        vals = fill(NaN, K)
        for k in 1:K
            cs = cells_of[k]
            if a.reduce === :outlet
                vals[k] = R[tree.outlets[k]]
            elseif a.reduce === :mean
                v = filter(isfinite, R[cs])
                isempty(v) || (vals[k] = mean(v))
            elseif a.reduce === :river_mean
                v = filter(isfinite, R[filter(c -> river[c], cs)])
                isempty(v) || (vals[k] = mean(v))
            elseif a.reduce === :presence
                isempty(cs) || (vals[k] = count(c -> isfinite(R[c]) && R[c] > 0, cs) / length(cs))
            else
                throw(ArgumentError("unknown reduction $(a.reduce)"))
            end
        end
        push!(out, a.name => vals)
    end
    return out
end

# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

"""
    spearman_test(x, y; nperm = 5000, seed = 1) -> NamedTuple

Spearman rank correlation of `x` and `y` over the pairs where both are finite, with

- `rho`      : the coefficient (`NaN` if either input is constant or `n < 4`),
- `n`        : number of pairs,
- `p`        : two-sided permutation p-value (`nperm` shuffles of the ranks; honest for
               the small `n` typical here),
- `ci_lo/hi` : approximate 95 % Fisher-z interval with the Bonett–Wright variance
               `(1 + ρ²/2)/(n − 3)` (unreliable for very small `n`).
"""
function spearman_test(x::AbstractVector, y::AbstractVector; nperm::Int = 5000, seed::Int = 1)
    length(x) == length(y) || throw(ArgumentError("x and y must have equal length"))
    m = isfinite.(x) .& isfinite.(y)
    n = count(m)
    nan = (; rho = NaN, n, p = NaN, ci_lo = NaN, ci_hi = NaN)
    n >= 4 || return nan
    xs = Float64.(x[m]); ys = Float64.(y[m])
    (allequal(xs) || allequal(ys)) && return nan
    rx = _tiedranks(xs); ry = _tiedranks(ys)
    ρ = cor(rx, ry)
    rng = MersenneTwister(seed)
    perm = copy(ry)
    hits = 1
    for _ in 1:nperm
        shuffle!(rng, perm)
        abs(cor(rx, perm)) >= abs(ρ) - 1e-12 && (hits += 1)
    end
    p = hits / (nperm + 1)
    ρc = clamp(ρ, -0.999999, 0.999999)
    se = sqrt((1 + ρc^2 / 2) / max(n - 3, 1))
    return (; rho = ρ, n, p, ci_lo = tanh(atanh(ρc) - 1.96se), ci_hi = tanh(atanh(ρc) + 1.96se))
end

"""
    rank_attributes(attrs, target; min_n = 5, nperm = 5000) -> Vector{NamedTuple}

Spearman-rank every attribute in `attrs` (a `Vector{Pair{String,Vector{Float64}}}`)
against `target`, sorted by `|ρ|`. Adds `p_bonferroni = min(1, p·m)` over the `m`
attributes tested and `low_power = n < 10`. Attributes that are effectively constant
(relative spread `(max − min)/|mean| ≤ rel_spread_min` over the finite values) are skipped,
since their ranks only reflect rounding noise.
"""
function rank_attributes(attrs, target::AbstractVector; min_n::Int = 5, nperm::Int = 5000,
                         rel_spread_min::Real = 1e-3)
    rows = NamedTuple[]
    for (name, vals) in attrs
        fv = filter(isfinite, vals)
        isempty(fv) && continue
        (maximum(fv) - minimum(fv)) <= rel_spread_min * abs(mean(fv)) && continue
        s = spearman_test(vals, target; nperm)
        s.n >= min_n || continue
        isfinite(s.rho) || continue
        push!(rows, (; attribute = name, s.rho, s.n, s.p, s.ci_lo, s.ci_hi))
    end
    m = length(rows)
    rows = [merge(r, (; p_bonferroni = min(1.0, r.p * m), low_power = r.n < 10)) for r in rows]
    sort!(rows; by = r -> -abs(r.rho))
    return rows
end

# ---------------------------------------------------------------------------
# Run-level driver
# ---------------------------------------------------------------------------

function _read_raster(ds, name::AbstractString)
    haskey(ds, name) || return nothing
    v = ds[name]
    length(dimnames(v)) == 2 || return nothing
    eltype(v) <: Union{Missing,Number} || return nothing
    return Float64.(coalesce.(v[:, :], NaN))
end

"""
    subcatch_error_attribution(staticmaps_file, schema, pred_q, true_q;
                               gauge_var = nothing, min_area_km2 = 100.0, outlets = nothing,
                               min_n_rank = 5, nperm = 5000) -> NamedTuple

Core of [`subcatch_error_attribution_run`](@ref), on in-memory `(nrows, ncols, T)`
discharge grids (`NaN` at inactive cells). Outlets are, in order of priority, the
explicit cell vector `outlets`; the gauge cells of the raster `gauge_var` (Route A); or
the network confluence outlets (Route B, [`network_outlets`](@ref)). Outlets without a
finite discharge series are dropped.

Returns `(; tree, rows, attributes, ranking, route, nrows, ncols, lon, lat, min_area_km2)`.
`rows` has one `NamedTuple` per outlet (skill, error decomposition, attributes, `retained`);
`ranking` maps each target (`"rms_local_per_km2"`, `"rel_local"`) to the
[`rank_attributes`](@ref) result over the retained regions.
"""
function subcatch_error_attribution(staticmaps_file::AbstractString, schema::WflowSchema,
                                    pred_q::AbstractArray{<:Real,3}, true_q::AbstractArray{<:Real,3};
                                    gauge_var::Union{Nothing,AbstractString} = nothing,
                                    min_area_km2::Real = 100.0,
                                    outlets::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                    min_n_rank::Int = 5, nperm::Int = 5000)
    size(pred_q) == size(true_q) || throw(ArgumentError("pred_q and true_q must have the same shape"))
    nrows, ncols, T = size(pred_q)

    src, tgt = ldd_to_graph(String(staticmaps_file), "land", schema)
    local ldd, river_raw, lon, lat, rasters, gauges
    rasters = Dict{String,Matrix{Float64}}()
    NCDataset(staticmaps_file, "r") do ds
        ldd = _read_raster(ds, schema.ldd_var)
        river_raw = _read_raster(ds, schema.mask_var)
        size(ldd) == (nrows, ncols) ||
            throw(ArgumentError("staticmaps raster $(size(ldd)) does not match the discharge grid ($nrows, $ncols)"))
        dn = dimnames(ds[schema.ldd_var])
        lon = Float64.(ds[dn[1]][:]); lat = Float64.(ds[dn[2]][:])
        for a in SUBCATCH_ATTRIBUTES
            haskey(rasters, a.var) && continue
            r = _read_raster(ds, a.var)
            r === nothing || (rasters[a.var] = r)
        end
        for v in ("wflow_uparea", "qbankfull")
            haskey(rasters, v) || (r = _read_raster(ds, v); r === nothing || (rasters[v] = r))
        end
        gauges = gauge_var === nothing ? nothing : _read_raster(ds, gauge_var)
        gauge_var !== nothing && gauges === nothing &&
            throw(ArgumentError("staticmaps has no 2-D variable \"$gauge_var\""))
    end

    active = vec(isfinite.(ldd))
    river  = vec(isfinite.(river_raw) .& (river_raw .!= 0))
    down   = downstream_map(src, tgt, nrows * ncols)
    uparea = haskey(rasters, "wflow_uparea") ? vec(rasters["wflow_uparea"]) : nothing
    area   = vec(cell_areas_km2(lat, length(lon) > 1 ? lon[2] - lon[1] : 0.0,
                                length(lat) > 1 ? lat[2] - lat[1] : 0.0, nrows))

    P = reshape(pred_q, nrows * ncols, T)
    Q = reshape(true_q, nrows * ncols, T)

    max_pred = maximum((abs(x) for x in P if isfinite(x)); init = 0.0)
    max_true = maximum((abs(x) for x in Q if isfinite(x)); init = 0.0)
    max_pred > 1e3 * max(max_true, eps()) &&
        @warn "predicted discharge is implausibly large (max |Q_pred| = $max_pred vs max |Q_true| = $max_true): " *
              "the rollout looks numerically diverged, so the attribution below is meaningless"

    ids = nothing
    route = :explicit
    if outlets !== nothing
        cells = collect(Int, outlets)
    elseif gauge_var !== nothing
        cells, ids = gauge_outlets(gauges, river)
        route = :gauges
    else
        uparea === nothing && throw(ArgumentError("Route B needs wflow_uparea in staticmaps.nc"))
        cells = network_outlets(down, river, uparea; min_area_km2)
        route = :network
    end
    keep = [any(isfinite, @view P[c, :]) && any(isfinite, @view Q[c, :]) for c in cells]
    all(keep) || @warn "dropping $(count(!, keep)) outlet(s) without a finite discharge series"
    cells = cells[keep]; ids === nothing || (ids = ids[keep])
    isempty(cells) && throw(ArgumentError("no usable outlets (route = $route)"))

    tree = build_subcatchment_tree(down, active, cells, area; uparea_km2 = uparea)
    tree.check.boundary_ok && tree.check.covered_ok ||
        error("subcatchment tessellation failed verification: $(tree.check)")

    qbf = haskey(rasters, "qbankfull") ? vec(rasters["qbankfull"]) : nothing
    dec = subcatch_error_decomposition(tree, P, Q; qbankfull = qbf)
    attrs = subcatch_attributes(tree, rasters, river)
    push!(attrs, "inc_area_km2" => copy(tree.inc_area_km2))
    qbf === nothing || push!(attrs, "frac_overbank" => [d.frac_overbank for d in dec])

    K = length(cells)
    retained = tree.inc_area_km2 .>= min_area_km2
    rows = map(1:K) do k
        c = cells[k]
        d = dec[k]
        p = tree.parent[k]
        merge((; outlet_id = ids === nothing ? c : ids[k], outlet_cell = c,
                 lon = lon[mod1(c, nrows)], lat = lat[cld(c, nrows)],
                 parent_id = p == 0 ? 0 : (ids === nothing ? cells[p] : ids[p]),
                 n_inlets = length(tree.children[k]), n_cells = tree.n_cells[k],
                 inc_area_km2 = tree.inc_area_km2[k], cum_area_km2 = tree.cum_area_km2[k],
                 retained = retained[k]),
              d,
              (; rms_local_per_km2 = d.rms_local / tree.inc_area_km2[k]),
              NamedTuple(Symbol(n) => v[k] for (n, v) in attrs))
    end

    sel = findall(retained)
    ranking = Dict{String,Vector{NamedTuple}}()
    for tname in (:rms_local_per_km2, :rel_local)
        y = [getproperty(rows[k], tname) for k in sel]
        sub = [n => v[sel] for (n, v) in attrs]
        ranking[String(tname)] = rank_attributes(sub, y; min_n = min_n_rank, nperm)
    end
    return (; tree, rows, attributes = attrs, ranking, route, nrows, ncols, lon, lat, min_area_km2)
end

function _write_rows_csv(path::AbstractString, rows::AbstractVector{<:NamedTuple})
    isempty(rows) && return path
    ks = collect(keys(first(rows)))
    return _write_plot_csv(path, String.(ks), Any[[getproperty(r, k) for r in rows] for k in ks])
end

"""
    subcatch_error_attribution_run(run_dir; split = "val", gauge_var = nothing,
                                   min_area_km2 = 100.0, staticmaps_file = nothing,
                                   qvar = "river_q", ...) -> NamedTuple

Post-hoc subcatchment error attribution for an existing run, from its gridded
`output/<split>_pred.nc` / `output/<split>_true.nc` (settings from `model/*.toml`;
legacy runs with everything in the run root are accepted). Works without a model or
retraining. See [`subcatch_error_attribution`](@ref) for the method and the outlet
choice (`gauge_var = "wflow_gauges_grdc"` etc. selects Route A, the default is the
network-derived Route B).

Writes

    metrics/<split>_subcatch_error_attribution.csv   one row per subcatchment
    metrics/<split>_subcatch_attribute_ranking.csv   Spearman ρ of attributes vs error
    plots/<split>_subcatch_error_map.png, <split>_subcatch_mainstem.png,
    plots/<split>_subcatch_attributes.png
"""
function subcatch_error_attribution_run(run_dir::AbstractString;
                                        split::AbstractString = "val",
                                        staticmaps_file::Union{Nothing,AbstractString} = nothing,
                                        qvar::AbstractString = "river_q",
                                        kwargs...)
    run_dir = abspath(run_dir)
    isdir(run_dir) || throw(ArgumentError("not a directory: $run_dir"))
    dirs = _run_dirs(run_dir)
    model_dir  = isfile(joinpath(dirs.model, "data_settings.toml")) ? dirs.model : run_dir
    output_dir = isfile(joinpath(dirs.output, "$(split)_pred.nc")) ? dirs.output : run_dir
    ds = load_data_settings(joinpath(model_dir, "data_settings.toml"))
    schema = load_schema(ds.wflow_schema)
    sm = staticmaps_file === nothing ? resolve_staticmaps(ds.wflow_model_path, run_dir) : String(staticmaps_file)
    pred_path = joinpath(output_dir, "$(split)_pred.nc")
    true_path = joinpath(output_dir, "$(split)_true.nc")
    isfile(pred_path) || throw(ArgumentError("missing $pred_path"))
    isfile(true_path) || throw(ArgumentError("missing $true_path"))
    foreach(mkpath, (dirs.metrics, dirs.plots))

    @info "Loading $qvar ($split) from $run_dir"
    pg = load_regrid_nc(pred_path, [qvar])[qvar]
    tg = load_regrid_nc(true_path, [qvar])[qvar]
    res = subcatch_error_attribution(sm, schema, pg, tg; kwargs...)

    csv  = joinpath(dirs.metrics, "$(split)_subcatch_error_attribution.csv")
    rcsv = joinpath(dirs.metrics, "$(split)_subcatch_attribute_ranking.csv")
    _write_rows_csv(csv, res.rows)
    rk = [merge((; target = t), r) for t in sort(collect(keys(res.ranking))) for r in res.ranking[t]]
    _write_rows_csv(rcsv, rk)
    pngs = plot_subcatch_error_attribution(res; dir = dirs.plots, prefix = "$(split)_subcatch")
    @info "Wrote $csv, $rcsv and $(length(pngs)) plots" check = res.tree.check
    return merge(res, (; csv, ranking_csv = rcsv, plots = pngs))
end

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

"""
    plot_subcatch_error_attribution(res; dir, prefix) -> Vector{String}

Three figures from a [`subcatch_error_attribution`](@ref) result: (1) maps of the
subcatchments coloured by specific local error `rms_local / area` and by the local share
of the outlet error; (2) advected vs local RMS error along the main stem; (3) the
strongest attribute-vs-error rank correlations. Returns the written paths.
"""
function plot_subcatch_error_attribution(res; dir::AbstractString, prefix::AbstractString)
    mkpath(dir)
    rows, tree = res.rows, res.tree
    K = length(rows)
    paths = String[]

    # (1) maps ---------------------------------------------------------------
    lon, lat = res.lon, res.lat
    flip = length(lat) > 1 && lat[1] > lat[end]
    latp = flip ? reverse(lat) : lat
    fig = Figure(size = (1300, 560))
    for (j, (field, ttl)) in enumerate(((:rms_local_per_km2, "specific local error rms(e_local)/area  [m³/s/km²]"),
                                        (:local_over_total,  "local share rms(e_local)/rms(ΔQ outlet)")))
        G = fill(NaN, res.nrows * res.ncols)
        for c in eachindex(tree.label)
            k = tree.label[c]
            (k > 0 && rows[k].retained) && (G[c] = getproperty(rows[k], field))
        end
        G = reshape(G, res.nrows, res.ncols)
        G = flip ? G[:, end:-1:1] : G
        ax = Axis(fig[1, 2j - 1]; title = ttl, titlesize = 12, xlabel = "lon", ylabel = "lat",
                  aspect = DataAspect())
        hm = heatmap!(ax, lon, latp, G; colormap = :viridis, nan_color = (:lightgray, 0.6))
        scatter!(ax, [r.lon for r in rows], [r.lat for r in rows]; color = :red, markersize = 7,
                 strokecolor = :black, strokewidth = 0.5)
        Colorbar(fig[1, 2j], hm)
    end
    p1 = joinpath(dir, "$(prefix)_error_map.png"); save(p1, fig); push!(paths, p1)

    # (2) main stem ----------------------------------------------------------
    ms = mainstem(tree)
    if !isempty(ms)
        fig = Figure(size = (1000, 480))
        ax = Axis(fig[1, 1]; xlabel = "outlets along main stem (upstream → downstream)",
                  ylabel = "RMS discharge error [m³/s]",
                  title = "advected vs local error along the main stem",
                  xticks = (1:length(ms), [string(round(Int, rows[k].cum_area_km2)) * " km²" for k in ms]),
                  xticklabelrotation = π / 4)
        x = 1:length(ms)
        barplot!(ax, x .- 0.2, [rows[k].rms_adv for k in ms]; width = 0.4, color = :steelblue,
                 label = "advected  rms(Σ ΔQ inlets)")
        barplot!(ax, x .+ 0.2, [rows[k].rms_local for k in ms]; width = 0.4, color = :tomato,
                 label = "local  rms(e_local)")
        lines!(ax, x, [rows[k].rms_total for k in ms]; color = :black, label = "outlet  rms(ΔQ)")
        axislegend(ax; position = :lt)
        p2 = joinpath(dir, "$(prefix)_mainstem.png"); save(p2, fig); push!(paths, p2)
    end

    # (3) attribute scatter --------------------------------------------------
    rk = get(res.ranking, "rms_local_per_km2", NamedTuple[])
    if !isempty(rk)
        top = first(rk, min(6, length(rk)))
        sel = findall(r -> r.retained, rows)
        y = [rows[k].rms_local_per_km2 for k in sel]
        fig = Figure(size = (1200, 760))
        for (i, r) in enumerate(top)
            vals = Dict(res.attributes)[r.attribute][sel]
            ax = Axis(fig[cld(i, 3), mod1(i, 3)]; xlabel = r.attribute,
                      ylabel = "rms(e_local)/area  [m³/s/km²]",
                      title = "ρ = $(round(r.rho; digits = 2))  (n = $(r.n), p = $(round(r.p; digits = 3)))",
                      titlesize = 12)
            scatter!(ax, vals, y; color = :steelblue)
        end
        p3 = joinpath(dir, "$(prefix)_attributes.png"); save(p3, fig); push!(paths, p3)
    end
    return paths
end
