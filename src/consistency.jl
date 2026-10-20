using NCDatasets
using Statistics

# ---------------------------------------------------------------------------
# Layer-free consistency checks of wflow data (promoted from
# `scripts/check_mass_balance.jl` and `scripts/check_mass_balance_details.jl`).
#
# Everything here works on the raw `staticmaps.nc` / `output.nc` of a wflow model:
# no model, no normalisation. It answers "does the kinematic-wave mass balance
# hold in the training data, and which discretisation does wflow use?", the
# prototype of the post-hoc residual diagnostic `mb_residual_diagnostics`
# (src/postprocess.jl), which applies the same residual to predicted fields.
# ---------------------------------------------------------------------------

_nan32(x) = ismissing(x) ? NaN32 : Float32(x)

"""
    river_layout(staticmaps_file, output_file) -> NamedTuple

Compact river-node layout of a wflow model, built exactly like preprocessing does:

    n_nodes, nrows, ncols, rows, cols, node_ids   node → raster cell (column-major ids)
    edge_src, edge_tgt                           edges in compact node indices (`src` drains into `tgt`)
    width, length_m                              channel geometry per node [m]
    dim1_flip, dim2_flip                         axis direction of `output_file` vs `staticmaps_file`
    dt                                           model timestep [s]
"""
function river_layout(staticmaps_file::AbstractString, output_file::AbstractString)
    staticmaps_file, output_file = String(staticmaps_file), String(output_file)
    dt        = get_timestep(output_file)
    alignment = check_and_correct_grid_alignment(staticmaps_file, output_file, "river")
    src_raw, tgt_raw = ldd_to_graph(staticmaps_file, "river")

    nrows, ncols, rows, cols, node_ids = NCDataset(staticmaps_file, "r") do ds
        mask = ds["river_mask"][:, :]
        nr, nc = size(mask)
        ids = sort([((c - 1) * nr + r)
                    for c in 1:nc, r in 1:nr
                    if !ismissing(mask[r, c]) && mask[r, c] != 0])
        rs  = [(id - 1) % nr + 1 for id in ids]
        cs  = [(id - 1) ÷ nr + 1 for id in ids]
        nr, nc, rs, cs, ids
    end

    id_to_idx = Dict(id => i for (i, id) in enumerate(node_ids))
    edge_src  = [id_to_idx[s] for s in src_raw]
    edge_tgt  = [id_to_idx[t] for t in tgt_raw]

    width, length_m = NCDataset(staticmaps_file, "r") do ds
        n = length(rows)
        w = [_nan32(ds["river_width"][rows[i],  cols[i]]) for i in 1:n]
        l = [_nan32(ds["river_length"][rows[i], cols[i]]) for i in 1:n]
        w, l
    end

    return (; n_nodes = length(rows), nrows, ncols, rows, cols, node_ids,
            edge_src, edge_tgt, width, length_m,
            dim1_flip = alignment.dim1_flip, dim2_flip = alignment.dim2_flip, dt)
end

"""
    read_river_series(ds, layout, varname) -> Union{Nothing, Matrix{Float32}}

Read the 3-D `(lon, lat, time)` variable `varname` of the open `output.nc` dataset
`ds` as an `(n_nodes, ntimes)` matrix in the node order of `layout`
([`river_layout`](@ref)), undoing axis flips. `missing` becomes `NaN`. Returns
`nothing` (with a warning for fewer than three dimensions) when the variable is
absent or not 3-D.
"""
function read_river_series(ds, layout, varname::AbstractString)
    varname in keys(ds) || return nothing
    v = ds[varname]
    if ndims(v) < 3
        @warn "  $varname has fewer than 3 dimensions — skipping"
        return nothing
    end
    data3d = _nan32.(v[:, :, :])
    nt     = size(data3d, 3)
    mat    = Matrix{Float32}(undef, layout.n_nodes, nt)
    for (i, (r, c)) in enumerate(zip(layout.rows, layout.cols))
        r_out = layout.dim1_flip ? layout.nrows - r + 1 : r
        c_out = layout.dim2_flip ? layout.ncols - c + 1 : c
        mat[i, :] = data3d[r_out, c_out, :]
    end
    return mat
end

"""
    upstream_sum(q, edge_src, edge_tgt) -> similar(q)

Sum of `q` over the upstream neighbours of every node: `out[i] = Σ q[j]` over the
edges `j → i`. `q` is a vector `(n_nodes,)` or a matrix `(n_nodes, T)` (per column).
"""
function upstream_sum(q::AbstractVector, edge_src::AbstractVector{<:Integer},
                      edge_tgt::AbstractVector{<:Integer})
    out = zeros(eltype(q), length(q))
    for k in eachindex(edge_src)
        out[edge_tgt[k]] += q[edge_src[k]]
    end
    return out
end

function upstream_sum(q::AbstractMatrix, edge_src::AbstractVector{<:Integer},
                      edge_tgt::AbstractVector{<:Integer})
    out = zeros(eltype(q), size(q))
    for t in axes(q, 2), k in eachindex(edge_src)
        out[edge_tgt[k], t] += q[edge_src[k], t]
    end
    return out
end

"""
    MB_SCHEMES

Discretisations of the kinematic-wave mass balance compared by
[`mb_scheme_residuals`](@ref), with the update `h[t+1] = h[t] + dt/(w·l)·(Σq_up + iw − q[t+1])`:

| scheme              | upstream sum | lateral inflow |
|:--------------------|:-------------|:---------------|
| `:semi_implicit`    | `Σq[t+1]`    | `iw[t]`        |
| `:fully_implicit`   | `Σq[t+1]`    | `iw[t+1]`      |
| `:lagged_upstream`  | `Σq[t]`      | `iw[t]`        |

wflow uses a same-timestep upstream sum (`qin_av[t] = Σ q_av[t]`), so the
semi-/fully-implicit schemes are the consistent ones.
"""
const MB_SCHEMES = (:semi_implicit, :fully_implicit, :lagged_upstream)

"""
    mb_scheme_residuals(q, h, iw; edge_src, edge_tgt, wl, dt, scheme) -> (; h_pred, residual)

Residual of one mass-balance discretisation (see [`MB_SCHEMES`](@ref)) on node-order
fields `q`, `h`, `iw` of shape `(n_nodes, T)`. `h_pred[:, t]` is the water depth
predicted for step `t+1` from step `t` and floored at zero (as the mass-balance
layer does), `residual = h[:, t+1] − h_pred[:, t]`; both are `(n_nodes, T-1)`.
`wl` is the channel area `width·length` per node; nodes with a non-finite or
non-positive `wl` give `NaN`.

Unlike [`mb_residual_diagnostics`](@ref) (θ-weighted, un-floored, for predictions),
this reproduces wflow's three candidate update rules on the truth data.
"""
function mb_scheme_residuals(q::AbstractMatrix, h::AbstractMatrix, iw::AbstractMatrix;
                             edge_src, edge_tgt, wl::AbstractVector, dt::Real,
                             scheme::Symbol)
    scheme in MB_SCHEMES ||
        throw(ArgumentError("unknown scheme $(repr(scheme)); expected one of $MB_SCHEMES"))
    n, T = size(q)
    (size(h) == (n, T) && size(iw) == (n, T) && length(wl) == n) ||
        throw(ArgumentError("q, h, iw must share the shape (n_nodes, T) and wl have length n_nodes"))
    T >= 2 || throw(ArgumentError("need at least two time steps"))

    Q = Float32.(q)
    h_pred   = Matrix{Float32}(undef, n, T - 1)
    residual = Matrix{Float32}(undef, n, T - 1)
    for t in 1:(T - 1)
        uq = upstream_sum(@view(Q[:, scheme === :lagged_upstream ? t : t + 1]), edge_src, edge_tgt)
        for i in 1:n
            w = wl[i]
            if isfinite(w) && w > 0
                coeff = Float32(dt / w)
                iw_i  = scheme === :fully_implicit ? iw[i, t + 1] : iw[i, t]
                dh    = coeff * (uq[i] + iw_i - Q[i, t + 1])
                h_pred[i, t]   = max(0f0, h[i, t] + dh)
                residual[i, t] = h[i, t + 1] - h_pred[i, t]
            else
                h_pred[i, t]   = NaN32
                residual[i, t] = NaN32
            end
        end
    end
    return (; h_pred, residual)
end

"""
    residual_summary(residual; ref = nothing) -> NamedTuple

Scalar statistics of the finite entries of `residual`: `n_finite`, `n`, `mean`,
`std`, `median`, `p5`, `p95`, `max_abs`, `mae`, `rmse`, `frac_exact` (share of
`|r| < 1e-6`) and, when `ref` (same units) is given, `median_abs_ref`.
`nothing` when no finite value exists.
"""
function residual_summary(residual::AbstractArray; ref = nothing)
    v = filter(isfinite, vec(residual))
    isempty(v) && return nothing
    med_ref = if ref === nothing
        nothing
    else
        fr = filter(isfinite, vec(ref))
        isempty(fr) ? nothing : median(abs.(fr))
    end
    return (; n_finite = length(v), n = length(residual),
            mean = mean(v), std = length(v) > 1 ? std(v) : NaN,
            median = median(v), p5 = quantile(v, 0.05), p95 = quantile(v, 0.95),
            max_abs = maximum(abs, v), mae = mean(abs, v), rmse = sqrt(mean(abs2, v)),
            frac_exact = count(x -> abs(x) < 1f-6, v) / length(v),
            median_abs_ref = med_ref)
end

"""
    graph_topology_checks(edge_src, edge_tgt, n_nodes) -> NamedTuple

Sanity checks of the routing graph and of [`upstream_sum`](@ref): node/edge counts,
`headwaters` (no upstream neighbour), `outlets` (no downstream neighbour), `isolated`
nodes, whether `upstream_sum` is exactly zero on headwaters (`headwater_ok`) and
equals the neighbour's value on single-upstream nodes (`n_single_upstream`,
`single_upstream_mismatches`).
"""
function graph_topology_checks(edge_src, edge_tgt, n_nodes::Integer)
    in_degree  = zeros(Int, n_nodes)
    out_degree = zeros(Int, n_nodes)
    for k in eachindex(edge_src)
        in_degree[edge_tgt[k]]  += 1
        out_degree[edge_src[k]] += 1
    end
    headwaters = findall(==(0), in_degree)
    outlets    = findall(==(0), out_degree)
    isolated   = findall(i -> in_degree[i] == 0 && out_degree[i] == 0, 1:n_nodes)

    headwater_ok = all(upstream_sum(ones(Float32, n_nodes), edge_src, edge_tgt)[headwaters] .== 0f0)

    one_up = findall(==(1), in_degree)
    mismatches = 0
    if !isempty(one_up)
        upstream_of = [Int[] for _ in 1:n_nodes]
        for k in eachindex(edge_src)
            push!(upstream_of[edge_tgt[k]], edge_src[k])
        end
        rng_q  = Float32.(1:n_nodes)
        uq_rng = upstream_sum(rng_q, edge_src, edge_tgt)
        mismatches = sum(uq_rng[i] != rng_q[upstream_of[i][1]] for i in one_up)
    end
    return (; n_nodes = Int(n_nodes), n_edges = length(edge_src),
            headwaters, outlets, isolated, out_degree, headwater_ok,
            n_single_upstream = length(one_up), single_upstream_mismatches = mismatches)
end

"""
    river_mask_consistency(staticmaps_file, layout) -> NamedTuple

Compare the routing graph of `layout` ([`river_layout`](@ref)) with `river_mask` and
`local_drain_direction`: number of mask cells vs. nodes, where each node's LDD
points (`ldd_to_river`, `ldd_to_nonriver`, `ldd_is_pit`, `ldd_missing`), the edge
count vs. `ldd_to_river`, river cells without an 8-connected river neighbour and
outlets whose LDD targets a river cell but have no edge (`missed_edges`, a bug
when non-empty).
"""
function river_mask_consistency(staticmaps_file::AbstractString, layout)
    (; n_nodes, rows, cols, edge_src) = layout
    out_degree = zeros(Int, n_nodes)
    for s in edge_src
        out_degree[s] += 1
    end

    NCDataset(staticmaps_file, "r") do ds
        mask   = ds["river_mask"][:, :]
        ldd    = ds["local_drain_direction"][:, :]
        nr, nc = size(mask)

        is_river(r, c) = 1 <= r <= nr && 1 <= c <= nc &&
                         !ismissing(mask[r, c]) && mask[r, c] != 0

        mask_count = count(is_river(r, c) for c in 1:nc, r in 1:nr)

        ldd_to_river = ldd_to_nonriver = ldd_is_pit = ldd_missing = 0
        for i in 1:n_nodes
            r0, c0 = rows[i], cols[i]
            val = ldd[r0, c0]
            if ismissing(val)
                ldd_missing += 1
                continue
            end
            v = Int(val)
            if v == 5
                ldd_is_pit += 1
                continue
            end
            offset = get(LDD_OFFSETS, v, nothing)
            if isnothing(offset)
                ldd_missing += 1
                continue
            end
            is_river(r0 + offset[1], c0 + offset[2]) ? (ldd_to_river += 1) : (ldd_to_nonriver += 1)
        end

        not_adjacent = count(1:n_nodes) do i
            r0, c0 = rows[i], cols[i]
            !any(is_river(r0 + dr, c0 + dc)
                 for dr in -1:1, dc in -1:1 if !(dr == 0 && dc == 0))
        end

        missed_edges = Int[]
        for i in 1:n_nodes
            out_degree[i] == 0 || continue
            r0, c0 = rows[i], cols[i]
            val = ldd[r0, c0]
            (ismissing(val) || Int(val) == 5) && continue
            offset = get(LDD_OFFSETS, Int(val), nothing)
            isnothing(offset) && continue
            is_river(r0 + offset[1], c0 + offset[2]) && push!(missed_edges, i)
        end

        return (; mask_count, n_nodes, ldd_to_river, ldd_to_nonriver, ldd_is_pit, ldd_missing,
                n_edges = length(edge_src), not_adjacent, missed_edges)
    end
end

"Variables of wflow's `output.nc` read by [`wflow_output_consistency`](@ref)."
const WFLOW_CONSISTENCY_VARS = (
    "river_q",                    "river_h",                "river_inwater",
    "river_flow_q_av",            "river_flow_h",           "river_bc_inwater",
    "river_bc_external_inflow",   "river_bc_abstraction",   "river_bc_external_abst_av",
    "water_balance_storage_prev", "river_flow_storage",     "river_flow_qin_av",
    "river_flow_q_lat",
)

"""
    wflow_output_consistency(vars, layout) -> Vector{NamedTuple}

Consistency checks between wflow's logged variables. `vars` maps each name of
[`WFLOW_CONSISTENCY_VARS`](@ref) to an `(n_nodes, ntimes)` matrix from
[`read_river_series`](@ref) (or `nothing` when absent); `layout` is from
[`river_layout`](@ref). Returns one `(; id, description, residual, ref)` per check
(`residual === nothing` when a required variable is missing), in the order

1. `river_flow_q_av ≈ river_q`,  2. `river_flow_h ≈ river_h`,
3. `river_bc_inwater ≈ river_inwater`,  4./5./6. `river_bc_external_inflow`,
`river_bc_abstraction`, `river_bc_external_abst_av` `≈ 0`,
7. `water_balance_storage_prev[t] ≈ river_flow_storage[t-1]`,
8. `river_flow_qin_av[t] ≈ Σ river_flow_q_av[t-1]` (upstream),
9. `river_flow_qin_av[t] ≈ Σ river_flow_q_av[t]` (upstream, same `t`; wflow's convention),
12. `river_flow_storage ≈ river_flow_h·width·length`.
"""
function wflow_output_consistency(vars::AbstractDict, layout)
    v = vars
    both(a, b) = !isnothing(v[a]) && !isnothing(v[b])
    checks = NamedTuple[]
    add(id, description, residual; ref = nothing) =
        push!(checks, (; id, description, residual, ref))

    add(1, "river_flow_q_av  ≈  river_q",
        both("river_flow_q_av", "river_q") ? v["river_flow_q_av"] .- v["river_q"] : nothing;
        ref = v["river_q"])
    add(2, "river_flow_h  ≈  river_h",
        both("river_flow_h", "river_h") ? v["river_flow_h"] .- v["river_h"] : nothing;
        ref = v["river_h"])
    add(3, "river_bc_inwater  ≈  river_inwater",
        both("river_bc_inwater", "river_inwater") ? v["river_bc_inwater"] .- v["river_inwater"] : nothing;
        ref = v["river_inwater"])
    add(4, "river_bc_external_inflow  ≈  0",   v["river_bc_external_inflow"])
    add(5, "river_bc_abstraction  ≈  0",       v["river_bc_abstraction"])
    add(6, "river_bc_external_abst_av  ≈  0",  v["river_bc_external_abst_av"])

    sp, fs = v["water_balance_storage_prev"], v["river_flow_storage"]
    add(7, "water_balance_storage_prev[t]  ≈  river_flow_storage[t-1]",
        (!isnothing(sp) && !isnothing(fs) && size(sp, 2) >= 2 && size(fs, 2) >= 2) ?
            sp[:, 2:end] .- fs[:, 1:end-1] : nothing;
        ref = fs)

    q_av, qin = v["river_flow_q_av"], v["river_flow_qin_av"]
    es, et = layout.edge_src, layout.edge_tgt
    if !isnothing(q_av) && !isnothing(qin)
        T = min(size(q_av, 2), size(qin, 2))
        r8 = Matrix{Float32}(undef, layout.n_nodes, T - 1)
        for t in 2:T
            r8[:, t - 1] = qin[:, t] .- upstream_sum(q_av[:, t - 1], es, et)
        end
        r9 = Matrix{Float32}(undef, layout.n_nodes, T)
        for t in 1:T
            r9[:, t] = qin[:, t] .- upstream_sum(q_av[:, t], es, et)
        end
        add(8, "river_flow_qin_av[t]  ≈  Σ q_av[t-1] (upstream)", r8; ref = q_av)
        add(9, "river_flow_qin_av[t]  ≈  Σ q_av[t]   (upstream, same t)", r9; ref = q_av)
    else
        add(8, "river_flow_qin_av[t]  ≈  Σ q_av[t-1] (upstream)", nothing; ref = q_av)
        add(9, "river_flow_qin_av[t]  ≈  Σ q_av[t]   (upstream, same t)", nothing; ref = q_av)
    end

    add(12, "river_flow_storage  ≈  river_flow_h × width × length",
        both("river_flow_storage", "river_flow_h") ?
            v["river_flow_h"] .* (layout.width .* layout.length_m) .- v["river_flow_storage"] : nothing;
        ref = fs)
    return checks
end
