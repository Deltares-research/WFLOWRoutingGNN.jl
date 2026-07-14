#!/usr/bin/env julia
# scripts/check_mass_balance_details.jl
#
# Verify data consistency properties in wflow output.nc:
#
#  1. river_flow_q_av          ≈ river_q
#  2. river_flow_h             ≈ river_h
#  3. river_bc_inwater         ≈ river_inwater
#  4. river_bc_external_inflow ≈ 0
#  5. river_bc_abstraction     ≈ 0
#  6. river_bc_external_abst_av≈ 0
#  7. water_balance_storage_prev[t] ≈ river_flow_storage[t-1]
#  8. river_flow_qin_av[t]     ≈ Σ river_flow_q_av[t-1] over inflowing cells
#  9. river_flow_storage       ≈ river_flow_h × river_width × river_length
#
# Usage:
#   julia --project=<repo_root> scripts/check_mass_balance_details.jl \
#       <staticmaps.nc> <output.nc>

if length(ARGS) < 2
    println(stderr,
        "Usage: julia --project=<repo_root> scripts/check_mass_balance_details.jl " *
        "<staticmaps.nc> <output.nc>")
    exit(1)
end

staticmaps_file = ARGS[1]
output_file     = ARGS[2]

isfile(staticmaps_file) || (println(stderr, "Not found: $staticmaps_file"); exit(1))
isfile(output_file)     || (println(stderr, "Not found: $output_file");     exit(1))

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using NCDatasets
using Statistics
using Printf
using WflowRoutingGNN

coerce(x) = ismissing(x) ? NaN32 : Float32(x)

# ── 1. Build compact river graph from staticmaps ─────────────────────────────
println("Reading staticmaps: $staticmaps_file")

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
sources   = [id_to_idx[s] for s in src_raw]
targets   = [id_to_idx[t] for t in tgt_raw]

# width and length needed for storage check (9)
width, length_m = NCDataset(staticmaps_file, "r") do ds
    n = length(rows)
    w = [coerce(ds["river_width"][rows[i],  cols[i]]) for i in 1:n]
    l = [coerce(ds["river_length"][rows[i], cols[i]]) for i in 1:n]
    w, l
end

n_nodes   = length(rows)
dim1_flip = alignment.dim1_flip
dim2_flip = alignment.dim2_flip

println("  River cells : $n_nodes")
println("  River edges : $(length(sources))")
println("  Axis flip   : dim1=$dim1_flip, dim2=$dim2_flip")

# ── 2. Read output variables ──────────────────────────────────────────────────
println("\nReading output: $output_file")

# Extract river node time series from a 3D (lon × lat × time) output variable.
# Returns Matrix{Float32}(n_nodes, ntimes), or nothing if absent or not 3-D.
function extract_var(ds, varname)
    varname in keys(ds) || return nothing
    v = ds[varname]
    if ndims(v) < 3
        @warn "  $varname has fewer than 3 dimensions — skipping"
        return nothing
    end
    data3d = coerce.(v[:, :, :])
    nt     = size(data3d, 3)
    mat    = Matrix{Float32}(undef, n_nodes, nt)
    for (i, (r, c)) in enumerate(zip(rows, cols))
        r_out = dim1_flip ? nrows - r + 1 : r
        c_out = dim2_flip ? ncols - c + 1 : c
        mat[i, :] = data3d[r_out, c_out, :]
    end
    mat
end

ALL_VARS = [
    "river_q",                    "river_h",                "river_inwater",
    "river_flow_q_av",            "river_flow_h",           "river_bc_inwater",
    "river_bc_external_inflow",   "river_bc_abstraction",   "river_bc_external_abst_av",
    "water_balance_storage_prev", "river_flow_storage",     "river_flow_qin_av",
    "river_flow_q_lat",
]

vars = NCDataset(output_file, "r") do ds
    println("  Variable presence:")
    for vn in ALL_VARS
        status = vn in keys(ds) ? "✓ found" : "✗ NOT FOUND"
        @printf("    %-40s %s\n", vn, status)
    end
    Dict(vn => extract_var(ds, vn) for vn in ALL_VARS)
end

# ── 3. Helpers ────────────────────────────────────────────────────────────────

# Upstream-sum aggregation: for each node i, sum q[j] for all j that flow into i
# Edge convention from ldd_to_graph: sources[k] drains INTO targets[k]
# So q[sources[k]] is upstream contribution to targets[k]. ✓
function upstream_sum(q_col::AbstractVector{Float32})
    uq = zeros(Float32, n_nodes)
    for k in eachindex(sources)
        uq[targets[k]] += q_col[sources[k]]
    end
    uq
end

# ── 3b. Verify upstream_sum topology ─────────────────────────────────────────
let
    in_degree  = zeros(Int, n_nodes)
    out_degree = zeros(Int, n_nodes)
    for k in eachindex(sources)
        in_degree[targets[k]]  += 1
        out_degree[sources[k]] += 1
    end
    headwaters = findall(==(0), in_degree)   # no upstream neighbours
    outlets    = findall(==(0), out_degree)  # no downstream neighbour (pit/outlet)
    isolated   = findall(i -> in_degree[i] == 0 && out_degree[i] == 0, 1:n_nodes)
    println("\nGraph topology check:")
    @printf("  nodes=%d  edges=%d  headwaters=%d  outlets=%d  isolated=%d\n",
            n_nodes, length(sources), length(headwaters), length(outlets), length(isolated))

    # For headwaters: upstream_sum must be exactly 0
    q_test = ones(Float32, n_nodes)
    uq_test = upstream_sum(q_test)
    hw_ok = all(uq_test[headwaters] .== 0f0)
    @printf("  Headwater upstream_sum == 0 : %s\n", hw_ok ? "✓ PASS" : "✗ FAIL")

    # For nodes with exactly 1 upstream neighbour: upstream_sum == q[that neighbour]
    one_up = findall(==(1), in_degree)
    if !isempty(one_up)
        upstream_of = [Int[] for _ in 1:n_nodes]
        for k in eachindex(sources)
            push!(upstream_of[targets[k]], sources[k])
        end
        rng_q = Float32.(1:n_nodes)
        uq_rng = upstream_sum(rng_q)
        mismatches = sum(uq_rng[i] != rng_q[upstream_of[i][1]] for i in one_up)
        @printf("  Single-upstream nodes=%d  upstream_sum==q[upstream] : %s\n",
                length(one_up), mismatches == 0 ? "✓ PASS" : "✗ FAIL ($mismatches mismatches)")
    end

    # ── Consistency with river_mask ──────────────────────────────────────────
    println("\nRiver mask consistency check:")
    NCDataset(staticmaps_file, "r") do ds
        mask   = ds["river_mask"][:, :]
        ldd    = ds["local_drain_direction"][:, :]
        nr, nc = size(mask)

        is_river(r, c) = 1 <= r <= nr && 1 <= c <= nc &&
                         !ismissing(mask[r, c]) && mask[r, c] != 0

        # 1. Node count from mask
        mask_count = count(is_river(r, c) for c in 1:nc, r in 1:nr)
        @printf("  river_mask non-zero cells : %d  (graph nodes: %d)  %s\n",
                mask_count, n_nodes, mask_count == n_nodes ? "✓ match" : "✗ MISMATCH")

        # 2. For every river node, check where its LDD points
        ldd_to_river    = 0   # LDD immediate neighbour is a river cell   ← should produce an edge
        ldd_to_nonriver = 0   # LDD neighbour is a non-river cell
        ldd_is_pit      = 0   # LDD == 5 (pit / outlet)
        ldd_missing     = 0   # LDD value is missing/unknown

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
            rn, cn = r0 + offset[1], c0 + offset[2]
            if is_river(rn, cn)
                ldd_to_river += 1
            else
                ldd_to_nonriver += 1
            end
        end

        @printf("  LDD → river cell (should be an edge) : %d\n", ldd_to_river)
        @printf("  LDD → non-river cell (no edge added)  : %d\n", ldd_to_nonriver)
        @printf("  LDD == 5 (pit/outlet)                 : %d\n", ldd_is_pit)
        @printf("  LDD missing/unknown                   : %d\n", ldd_missing)
        @printf("  sum                                   : %d  (should == %d)\n",
                ldd_to_river + ldd_to_nonriver + ldd_is_pit + ldd_missing, n_nodes)

        # Expected edges == ldd_to_river; compare with actual
        @printf("  graph edges vs. LDD→river count       : %d vs %d  %s\n",
                length(sources), ldd_to_river,
                length(sources) == ldd_to_river ? "✓ match" : "✗ MISMATCH")

        # 3. Spatial adjacency: every river cell should have ≥1 river neighbour (8-connected)
        not_adjacent = count(1:n_nodes) do i
            r0, c0 = rows[i], cols[i]
            !any(is_river(r0 + dr, c0 + dc)
                 for dr in -1:1, dc in -1:1 if !(dr == 0 && dc == 0))
        end
        @printf("  river cells with NO adjacent river neighbour (8-conn): %d  %s\n",
                not_adjacent, not_adjacent == 0 ? "✓ connected" : "✗ isolated pixels present")

        # 4. For "off-graph" outlets: what cell does their LDD actually point to?
        off_graph_sample = Int[]
        for i in 1:n_nodes
            out_degree[i] == 0 || continue
            r0, c0 = rows[i], cols[i]
            val = ldd[r0, c0]
            (ismissing(val) || Int(val) == 5) && continue
            offset = get(LDD_OFFSETS, Int(val), nothing)
            isnothing(offset) && continue
            rn, cn = r0 + offset[1], c0 + offset[2]
            is_river(rn, cn) && push!(off_graph_sample, i)   # LDD points to river but no edge!
        end
        @printf("  outlets whose LDD targets a river cell but have no edge: %d  %s\n",
                length(off_graph_sample),
                isempty(off_graph_sample) ? "✓ none" : "✗ BUG: edge was missed")
        if !isempty(off_graph_sample)
            i = off_graph_sample[1]
            r0, c0 = rows[i], cols[i]
            val    = Int(ldd[r0, c0])
            offset = LDD_OFFSETS[val]
            rn, cn = r0 + offset[1], c0 + offset[2]
            # Is (rn,cn) in our id_to_idx?
            tgt_id = (cn - 1) * nr + rn
            @printf("    example: node %d at (%d,%d) LDD=%d → (%d,%d) tgt_id=%d in_id_to_idx=%s\n",
                    i, r0, c0, val, rn, cn, tgt_id,
                    haskey(id_to_idx, tgt_id) ? "yes" : "NO ← missing from compact index")
        end
    end
end

# Print residual statistics for one check
function print_check(check_id, description, residual; ref=nothing)
    prefix = @sprintf("  [%d] %s", check_id, description)
    if isnothing(residual)
        @printf("%-70s  SKIP (variable(s) missing)\n", prefix)
        return
    end
    finite_r = filter(isfinite, vec(residual))
    if isempty(finite_r)
        @printf("%-70s  SKIP (no finite values)\n", prefix)
        return
    end
    mae  = mean(abs, finite_r)
    rmse = sqrt(mean(x -> x^2, finite_r))
    mxae = maximum(abs, finite_r)
    frac = count(x -> abs(x) < 1f-6, finite_r) / length(finite_r)

    ref_str = ""
    if !isnothing(ref)
        fr = filter(isfinite, vec(ref))
        isempty(fr) || (ref_str = @sprintf("  (median|ref|=%.3g)", median(abs.(fr))))
    end

    @printf("%-70s  MAE=%.3g  RMSE=%.3g  max=%.3g  exact(≤1e-6)=%.1f%%%s\n",
            prefix, mae, rmse, mxae, 100*frac, ref_str)
end

# ── 4. Checks ─────────────────────────────────────────────────────────────────
println("\n" * "─"^110)
println("Check results:")
println("─"^110)

v = vars   # shorthand

# 1. river_flow_q_av ≈ river_q
res1 = (!isnothing(v["river_flow_q_av"]) && !isnothing(v["river_q"])) ?
       v["river_flow_q_av"] .- v["river_q"] : nothing
print_check(1, "river_flow_q_av  ≈  river_q", res1; ref=v["river_q"])

# 2. river_flow_h ≈ river_h
res2 = (!isnothing(v["river_flow_h"]) && !isnothing(v["river_h"])) ?
       v["river_flow_h"] .- v["river_h"] : nothing
print_check(2, "river_flow_h  ≈  river_h", res2; ref=v["river_h"])

# 3. river_bc_inwater ≈ river_inwater
res3 = (!isnothing(v["river_bc_inwater"]) && !isnothing(v["river_inwater"])) ?
       v["river_bc_inwater"] .- v["river_inwater"] : nothing
print_check(3, "river_bc_inwater  ≈  river_inwater", res3; ref=v["river_inwater"])

# 4. river_bc_external_inflow ≈ 0
print_check(4, "river_bc_external_inflow  ≈  0", v["river_bc_external_inflow"])

# 5. river_bc_abstraction ≈ 0
print_check(5, "river_bc_abstraction  ≈  0", v["river_bc_abstraction"])

# 6. river_bc_external_abst_av ≈ 0
print_check(6, "river_bc_external_abst_av  ≈  0", v["river_bc_external_abst_av"])

# 7. water_balance_storage_prev[t] ≈ river_flow_storage[t-1]
res7 = let sp = v["water_balance_storage_prev"], fs = v["river_flow_storage"]
    (!isnothing(sp) && !isnothing(fs) && size(sp, 2) >= 2 && size(fs, 2) >= 2) ?
        sp[:, 2:end] .- fs[:, 1:end-1] : nothing
end
print_check(7, "water_balance_storage_prev[t]  ≈  river_flow_storage[t-1]", res7;
            ref=v["river_flow_storage"])

# 8. river_flow_qin_av[t] ≈ Σ river_flow_q_av[t-1] (upstream)
res8a = let q_av = v["river_flow_q_av"], qin = v["river_flow_qin_av"]
    if !isnothing(q_av) && !isnothing(qin)
        T   = min(size(q_av, 2), size(qin, 2))
        mat = Matrix{Float32}(undef, n_nodes, T - 1)
        for t in 2:T
            mat[:, t-1] = qin[:, t] .- upstream_sum(q_av[:, t-1])
        end
        mat
    else
        nothing
    end
end
print_check(8, "river_flow_qin_av[t]  ≈  Σ q_av[t-1] (upstream)", res8a;
            ref=v["river_flow_q_av"])

# 9. river_flow_qin_av[t] ≈ Σ river_flow_q_av[t] (upstream, same t)
res8b = let q_av = v["river_flow_q_av"], qin = v["river_flow_qin_av"]
    if !isnothing(q_av) && !isnothing(qin)
        T   = min(size(q_av, 2), size(qin, 2))
        mat = Matrix{Float32}(undef, n_nodes, T)
        for t in 1:T
            mat[:, t] = qin[:, t] .- upstream_sum(q_av[:, t])
        end
        mat
    else
        nothing
    end
end
print_check(9, "river_flow_qin_av[t]  ≈  Σ q_av[t]   (upstream, same t)", res8b;
            ref=v["river_flow_q_av"])

# 12. river_flow_storage ≈ river_flow_h × river_width × river_length
res9 = let fs = v["river_flow_storage"], fh = v["river_flow_h"]
    if !isnothing(fs) && !isnothing(fh)
        # width and length are per-node; broadcast over time
        wl = width .* length_m          # (n_nodes,)
        fh .* wl .- fs                  # (n_nodes, ntimes)
    else
        nothing
    end
end
print_check(12, "river_flow_storage  ≈  river_flow_h × width × length", res9;
            ref=v["river_flow_storage"])

println("─"^110)
println("Done.")
