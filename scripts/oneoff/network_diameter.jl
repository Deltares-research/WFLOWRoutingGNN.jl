# Network diameter of the river DAG = longest downstream path length (in hops).
# This is the number of A-multiplications needed for FULL upstream accumulation
# (transitive closure) via repeated SpMV, and the receptive field a k-hop scheme
# would need to see the whole contributing network.
#
# Replicates the package's ldd_to_graph edge logic with NCDatasets only (fast load).

using NCDatasets, Printf

catch_dir = length(ARGS) >= 1 ? ARGS[1] : "models/sava_small_v081"
static_f  = joinpath(catch_dir, "staticmaps.nc")
ldd_var  = occursin("v081", catch_dir) ? "wflow_ldd"   : "local_drain_direction"
mask_var = occursin("v081", catch_dir) ? "wflow_river" : "river_mask"

# LDD numpad direction -> (Δrow, Δcol) in (lon,lat) raster; matches preprocess.jl LDD_OFFSETS
const OFF = Dict(1=>(-1,1),2=>(0,1),3=>(1,1),4=>(-1,0),6=>(1,0),
                 7=>(-1,-1),8=>(0,-1),9=>(1,-1))   # 5 = pit

ds = NCDataset(static_f, "r")
ldd  = ds[ldd_var][:, :]
mask = ds[mask_var][:, :]
close(ds)
nrows, ncols = size(ldd)
active(r,c) = (1<=r<=nrows)&&(1<=c<=ncols)&&!ismissing(mask[r,c])&&mask[r,c]!=0
lin(r,c) = (c-1)*nrows + r

# downstream map (each active river cell drains to exactly one downstream cell)
downstream = Dict{Int,Int}()
nodes = Int[]
for c in 1:ncols, r in 1:nrows
    active(r,c) || continue
    push!(nodes, lin(r,c))
    v = ldd[r,c]
    ismissing(v) && continue
    off = get(OFF, Int(v), nothing)   # 5/pit/unknown -> outlet
    off === nothing && continue
    r2, c2 = r+off[1], c+off[2]
    active(r2,c2) && (downstream[lin(r,c)] = lin(r2,c2))
end
nodes = unique(nodes)
@printf("river nodes = %d, edges = %d\n", length(nodes), length(downstream))

# longest downstream path, iterative + on-stack cycle guard
depth = Dict{Int,Int}()
function pathlen(start)
    chain = Int[]
    u = start
    while true
        haskey(depth, u) && break
        if u in chain           # cycle guard (should not happen for a true DAG)
            @warn "cycle detected at node $u"
            depth[u] = 0
            break
        end
        push!(chain, u)
        if !haskey(downstream, u)
            depth[u] = 0
            break
        end
        u = downstream[u]
    end
    base = depth[u]
    for i in length(chain):-1:1
        base += 1
        depth[chain[i]] = base
    end
    return depth[start]
end
for u in nodes; pathlen(u); end

lens = sort(collect(values(depth)); rev = true)
diam = lens[1]
@printf("\n=== river DAG geometry: %s ===\n", catch_dir)
@printf("river nodes                 : %d\n", length(nodes))
@printf("network diameter (max hops) : %d\n", diam)
@printf("mean downstream path length : %.1f hops\n", sum(lens)/length(lens))
@printf("median path length          : %d hops\n", lens[cld(length(lens),2)])
@printf("\n=> full upstream accumulation = %d sparse mat-vecs (or one triangular solve)\n", diam)
