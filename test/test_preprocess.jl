using Test
using NCDatasets
using MLUtils
using GraphNeuralNetworks
using Statistics

const STATICMAPS = joinpath(@__DIR__, "..", "test_data", "test_model", "staticmaps.nc")
const OUTPUT_NC  = joinpath(@__DIR__, "..", "test_data", "test_model", "run_default", "output.nc")

# const LDD_OFFSETS = Dict(
#     1 => (+1, -1),  # Southwest
#     2 => (+1,  0),  # South
#     3 => (+1, +1),  # Southeast
#     4 => ( 0, -1),  # West
#     6 => ( 0, +1),  # East
#     7 => (-1, -1),  # Northwest
#     8 => (-1,  0),  # North
#     9 => (-1, +1),  # Northeast
#     # 5 = pit/outlet — no outflow, excluded
# )

# ── Pre-compute spot-check values in a single pass ─────────────────────────
# Find the first valid cell (non-pit, active target) for the spot check.
spot_src = 0
spot_tgt = 0

NCDataset(STATICMAPS, "r") do ds
    ldd_raw = ds["local_drain_direction"][:, :]
    nrows, ncols = size(ldd_raw)

    for col in 1:ncols, row in 1:nrows
        spot_src == 0 || break
        v = ldd_raw[row, col]
        ismissing(v)                && continue
        Int(v) == 5                 && continue
        haskey(LDD_OFFSETS, Int(v)) || continue
        Δrow, Δcol = LDD_OFFSETS[Int(v)]
        nr, nc = row + Δrow, col + Δcol
        (1 ≤ nr ≤ nrows && 1 ≤ nc ≤ ncols) || continue
        ismissing(ldd_raw[nr, nc])  && continue  # target must be active
        global spot_src = (col - 1) * nrows + row
        global spot_tgt = (nc  - 1) * nrows + nr
    end
end

@testset "ldd_to_graph" begin

    # ── Test 1: invalid domain throws ArgumentError ──────────────────────────
    @testset "invalid domain throws ArgumentError" begin
        @test_throws ArgumentError ldd_to_graph(STATICMAPS, "sea")
        @test_throws ArgumentError ldd_to_graph(STATICMAPS, "Land")  # case-sensitive
        @test_throws ArgumentError ldd_to_graph(STATICMAPS, "")
    end

    src_land,  tgt_land  = ldd_to_graph(STATICMAPS, "land")
    src_river, tgt_river = ldd_to_graph(STATICMAPS, "river")

    # ── Test 2: sources and targets have equal length ────────────────────────
    @testset "sources and targets equal length" begin
        @test length(src_land)  == length(tgt_land)
        @test length(src_river) == length(tgt_river)
    end

    # ── Test 3: all land graph nodes correspond to active cells ────────────
    # After filtering, no nodata cell should appear as source or target.
    @testset "land: all graph nodes are active (non-nodata) cells" begin
        NCDataset(STATICMAPS, "r") do ds
            ldd_flat = vec(ds["local_drain_direction"][:, :])
            all_nodes = unique(vcat(src_land, tgt_land))
            @test all(i -> !ismissing(ldd_flat[i]), all_nodes)
        end
    end

    # ── Test 4: all river graph nodes lie within the river mask ─────────────
    # After filtering, no non-river cell should appear as source or target.
    @testset "river: all graph nodes are within river_mask" begin
        NCDataset(STATICMAPS, "r") do ds
            river_flat = vec(ds["river_mask"][:, :])
            all_nodes = unique(vcat(src_river, tgt_river))
            @test all(i -> !ismissing(river_flat[i]) && river_flat[i] != 0, all_nodes)
        end
    end

    # ── Test 5: spot check ───────────────────────────────────────────────────
    @testset "spot check: first valid cell has correct source→target edge" begin
        @test spot_src != 0  # sanity: at least one valid cell must exist
        idx = findfirst(==(spot_src), src_land)
        @test idx !== nothing
        @test tgt_land[idx] == spot_tgt
    end

end

#########################
# Tests build_wflow_graph
#########################

# Pre-compute reference node mapping from river_mask (all active cells,
# including pit/outlet cells that carry no outgoing edges).
let
    global ref_node_ids = NCDataset(STATICMAPS, "r") do ds
        mask = ds["river_mask"][:, :]
        nr, nc = size(mask)
        sort([((c - 1) * nr + r)
              for c in 1:nc, r in 1:nr
              if !ismissing(mask[r, c]) && mask[r, c] != 0])
    end
end
const REF_N_NODES = length(ref_node_ids)

const REF_NROWS = NCDataset(STATICMAPS, "r") do ds
    size(ds["local_drain_direction"], 1)
end

const REF_ROWS = [(id - 1) % REF_NROWS + 1 for id in ref_node_ids]
const REF_COLS = [(id - 1) ÷ REF_NROWS + 1 for id in ref_node_ids]

const REF_NTIMES = NCDataset(OUTPUT_NC, "r") do ds
    size(ds["river_q"], 3)
end

# Reproduce the standardize! logic from gnn.jl for reference computations.
function _ref_stats(vals::Vector{Float32})
    v = filter(!isnan, vals)
    u = isempty(v) ? 0f0 : mean(v)
    s = (isempty(v) || length(v) == 1) ? 1f0 : std(v)
    s = s == 0f0 ? 1f0 : s
    u, s
end

# Extract pre-processed Float32 node values, applying the same per-variable
# transforms that build_wflow_graph applies before z-score normalisation.
# Uses VAR_SCALERS directly so the test always matches production code.
function _raw_nodes(ncfile::String, vname::String)
    NCDataset(ncfile, "r") do ds
        data = ds[vname][ntuple(_ -> Colon(), ndims(ds[vname]))...]
        coerce(x) = ismissing(x) ? NaN32 : Float32(x)
        if ndims(data) == 2
            # Static vars from staticmaps.nc — no axis flip, no extra transform
            [coerce(data[REF_ROWS[i], REF_COLS[i]]) for i in 1:REF_N_NODES]
        else
            # Time-varying vars: build (n_nodes × n_times) matrix, apply scaler, flatten.
            n_t = size(data, 3)
            mat = [coerce(data[OUT_ROWS[i], OUT_COLS[i], t])
                   for i in 1:REF_N_NODES, t in 1:n_t]
            domain_scalers = get(VAR_SCALERS, "river", Dict{String,Function}())
            if haskey(domain_scalers, vname)
                domain_scalers[vname](mat, REF_ROWS, REF_COLS, STATICMAPS)
            end
            vec(mat)
        end
    end
end

const BG_GRAPHS, BG_STATS, BG_GRID, BG_POSTSCALE = build_wflow_graph(STATICMAPS, OUTPUT_NC, "river")

const REF_NCOLS = NCDataset(STATICMAPS, "r") do ds
    size(ds["local_drain_direction"], 2)
end

# Compute the output-file indices for every node, applying any axis flips
# detected by check_and_correct_grid_alignment (dim2/latitude is reversed).
const BG_ALIGNMENT = check_and_correct_grid_alignment(STATICMAPS, OUTPUT_NC, "river")
const OUT_ROWS = [BG_ALIGNMENT.dim1_flip ? REF_NROWS - r + 1 : r for r in REF_ROWS]
const OUT_COLS = [BG_ALIGNMENT.dim2_flip ? REF_NCOLS - c + 1 : c for c in REF_COLS]

const STATE_VARS   = DOMAIN_VARS["river"]["state"]
const FORCING_VARS = DOMAIN_VARS["river"]["forcing"]
const STATIC_VARS  = DOMAIN_VARS["river"]["static"]

@testset "build_wflow_graph" begin

    @testset "graphs array length equals ntimes" begin
        @test length(BG_GRAPHS) == REF_NTIMES
    end

    @testset "feature shapes" begin
        g = BG_GRAPHS[1]
        @test size(g.ndata.state)   == (length(STATE_VARS),   REF_N_NODES)
        @test size(g.ndata.forcing) == (length(FORCING_VARS), REF_N_NODES)
        @test size(g.ndata.static)  == (length(STATIC_VARS),  REF_N_NODES)
        @test BG_GRAPHS[1].ndata.static === BG_GRAPHS[end].ndata.static
    end

    @testset "normalization stats match raw-data reference" begin
        for vname in [STATE_VARS; FORCING_VARS]
            raw          = _raw_nodes(OUTPUT_NC, vname)
            u_ref, s_ref = _ref_stats(raw)
            @test BG_STATS[vname].mean == u_ref
            @test BG_STATS[vname].std  == s_ref
        end
        for vname in STATIC_VARS
            raw          = _raw_nodes(STATICMAPS, vname)
            u_ref, s_ref = _ref_stats(raw)
            @test BG_STATS[vname].mean == u_ref
            @test BG_STATS[vname].std  == s_ref
        end
    end

    @testset "spot check: standardized values match (raw - mean) / std" begin
        # Find the first (node, time) pair where river_q is non-missing.
        spot_i, spot_t, raw_q_val = NCDataset(OUTPUT_NC, "r") do ds
            data = ds["river_q"][:, :, :]
            result = nothing
            for i in 1:REF_N_NODES, t in 1:REF_NTIMES
                v = data[OUT_ROWS[i], OUT_COLS[i], t]
                if !ismissing(v)
                    result = (i, t, Float32(v))
                    break
                end
            end
            result
        end
        @test spot_i !== nothing  # sanity: at least one non-missing value must exist

        # Apply the same per-variable preprocessing as build_wflow_graph,
        # using VAR_SCALERS so the test automatically stays in sync with production.
        domain_scalers = get(VAR_SCALERS, "river", Dict{String,Function}())

        slice_q = reshape([raw_q_val], 1, 1)
        haskey(domain_scalers, "river_q") &&
            domain_scalers["river_q"](slice_q, [REF_ROWS[spot_i]], [REF_COLS[spot_i]], STATICMAPS)
        expected_q = (slice_q[1,1] - BG_STATS["river_q"].mean) / BG_STATS["river_q"].std
        @test BG_GRAPHS[spot_t].ndata.state[1, spot_i] ≈ expected_q  atol=1f-5

        raw_h = NCDataset(OUTPUT_NC, "r") do ds
            Float32(ds["river_h"][OUT_ROWS[spot_i], OUT_COLS[spot_i], spot_t])
        end
        slice_h = reshape([raw_h], 1, 1)
        haskey(domain_scalers, "river_h") &&
            domain_scalers["river_h"](slice_h, [REF_ROWS[spot_i]], [REF_COLS[spot_i]], STATICMAPS)
        expected_h = (slice_h[1,1] - BG_STATS["river_h"].mean) / BG_STATS["river_h"].std
        @test BG_GRAPHS[spot_t].ndata.state[2, spot_i] ≈ expected_h  atol=1f-5

        raw_len = NCDataset(STATICMAPS, "r") do ds
            Float32(ds["river_length"][REF_ROWS[spot_i], REF_COLS[spot_i]])
        end
        expected_len = (raw_len - BG_STATS["river_length"].mean) / BG_STATS["river_length"].std
        @test BG_GRAPHS[1].ndata.static[1, spot_i] ≈ expected_len  atol=1f-5
    end

    @testset "grid: dimensions and node positions" begin
        @test BG_GRID.nrows == REF_NROWS
        @test BG_GRID.ncols == REF_NCOLS
        @test length(BG_GRID.rows) == REF_N_NODES
        @test length(BG_GRID.cols) == REF_N_NODES
        # Positions must match the reference computed from node_ids
        @test BG_GRID.rows == REF_ROWS
        @test BG_GRID.cols == REF_COLS
    end

    @testset "grid alignment: staticmaps and output are aligned (with flip detection)" begin
        alignment = check_and_correct_grid_alignment(STATICMAPS, OUTPUT_NC, "river")
        @test alignment isa NamedTuple
        # longitude axis: both ascending → no flip
        @test alignment.dim1_flip == false
        # latitude axis: staticmaps descending, output ascending → flipped
        @test alignment.dim2_flip == true
    end

    @testset "node count equals number of nonzero river_mask cells" begin
        n_river_cells = NCDataset(STATICMAPS, "r") do ds
            mask = ds["river_mask"][:, :]
            count(x -> !ismissing(x) && x != 0, mask)
        end
        @test BG_GRAPHS[1].num_nodes == n_river_cells
    end

end

#########################
# Tests make_horizon_dataset
#########################

const MHD_NHORIZON = 2
const MHD_AT       = (0.6, 0.2)  # per-split fractions: 60% train, 20% val, 20% test
const MHD_NWINDOWS = REF_NTIMES - MHD_NHORIZON + 1

# Derive expected split sizes via the same splitobs call — robust to any internal rounding.
let splits = splitobs(1:MHD_NWINDOWS; at = MHD_AT)
    global MHD_N_TRAIN = length(splits[1])
    global MHD_N_VAL   = length(splits[2])
    global MHD_N_TEST  = length(splits[3])
end

const MHD = make_horizon_dataset(BG_GRAPHS, MHD_NHORIZON; at = MHD_AT)

@testset "make_horizon_dataset" begin

    @testset "total number of windows" begin
        @test MHD_N_TRAIN + MHD_N_VAL + MHD_N_TEST == MHD_NWINDOWS
        @test length(MHD.train) + length(MHD.val) + length(MHD.test) == MHD_NWINDOWS
    end

    @testset "split sizes" begin
        @test length(MHD.train) == MHD_N_TRAIN
        @test length(MHD.val)   == MHD_N_VAL
        @test length(MHD.test)  == MHD_N_TEST
    end

    @testset "each window has length nhorizon" begin
        @test all(w -> length(w) == MHD_NHORIZON, MHD.train)
        @test all(w -> length(w) == MHD_NHORIZON, MHD.val)
        @test all(w -> length(w) == MHD_NHORIZON, MHD.test)
    end

end

#########################
# Tests MLUtils.batch overload
#########################

const BATCH_SIZE   = max(1, MHD_N_TRAIN ÷ 3)
const TRAIN_LOADER = DataLoader(MHD.train; batchsize = BATCH_SIZE, shuffle = false, collate = true)

# Expected number of batches (last batch may be smaller)
const BATCH_N_BATCHES = ceil(Int, MHD_N_TRAIN / BATCH_SIZE)

@testset "MLUtils.batch collation" begin

    @testset "DataLoader produces expected number of batches" begin
        @test length(TRAIN_LOADER) == BATCH_N_BATCHES
    end

    @testset "each batch is a Vector{GNNGraph} of length nhorizon" begin
        @test all(b -> b isa Vector && length(b) == MHD_NHORIZON, TRAIN_LOADER)
    end

    @testset "each element of a batch is a GNNGraph" begin
        @test all(b -> all(g -> g isa GNNGraph, b), TRAIN_LOADER)
    end

    @testset "first batch has correct node count" begin
        # First full batch: BATCH_SIZE windows x REF_N_NODES nodes each
        first_batch = first(TRAIN_LOADER)
        expected_nodes = min(BATCH_SIZE, MHD_N_TRAIN) * REF_N_NODES
        @test first_batch[1].num_nodes == expected_nodes
    end

    @testset "total observations across all batches equals train set size" begin
        total = sum(b[1].num_nodes for b in TRAIN_LOADER)
        @test total == MHD_N_TRAIN * REF_N_NODES
    end

end

# ---------------------------------------------------------------------------
# DataSettings
# ---------------------------------------------------------------------------

const VALID_DS_KWARGS = (
    run_name         = "myrun",
    runs_dir         = tempdir(),
    wflow_model_path = tempdir(),
    train_frac       = 0.6,
    val_frac         = 0.2,
)

@testset "DataSettings constructor validation" begin

    @test DataSettings(; VALID_DS_KWARGS...) isa DataSettings

    @testset "empty run_name throws" begin
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., run_name = "")
    end

    @testset "non-positive train_frac throws" begin
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., train_frac = 0.0)
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., train_frac = -0.1)
    end

    @testset "non-positive val_frac throws" begin
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., val_frac = 0.0)
    end

    @testset "train_frac + val_frac >= 1 throws" begin
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., train_frac = 0.8, val_frac = 0.2)
        @test_throws ArgumentError DataSettings(; VALID_DS_KWARGS..., train_frac = 0.9, val_frac = 0.2)
    end

    @testset "fields are stored correctly" begin
        s = DataSettings(; VALID_DS_KWARGS...)
        @test s.run_name         == "myrun"
        @test s.runs_dir         == tempdir()
        @test s.wflow_model_path == tempdir()
        @test s.train_frac       == 0.6
        @test s.val_frac         == 0.2
    end

end

@testset "DataSettings TOML round-trip" begin

    s    = DataSettings(; VALID_DS_KWARGS...)
    path = tempname() * ".toml"
    save_data_settings(path, s)
    s2   = load_data_settings(path)
    rm(path)

    @test s2.run_name         == s.run_name
    @test s2.runs_dir         == s.runs_dir
    @test s2.wflow_model_path == s.wflow_model_path
    @test s2.train_frac       == s.train_frac
    @test s2.val_frac         == s.val_frac

end

@testset "DataSettings show" begin
    s   = DataSettings(; VALID_DS_KWARGS...)
    str = sprint(show, s)
    @test occursin("DataSettings", str)
    @test occursin("myrun",        str)
    @test occursin("0.6",          str)
    @test occursin("0.2",          str)
end