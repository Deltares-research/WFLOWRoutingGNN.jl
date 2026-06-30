using Test
using Flux
using GraphNeuralNetworks
using MLUtils
using NCDatasets

# ---------------------------------------------------------------------------
# Dataset from real test model data
# ---------------------------------------------------------------------------

const TR_STATICMAPS = joinpath(@__DIR__, "..", "test_data", "test_model", "staticmaps.nc")
const TR_OUTPUT_NC  = joinpath(@__DIR__, "..", "test_data", "test_model", "run_default", "output.nc")

const TR_NHORIZON = 3   # must be >= max(strategy.steps) + 1 = 2 + 1
const TR_EPOCHS   = 4

const TR_GRAPHS, TR_STATS, TR_GRID, TR_POSTSCALE = build_wflow_graph(TR_STATICMAPS, TR_OUTPUT_NC, "river")

const TR_N_NODES   = TR_GRAPHS[1].num_nodes
const TR_N_STATE   = length(DOMAIN_VARS["river"]["state"])
const TR_N_FORCING = length(DOMAIN_VARS["river"]["forcing"])
const TR_N_STATIC  = length(DOMAIN_VARS["river"]["static"])

const TR_DATASET = make_horizon_dataset(TR_GRAPHS, TR_NHORIZON; at = (0.7, 0.15))

const TR_TRAIN_LOADER = DataLoader(TR_DATASET.train; batchsize = min(4, length(TR_DATASET.train)), shuffle = false, collate = true)
const TR_VAL_LOADER   = DataLoader(TR_DATASET.val;   batchsize = min(4, length(TR_DATASET.val)),   shuffle = false, collate = true)

const TR_MODEL = WflowGNN(ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1))

# ---------------------------------------------------------------------------
# TrainSettings constructor validation
# ---------------------------------------------------------------------------

const VALID_TS_KWARGS = (
    epochs     = TR_EPOCHS,
    batch_size = 4,
    lr_start   = 1f-3,
    lr_final   = 1f-5,
    lr_steps   = 2,
    strategy   = TrainingStrategy([1, 2], [2, 2]),
    device     = :cpu,
)

@testset "TrainSettings constructor validation" begin

    @test TrainSettings(; VALID_TS_KWARGS...) isa TrainSettings

    @testset "non-positive epochs throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., epochs = 0)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., epochs = -1)
    end

    @testset "non-positive batch_size throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., batch_size = 0)
    end

    @testset "non-positive lr_steps throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., lr_steps = 0)
    end

    @testset "non-positive lr_start throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., lr_start = 0)
    end

    @testset "non-positive lr_final throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., lr_final = 0)
    end

    @testset "lr_final > lr_start throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., lr_final = 1f0, lr_start = 1f-3)
    end

    @testset "invalid device throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., device = :tpu)
    end

end

# ---------------------------------------------------------------------------
# TrainSettings TOML round-trip
# ---------------------------------------------------------------------------

@testset "TrainSettings TOML round-trip" begin

    s    = TrainSettings(; VALID_TS_KWARGS...)
    path = tempname() * ".toml"
    save_train_settings(path, s)
    s2   = load_train_settings(path)
    rm(path)

    @test s2.epochs     == s.epochs
    @test s2.batch_size == s.batch_size
    @test s2.lr_start   == s.lr_start
    @test s2.lr_final   == s.lr_final
    @test s2.lr_steps   == s.lr_steps
    @test s2.strategy.steps       == s.strategy.steps
    @test s2.strategy.durations   == s.strategy.durations
    @test s2.strategy.noise_scale == s.strategy.noise_scale
    @test s2.device               == s.device

end

# ---------------------------------------------------------------------------
# Integration: small training run
# ---------------------------------------------------------------------------

@testset "train_model! integration" begin

    ts    = TrainSettings(; VALID_TS_KWARGS..., epochs = TR_EPOCHS, lr_steps = 2)
    model = deepcopy(TR_MODEL)

    train_rollout, val_rollout, train_1step, val_1step =
        train_model!(model, TR_TRAIN_LOADER, TR_VAL_LOADER, ts)

    @testset "loss arrays have length == epochs" begin
        @test length(train_rollout) == TR_EPOCHS
        @test length(val_rollout)   == TR_EPOCHS
        @test length(train_1step)   == TR_EPOCHS
        @test length(val_1step)     == TR_EPOCHS
    end

    @testset "loss arrays are Float32" begin
        @test eltype(train_rollout) == Float32
        @test eltype(val_rollout)   == Float32
        @test eltype(train_1step)   == Float32
        @test eltype(val_1step)     == Float32
    end

    @testset "all losses are finite and positive" begin
        @test all(isfinite, train_rollout)
        @test all(isfinite, val_rollout)
        @test all(isfinite, train_1step)
        @test all(isfinite, val_1step)
        @test all(>(0), train_rollout)
        @test all(>(0), val_rollout)
        @test all(>(0), train_1step)
        @test all(>(0), val_1step)
    end

end
