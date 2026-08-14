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

const TR_GRAPHS, TR_STATS, TR_GRID, TR_POSTSCALE, TR_STATIC = build_wflow_graph(TR_STATICMAPS, TR_OUTPUT_NC, "river")

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

    @testset "invalid h_loss_scale throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., h_loss_scale = :relative)
        @test TrainSettings(; VALID_TS_KWARGS..., h_loss_scale = :increment) isa TrainSettings
    end

    @testset "invalid phase_backoff_factor throws" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., phase_backoff_factor = 0.0)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., phase_backoff_factor = 1.5)
        @test TrainSettings(; VALID_TS_KWARGS..., phase_backoff_factor = 1.0) isa TrainSettings
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
    @test s2.h_loss_scale         == s.h_loss_scale
    @test s2.phase_backoff_factor == s.phase_backoff_factor

    # Non-default scale survives the round-trip too.
    si   = TrainSettings(; VALID_TS_KWARGS..., h_loss_scale = :increment)
    save_train_settings(path, si)
    si2  = load_train_settings(path)
    rm(path)
    @test si2.h_loss_scale == :increment

    # Non-default backoff factor survives too.
    sb   = TrainSettings(; VALID_TS_KWARGS..., phase_backoff_factor = 0.25)
    save_train_settings(path, sb)
    sb2  = load_train_settings(path)
    rm(path)
    @test sb2.phase_backoff_factor ≈ 0.25f0

end

# ---------------------------------------------------------------------------
# TrainSettings: fixed-horizon eval / early stopping / checkpoint fields
# ---------------------------------------------------------------------------

@testset "TrainSettings fixed-horizon/early-stop/checkpoint fields" begin

    s = TrainSettings(; VALID_TS_KWARGS...)
    @test s.eval_horizon            == 30
    @test s.eval_anchors            == 32
    @test s.early_stopping          == false
    @test s.early_stopping_patience == 20
    @test s.checkpoint_every        == 0
    @test s.checkpoint_full_eval    == false

    @testset "validation" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., eval_horizon = -1)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., early_stopping_patience = 0)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., checkpoint_every = -1)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., early_stopping = true, eval_horizon = 0)
        @test TrainSettings(; VALID_TS_KWARGS..., eval_horizon = 0) isa TrainSettings
    end

    @testset "TOML round-trip" begin
        sc = TrainSettings(; VALID_TS_KWARGS..., eval_horizon = 12, eval_anchors = 5,
                             early_stopping = true, early_stopping_patience = 7,
                             checkpoint_every = 3, checkpoint_full_eval = true)
        path = tempname() * ".toml"
        save_train_settings(path, sc)
        sc2 = load_train_settings(path)
        rm(path)
        @test sc2.eval_horizon            == 12
        @test sc2.eval_anchors            == 5
        @test sc2.early_stopping          == true
        @test sc2.early_stopping_patience == 7
        @test sc2.checkpoint_every        == 3
        @test sc2.checkpoint_full_eval    == true
    end

end

# ---------------------------------------------------------------------------
# Fixed-horizon validation eval
# ---------------------------------------------------------------------------

@testset "fixed-horizon validation eval" begin

    fh = build_fixed_horizon_eval(TR_DATASET.val, TR_STATIC, TR_STATS, "river",
                                  TR_POSTSCALE; horizon = 2, n_anchors = 3)
    @test fh isa FixedHorizonEval
    @test fh.horizon == 2
    @test fh.B >= 1
    @test fh.N == TR_N_NODES
    @test size(fh.forcing)     == (TR_N_FORCING, fh.N * fh.B, fh.horizon)
    @test size(fh.states0)     == (TR_N_STATE,   fh.N * fh.B)
    @test size(fh.true_q_phys) == (fh.N, fh.horizon, fh.B)

    rmse, peak = fixed_horizon_metrics(deepcopy(TR_MODEL), fh; device = :cpu)
    @test isfinite(rmse) && rmse >= 0
    @test isfinite(peak) && peak >= 0

    # Non-positive horizon / anchors → no metric.
    @test isnothing(build_fixed_horizon_eval(TR_DATASET.val, TR_STATIC, TR_STATS,
                                             "river", TR_POSTSCALE; horizon = 0))

end

# ---------------------------------------------------------------------------
# Integration: small training run
# ---------------------------------------------------------------------------

@testset "train_model! integration" begin

    ts    = TrainSettings(; VALID_TS_KWARGS..., epochs = TR_EPOCHS, lr_steps = 2)
    model = deepcopy(TR_MODEL)

    losses = train_model!(model, TR_TRAIN_LOADER, TR_VAL_LOADER, ts, TR_STATIC)
    train_rollout = losses.train_rollout
    val_rollout   = losses.val_rollout
    train_1step   = losses.train_1step
    val_1step     = losses.val_1step

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

# ---------------------------------------------------------------------------
# Integration: fixed-horizon eval, early stopping & checkpoint callback
# ---------------------------------------------------------------------------

@testset "train_model! fixed-horizon eval + checkpoints" begin

    fh    = build_fixed_horizon_eval(TR_DATASET.val, TR_STATIC, TR_STATS, "river",
                                     TR_POSTSCALE; horizon = 2, n_anchors = 3)
    ts    = TrainSettings(; VALID_TS_KWARGS..., epochs = TR_EPOCHS, lr_steps = 2,
                          eval_horizon = 2, eval_anchors = 3, checkpoint_every = 2)
    model = deepcopy(TR_MODEL)

    ckpt_epochs = Int[]
    cb = (m, epoch) -> push!(ckpt_epochs, epoch)

    losses = train_model!(model, TR_TRAIN_LOADER, TR_VAL_LOADER, ts, TR_STATIC;
                          fixed_eval = fh, checkpoint_callback = cb)

    @test length(losses.val_fixed_rmse) == TR_EPOCHS
    @test length(losses.val_peak_ratio) == TR_EPOCHS
    @test all(isfinite, losses.val_fixed_rmse)
    @test all(>=(0),    losses.val_fixed_rmse)
    @test all(isfinite, losses.val_peak_ratio)
    @test ckpt_epochs == [2, 4]          # checkpoint_every = 2, epochs = 4
    @test losses.stopped_epoch == TR_EPOCHS

end

@testset "train_model! early stopping" begin

    fh    = build_fixed_horizon_eval(TR_DATASET.val, TR_STATIC, TR_STATS, "river",
                                     TR_POSTSCALE; horizon = 2, n_anchors = 3)
    ts    = TrainSettings(; VALID_TS_KWARGS..., epochs = 12, lr_steps = 2,
                          strategy = TrainingStrategy([1, 2], [6, 6]),
                          eval_horizon = 2, eval_anchors = 3,
                          early_stopping = true, early_stopping_patience = 1)
    model = deepcopy(TR_MODEL)

    losses = train_model!(model, TR_TRAIN_LOADER, TR_VAL_LOADER, ts, TR_STATIC;
                          fixed_eval = fh)

    # With patience 1, training stops (and restores best weights) as soon as the
    # fixed-horizon RMSE fails to improve; history is truncated to the stop epoch.
    @test losses.stopped_epoch <= 12
    @test length(losses.val_fixed_rmse) == losses.stopped_epoch
    @test length(losses.train_rollout)  == losses.stopped_epoch
    @test losses.best_epoch >= 1

end
