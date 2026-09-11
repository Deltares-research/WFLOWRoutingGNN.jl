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
    @test s.upstream_points         == 5
    @test s.early_stopping          == false
    @test s.early_stopping_patience == 20
    @test s.checkpoint_every        == 0
    @test s.checkpoint_full_eval    == false

    @testset "validation" begin
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., eval_horizon = -1)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., early_stopping_patience = 0)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., checkpoint_every = -1)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., upstream_points = 0)
        @test_throws ArgumentError TrainSettings(; VALID_TS_KWARGS..., early_stopping = true, eval_horizon = 0)
        @test TrainSettings(; VALID_TS_KWARGS..., eval_horizon = 0) isa TrainSettings
    end

    @testset "TOML round-trip" begin
        sc = TrainSettings(; VALID_TS_KWARGS..., eval_horizon = 12, eval_anchors = 5,
                             upstream_points = 7,
                             early_stopping = true, early_stopping_patience = 7,
                             checkpoint_every = 3, checkpoint_full_eval = true)
        path = tempname() * ".toml"
        save_train_settings(path, sc)
        sc2 = load_train_settings(path)
        rm(path)
        @test sc2.eval_horizon            == 12
        @test sc2.eval_anchors            == 5
        @test sc2.upstream_points         == 7
        @test sc2.early_stopping          == true
        @test sc2.early_stopping_patience == 7
        @test sc2.checkpoint_every        == 3
        @test sc2.checkpoint_full_eval    == true
    end

end

# ---------------------------------------------------------------------------
# Upstream percentile node selection
# ---------------------------------------------------------------------------

@testset "upstream percentile ladder selection" begin
    areas = Float32[100, 80, 60, 40, 20]
    idxs = WflowRoutingGNN._upstream_percentile_nodes(areas, 3)
    @test idxs == [1, 3, 5]

    idxs_nan = WflowRoutingGNN._upstream_percentile_nodes(Float32[NaN, 90, 30, NaN, 10], 4)
    @test idxs_nan == [2, 3, 5]
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
    @test length(fh.anchor_starts) == fh.B
    @test length(fh.start_q_phys) == fh.B
    @test length(fh.start_q_percentile) == fh.B
    @test all(0f0 .<= fh.start_q_percentile .<= 1f0)
    @test size(fh.true_q_phys) == (fh.N, fh.horizon, fh.B)

    m = fixed_horizon_metrics(deepcopy(TR_MODEL), fh; device = :cpu)
    @test isfinite(m.rmse_q) && m.rmse_q >= 0
    @test isfinite(m.peak_ratio) && m.peak_ratio >= 0
    @test length(m.rmse_q_anchor) == fh.B
    @test length(m.peak_ratio_anchor) == fh.B
    @test all(isfinite, m.rmse_q_anchor)
    @test all(>=(0), m.rmse_q_anchor)
    @test all(isfinite, m.peak_ratio_anchor)
    @test all(>=(0), m.peak_ratio_anchor)

    @testset "per-anchor fixed-horizon CSV writer" begin
        csv_path = tempname() * ".csv"
        WflowRoutingGNN.write_fixed_horizon_anchor_table(csv_path, fh,
                                                         m.rmse_q_anchor,
                                                         m.peak_ratio_anchor)
        lines = readlines(csv_path)
        rm(csv_path)
        @test length(lines) == fh.B + 1
        @test occursin("anchor_index,start_step,start_flow_q_phys,start_flow_percentile,fixed_rmse,peak_ratio",
                       first(lines))
    end

    # Non-positive horizon / anchors → no metric.
    @test isnothing(build_fixed_horizon_eval(TR_DATASET.val, TR_STATIC, TR_STATS,
                                             "river", TR_POSTSCALE; horizon = 0))

    @testset "same-start anchor rollout matches trajectory rollout" begin
        model = deepcopy(TR_MODEL)
        split = [TR_GRAPHS[1:3]]
        pred_states, _ = evaluate_trajectory(model, split, TR_STATS, "river", TR_STATIC;
                                            device = :cpu, postscale = TR_POSTSCALE)

        fh = build_fixed_horizon_eval(split, TR_STATIC, TR_STATS, "river", TR_POSTSCALE;
                                     horizon = 2, n_anchors = 1)
        @test fh.B == 1
        state = copy(fh.states0[:, 1:TR_N_NODES])
        q_pred = Matrix{Float32}(undef, TR_N_NODES, fh.horizon)
        for k in 1:fh.horizon
            f_t    = fh.forcing[:, :, k]
            f_next = fh.forcing[:, :, min(k + 1, fh.horizon)]
            state  = model(fh.gB, state, f_t, fh.static, f_next)
            q_pred[:, k] = state[fh.qi, :]
        end
        q_phys = (q_pred .* fh.q_sigma .+ fh.q_mu) .* reshape(fh.q_postscale, TR_N_NODES, 1)
        @test maximum(abs, q_phys .- pred_states[1, :, :]) < 1f-3
    end

end

# ---------------------------------------------------------------------------
# System volume / budget diagnostic
# ---------------------------------------------------------------------------

@testset "volume_budget_diagnostics" begin
    diags = (
        pred_h = Float32[1 2 3; 2 3 4],
        true_h = Float32[1 2 2; 1 2 3],
        pred_q = Float32[5 6 7; 8 9 10],
        true_q = Float32[4 5 6; 7 8 9],
        inwater = Float32[1 1 1; 2 2 2],
        net_flux = Float32[0 2 2; 0 3 3],
    )

    vb = volume_budget_diagnostics(diags;
                                   postscale_q = Float32[10, 20],
                                   postscale_h = Float32[2, 4],
                                   dt = 2f0,
                                   upstream_area = Float32[10, 50])

    @test vb.outlet_idx == 2
    @test vb.v_pred ≈ Float32[15, 25, 35]
    @test vb.v_true ≈ Float32[10, 20, 25]
    @test vb.delta_v_pred ≈ Float32[0, 10, 20]
    @test vb.delta_v_true ≈ Float32[0, 10, 15]
    @test vb.cum_inflow_pred ≈ Float32[6, 12, 18]
    @test vb.cum_outflow_pred ≈ Float32[16, 34, 54]
    @test vb.cum_outflow_true ≈ Float32[14, 30, 48]
    @test vb.budget_residual_pred ≈ Float32[0, 0, 0]
    @test isapprox(vb.volume_pbias, 36.363636f0; atol = 1f-4)
    @test isapprox(vb.volume_drift, 2.5f0; atol = 1f-5)
    @test vb.budget_residual_rms ≈ 0f0
end

# ---------------------------------------------------------------------------
# Tier-1 peak-loss diagnostics
# ---------------------------------------------------------------------------

@testset "peak_epoch_diagnostics" begin

    diag_batch = first(TR_TRAIN_LOADER)

    @testset "non-huber strategy returns all-NaN32" begin
        strat = TrainingStrategy([1, 2], [2, 2])  # default loss_type = :mse
        model = deepcopy(TR_MODEL)
        d = peak_epoch_diagnostics(model, diag_batch, strat, TR_STATIC)
        @test isnan(d.c_peak)
        @test isnan(d.rmse_high)
        @test isnan(d.mae_high)
        @test isnan(d.w_mean)
        @test isnan(d.w_max)
        @test isnan(d.w_min)
        @test isnan(d.q_grad_norm)
        @test isnan(d.h_grad_norm)
        @test isnan(d.peak_grad_frac)
    end

    @testset "huber strategy without peak_stats fallback" begin
        strat = TrainingStrategy([1, 2], [2, 2]; loss_type = :huber, peak_lambda = 2.0f0)
        model = deepcopy(TR_MODEL)
        d = peak_epoch_diagnostics(model, diag_batch, strat, TR_STATIC)
        @test isfinite(d.c_peak)
        @test 0f0 <= d.c_peak <= 1f0
        @test isfinite(d.w_mean) && d.w_mean >= 1f0
        @test isfinite(d.w_max)  && d.w_max  >= 1f0
        @test isfinite(d.w_min)  && d.w_min  >= 1f0
        @test isfinite(d.q_grad_norm) && d.q_grad_norm >= 0f0
        @test isfinite(d.h_grad_norm) && d.h_grad_norm >= 0f0
    end

    @testset "huber strategy with peak_stats" begin
        strat = TrainingStrategy([1, 2], [2, 2]; loss_type = :huber, peak_lambda = 2.0f0)
        model = deepcopy(TR_MODEL)
        node_stats = peak_node_stats(TR_GRAPHS, "river"; frac_train = 0.7)
        state_vars = DOMAIN_VARS["river"]["state"]
        peak_stats = (q = node_stats[state_vars[1]], h = node_stats[state_vars[2]])
        d = peak_epoch_diagnostics(model, diag_batch, strat, TR_STATIC; peak_stats)
        @test isfinite(d.c_peak)
        @test isfinite(d.w_mean)
    end

    @testset "single-graph batch (length < 2) returns all-NaN32" begin
        strat = TrainingStrategy([1, 2], [2, 2]; loss_type = :huber, peak_lambda = 2.0f0)
        model = deepcopy(TR_MODEL)
        d = peak_epoch_diagnostics(model, diag_batch[1:1], strat, TR_STATIC)
        @test isnan(d.c_peak)
    end

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

    @testset "peak-loss history is all-NaN32 for :mse strategy (default)" begin
        @test length(losses.peak_c_peak) == TR_EPOCHS
        @test all(isnan, losses.peak_c_peak)
        @test all(isnan, losses.peak_rmse_high)
        @test all(isnan, losses.peak_mae_high)
        @test all(isnan, losses.peak_w_mean)
        @test all(isnan, losses.peak_w_max)
        @test all(isnan, losses.peak_w_min)
        @test all(isnan, losses.peak_q_grad_norm)
        @test all(isnan, losses.peak_h_grad_norm)
        @test all(isnan, losses.peak_grad_frac)
    end

    @testset "fixed-horizon guard histories are NaN when fixed eval is disabled" begin
        @test length(losses.val_peak_ratio_frac_gt2) == TR_EPOCHS
        @test length(losses.val_fixed_rmse_highflow) == TR_EPOCHS
        @test all(isnan, losses.val_peak_ratio_frac_gt2)
        @test all(isnan, losses.val_fixed_rmse_highflow)
    end

end

# ---------------------------------------------------------------------------
# Integration: small training run with :huber peak-weighted loss
# ---------------------------------------------------------------------------

@testset "train_model! integration (huber loss, peak diagnostics)" begin

    huber_strategy = TrainingStrategy([1, 2], [2, 2]; loss_type = :huber, peak_lambda = 2.0f0)
    ts    = TrainSettings(; VALID_TS_KWARGS..., epochs = TR_EPOCHS, lr_steps = 2,
                         strategy = huber_strategy)
    model = deepcopy(TR_MODEL)

    node_stats = peak_node_stats(TR_GRAPHS, "river"; frac_train = 0.7)
    state_vars = DOMAIN_VARS["river"]["state"]
    peak_stats = (q = node_stats[state_vars[1]], h = node_stats[state_vars[2]])

    losses = train_model!(model, TR_TRAIN_LOADER, TR_VAL_LOADER, ts, TR_STATIC;
                          peak_stats = peak_stats)

    @test length(losses.peak_c_peak) == TR_EPOCHS
    @test all(isfinite, losses.peak_c_peak)
    @test all(isfinite, losses.peak_w_mean)
    @test all(isfinite, losses.peak_q_grad_norm)
    @test all(isfinite, losses.peak_h_grad_norm)
    @test all(v -> 0f0 <= v <= 1f0, losses.peak_c_peak)

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
    @test length(losses.val_peak_ratio_frac_gt2) == TR_EPOCHS
    @test length(losses.val_fixed_rmse_highflow) == TR_EPOCHS
    @test all(isfinite, losses.val_fixed_rmse)
    @test all(>=(0),    losses.val_fixed_rmse)
    @test all(isfinite, losses.val_peak_ratio)
    @test all(v -> 0f0 <= v <= 1f0, losses.val_peak_ratio_frac_gt2)
    @test all(v -> isnan(v) || v >= 0f0, losses.val_fixed_rmse_highflow)
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
