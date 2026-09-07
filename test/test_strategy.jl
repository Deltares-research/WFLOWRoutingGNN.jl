using Test
using Flux
using GraphNeuralNetworks
using Statistics
using CUDA

# ---------------------------------------------------------------------------
# Shared synthetic graph data
# ---------------------------------------------------------------------------

const ST_N_STATE   = length(DOMAIN_VARS["river"]["state"])
const ST_N_FORCING = length(DOMAIN_VARS["river"]["forcing"])
const ST_N_STATIC  = length(DOMAIN_VARS["river"]["static"])
const ST_N_NODES   = 6
const ST_N_EDGES   = 10

const ST_TOPO = rand_graph(ST_N_NODES, ST_N_EDGES)

function make_st_graph()
    GNNGraph(ST_TOPO;
             ndata = (state   = rand(Float32, ST_N_STATE,   ST_N_NODES),
                      forcing = rand(Float32, ST_N_FORCING, ST_N_NODES)))
end

# A batch of 3 consecutive graphs (supports up to 2-step rollout)
const ST_BATCH  = [make_st_graph() for _ in 1:3]
const ST_STATIC = rand(Float32, ST_N_STATIC, ST_N_NODES)
const ST_MODEL  = WflowGNN(ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1))

# ---------------------------------------------------------------------------
# TrainingStrategy constructor validation
# ---------------------------------------------------------------------------

@testset "TrainingStrategy constructor validation" begin

    @test TrainingStrategy([1, 2], [3, 4]) isa TrainingStrategy
    @test TrainingStrategy([1], [3]; loss_type = :huber) isa TrainingStrategy

    @testset "mismatched steps/durations throws" begin
        @test_throws ArgumentError TrainingStrategy([1, 2], [3])
    end

    @testset "empty steps throws" begin
        @test_throws ArgumentError TrainingStrategy(Int[], Int[])
    end

    @testset "non-positive steps throws" begin
        @test_throws ArgumentError TrainingStrategy([0, 2], [3, 3])
        @test_throws ArgumentError TrainingStrategy([-1], [3])
    end

    @testset "non-positive durations throws" begin
        @test_throws ArgumentError TrainingStrategy([1], [0])
    end

    @testset "negative noise_scale throws" begin
        @test_throws ArgumentError TrainingStrategy([1], [3], -0.1)
    end

    @testset "current_steps initialised to steps[1]" begin
        s = TrainingStrategy([2, 5], [10, 10])
        @test s.current_steps == 2
    end

end

# ---------------------------------------------------------------------------
# TrainingStrategy TOML round-trip
# ---------------------------------------------------------------------------

@testset "TrainingStrategy TOML round-trip" begin

    s    = TrainingStrategy([1, 3], [5, 10], 0.05)
    path = tempname() * ".toml"
    save_training_strategy(path, s)
    s2   = load_training_strategy(path)
    rm(path)

    @test s2.steps       == s.steps
    @test s2.durations   == s.durations
    @test s2.noise_scale == s.noise_scale
    # current_steps is re-initialised to steps[1] on load
    @test s2.current_steps == s.steps[1]

end

# ---------------------------------------------------------------------------
# update_steps!
# ---------------------------------------------------------------------------

@testset "update_steps!" begin

    # Schedule: 1 step for epochs 1-3, 2 steps for epochs 4-6, 3 steps for 7-8
    strat = TrainingStrategy([1, 2, 3], [3, 3, 2])

    @testset "phase 1" begin
        for epoch in 1:3
            update_steps!(strat, epoch)
            @test strat.current_steps == 1
        end
    end

    @testset "phase 2" begin
        for epoch in 4:6
            update_steps!(strat, epoch)
            @test strat.current_steps == 2
        end
    end

    @testset "phase 3" begin
        for epoch in 7:8
            update_steps!(strat, epoch)
            @test strat.current_steps == 3
        end
    end

    @testset "beyond last phase stays at last step" begin
        for epoch in 9:12
            update_steps!(strat, epoch)
            @test strat.current_steps == 3
        end
    end

end

# ---------------------------------------------------------------------------
# Huber loss helper
# ---------------------------------------------------------------------------

@testset "peak_weighted_huber_loss" begin
    pred = Float32[1.0 2.0 3.0; 1.5 2.5 3.5]
    target = Float32[1.1 2.2 2.4; 1.6 2.3 4.5]
    u = Float32[2.0, 3.0]
    s = Float32[1.0, 1.5]

    loss = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u, s; delta = 1.0f0,
                                                  lambda = 2.0f0,
                                                  gamma = 1.0f0, w_max = 4.0f0)
    @test loss isa Float32
    @test isfinite(loss)
    @test loss >= 0f0

    loss_mse = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u, s; delta = 0.0f0,
                                                      lambda = 0.0f0,
                                                      gamma = 1.0f0, w_max = 1.0f0)
    @test isfinite(loss_mse)
    @test loss_mse >= 0f0

    @testset "matrix u/s (per-node) reproduces vector (per-channel) result when constant per row" begin
        # A (nvar, nnode) matrix with each row equal to the scalar channel
        # value must give exactly the same loss as the plain-vector call.
        u_mat = repeat(u, 1, size(pred, 2))
        s_mat = repeat(s, 1, size(pred, 2))
        loss_mat = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u_mat, s_mat;
                                                          delta = 1.0f0, lambda = 2.0f0,
                                                          gamma = 1.0f0, w_max = 4.0f0)
        @test loss_mat ≈ loss
    end

    @testset "matrix u/s with genuinely different per-node thresholds" begin
        # A node-varying threshold must change the loss relative to a
        # uniform (per-channel scalar) threshold.
        u_mat = Float32[1.0 2.0 5.0; 1.0 2.0 5.0]
        s_mat = repeat(s, 1, size(pred, 2))
        loss_mat = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u_mat, s_mat;
                                                          delta = 1.0f0, lambda = 2.0f0,
                                                          gamma = 1.0f0, w_max = 4.0f0)
        @test isfinite(loss_mat)
        @test loss_mat != loss
    end
end

@testset "peak loss estimation helpers" begin
    target = Float32[
        1.0 2.0 3.0 4.0 5.0 6.0;
        10.0 12.0 14.0 16.0 18.0 20.0
    ]
    u, s = WflowRoutingGNN.peak_quantile_stats(target)
    @test length(u) == 2
    @test length(s) == 2
    @test all(>(0), u)
    @test all(>(0), s)

    est = WflowRoutingGNN.estimate_peak_loss_parameters(target)
    @test est.delta > 0
    @test est.w_max >= 1f0
    @test length(est.u) == 2
    @test length(est.s) == 2

    summary = WflowRoutingGNN.peak_loss_summary(target, target .+ 0.5f0, u, s;
                                              delta = 1.0f0, lambda = 2.0f0,
                                              gamma = 1.0f0, w_max = 4.0f0)
    @test summary.loss >= 0f0
    @test summary.w_mean >= 0f0
    @test summary.w_max >= 1f0
end

if CUDA.functional()
    @testset "peak_weighted_huber_loss CUDA regression" begin
        pred = CUDA.rand(Float32, 2, 4)
        target = CUDA.rand(Float32, 2, 4)
        u = CUDA.rand(Float32, 2)
        s = CUDA.rand(Float32, 2)
        s = max.(s, eps(Float32))
        loss = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u, s;
                                                        delta = 1.0f0,
                                                        lambda = 2.0f0,
                                                        gamma = 1.0f0,
                                                        w_max = 4.0f0)
        @test loss isa Float32
        @test isfinite(loss)
        @test loss >= 0f0
    end
end

# ---------------------------------------------------------------------------
# peak_weight_matrix
# ---------------------------------------------------------------------------

@testset "peak_weight_matrix" begin
    pred = Float32[1.0 2.0 3.0; 1.5 2.5 3.5]
    target = Float32[1.1 2.2 2.4; 1.6 2.3 4.5]
    u = Float32[2.0, 3.0]
    s = Float32[1.0, 1.5]

    @testset "lambda == 0 gives all-ones weights and empty mask" begin
        weights, mask = WflowRoutingGNN.peak_weight_matrix(target, u, s; lambda = 0.0f0)
        @test size(weights) == size(target)
        @test size(mask)    == size(target)
        @test all(==(1f0), weights)
        @test !any(mask)
    end

    @testset "mask marks exactly the cells above threshold" begin
        weights, mask = WflowRoutingGNN.peak_weight_matrix(target, u, s; lambda = 2.0f0,
                                                          gamma = 1.0f0, w_max = 4.0f0)
        for i in 1:size(target, 1), j in 1:size(target, 2)
            @test mask[i, j] == (target[i, j] > u[i])
        end
        @test all(w -> 1f0 <= w <= 4.0f0, weights)
    end

    @testset "matches peak_weighted_huber_loss's internal weighting" begin
        # Reconstruct the (weight-normalised) loss from the forward-only
        # weights/mask and compare against the differentiable loss function's
        # own computation (peak_weighted_huber_loss normalises row_weights to
        # sum to 1 when lambda > 0).
        weights, _ = WflowRoutingGNN.peak_weight_matrix(target, u, s; lambda = 2.0f0,
                                                       gamma = 1.0f0, w_max = 4.0f0)
        delta = 1.0f0
        elementwise = WflowRoutingGNN._huber_element.(pred .- target, delta) .* weights
        reconstructed = sum(elementwise) / sum(weights)
        loss = WflowRoutingGNN.peak_weighted_huber_loss(pred, target, u, s; delta,
                                                       lambda = 2.0f0, gamma = 1.0f0,
                                                       w_max = 4.0f0)
        @test reconstructed ≈ loss
    end

    @testset "w_max caps the weight" begin
        weights, _ = WflowRoutingGNN.peak_weight_matrix(target, u, s; lambda = 1000.0f0, w_max = 2.0f0)
        @test all(<=(2.0f0), weights)
    end
end

# ---------------------------------------------------------------------------
# loss_function
# ---------------------------------------------------------------------------

@testset "tier 2 / tier 3 metrics" begin
    pred = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    truth = Float32[1.0, 2.0, 2.0, 4.0, 6.0]

    kge = WflowRoutingGNN.kge_metrics(pred, truth)
    @test isfinite(kge.kge)
    @test isfinite(kge.r)
    @test isfinite(kge.alpha)
    @test isfinite(kge.beta)
    @test kge.alpha > 0f0
    @test kge.beta > 0f0

    mae = WflowRoutingGNN.mae_metric(pred, truth)
    @test isfinite(mae)
    @test mae >= 0f0

    pb = WflowRoutingGNN.pbias(pred, truth)
    @test isfinite(pb)

    peak = WflowRoutingGNN.event_peak_metrics(pred, truth)
    @test isfinite(peak.peak_error)
    @test isfinite(peak.fhv)

    det = WflowRoutingGNN.event_detection_metrics(pred, truth; threshold = 2.5f0)
    @test 0f0 <= det.pod <= 1f0
    @test 0f0 <= det.false_alarm_ratio <= 1f0
    @test 0f0 <= det.csi <= 1f0

    @testset "kge_metrics uses pairwise finite mask" begin
        pred_nf  = Float32[1.0, NaN32, 3.0, 4.0, Inf32]
        truth_nf = Float32[1.0, 2.0, Inf32, 4.0, 5.0]
        k_nf = WflowRoutingGNN.kge_metrics(pred_nf, truth_nf)
        @test isfinite(k_nf.kge)
        @test isfinite(k_nf.r)
        @test isfinite(k_nf.alpha)
        @test isfinite(k_nf.beta)
    end
end

@testset "river_q_performance_metrics" begin
    # 3 nodes x 6 timesteps; node 3 has the largest upstream_area (outlet).
    pred_q = Float32[
        1.0 1.1 1.2 1.3 1.4 1.5;
        2.0 2.1 2.0 2.2 2.1 2.3;
        5.0 5.5 6.0 8.0 6.5 5.5
    ]
    true_q = Float32[
        1.0 1.0 1.3 1.2 1.5 1.4;
        2.0 2.0 2.1 2.1 2.0 2.2;
        5.0 5.2 5.8 8.5 6.0 5.6
    ]
    upstream_area = Float32[10.0, 25.0, 100.0]

    perf = WflowRoutingGNN.river_q_performance_metrics(pred_q, true_q, upstream_area)

    @test perf.gauge.outlet_idx == 3

    for m in (perf.pooled, perf.gauge)
        @test isfinite(m.kge)
        @test isfinite(m.r)
        @test isfinite(m.alpha)
        @test isfinite(m.beta)
        @test isfinite(m.mae)
        @test isfinite(m.pbias)
        @test isfinite(m.peak_error)
        @test isfinite(m.peak_ratio)
        @test isfinite(m.fhv)
        @test m.mae >= 0f0
    end

    # The gauge-only metrics must genuinely differ from the pooled ones
    # (node 3 alone is not representative of the pooled mix of all 3 nodes).
    @test perf.gauge.mae != perf.pooled.mae

    @testset "mismatched shapes throw" begin
        @test_throws ArgumentError WflowRoutingGNN.river_q_performance_metrics(
            pred_q, true_q[1:2, :], upstream_area)
        @test_throws ArgumentError WflowRoutingGNN.river_q_performance_metrics(
            pred_q, true_q, upstream_area[1:2])
    end

    @testset "NaN upstream_area entries are ignored when finding the outlet" begin
        ua_with_nan = Float32[10.0, NaN32, 100.0]
        perf2 = WflowRoutingGNN.river_q_performance_metrics(pred_q, true_q, ua_with_nan)
        @test perf2.gauge.outlet_idx == 3
    end

    @testset "non-finite mismatches do not throw" begin
        pred_q_nf = copy(pred_q)
        true_q_nf = copy(true_q)
        pred_q_nf[1, 2] = NaN32
        true_q_nf[2, 3] = Inf32
        perf3 = WflowRoutingGNN.river_q_performance_metrics(pred_q_nf, true_q_nf, upstream_area)
        @test isfinite(perf3.pooled.kge)
        @test isfinite(perf3.gauge.kge)
    end
end

@testset "loss_function" begin

    strat_1 = TrainingStrategy([1], [10])
    strat_2 = TrainingStrategy([2], [10])
    strat_2.current_steps = 2

    @testset "returns a finite non-negative Float32 (1-step)" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_1, ST_STATIC)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "returns a finite non-negative Float32 (2-step)" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_2, ST_STATIC)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "batch too short throws" begin
        strat_3 = TrainingStrategy([2], [1])
        strat_3.current_steps = 2
        short_batch = [make_st_graph(), make_st_graph()]   # length 2, need >= 3
        @test_throws ArgumentError loss_function(ST_MODEL, short_batch, strat_3, ST_STATIC)
    end

end

@testset "loss_function :huber with per-node peak_stats" begin
    strat_huber = TrainingStrategy([1], [10]; loss_type = :huber)

    peak_stats = (q = (u = Float32.(1:ST_N_NODES) .+ 1f0, s = fill(0.5f0, ST_N_NODES)),
                  h = (u = Float32.(ST_N_NODES:-1:1) .+ 1f0, s = fill(0.3f0, ST_N_NODES)))

    @testset "returns a finite non-negative Float32 with per-node stats supplied" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_huber, ST_STATIC; peak_stats = peak_stats)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0f0
    end

    @testset "falls back to coarse per-batch threshold when peak_stats omitted" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_huber, ST_STATIC)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0f0
    end

    @testset "tiles per-node stats across a multi-graph (batchsize > 1) batch" begin
        double_batch = [GNNGraphs.batch([g, g]) for g in ST_BATCH]
        l = loss_function(ST_MODEL, double_batch, strat_huber, ST_STATIC; peak_stats = peak_stats)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0f0
    end

    @testset "peak_stats node count not dividing batch node count throws" begin
        bad_stats = (q = (u = Float32[1.0, 2.0, 3.0, 4.0], s = Float32[1.0, 1.0, 1.0, 1.0]),
                     h = (u = Float32[1.0, 2.0, 3.0, 4.0], s = Float32[1.0, 1.0, 1.0, 1.0]))
        @test_throws ArgumentError loss_function(ST_MODEL, ST_BATCH, strat_huber, ST_STATIC;
                                                 peak_stats = bad_stats)
    end
end

# ---------------------------------------------------------------------------
# one_step_loss
# ---------------------------------------------------------------------------

@testset "one_step_loss" begin

    @testset "returns a finite non-negative Float32" begin
        l = one_step_loss(ST_MODEL, ST_BATCH, ST_STATIC)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "equals loss_function with current_steps=1" begin
        strat_1 = TrainingStrategy([1], [10])
        @test one_step_loss(ST_MODEL, ST_BATCH, ST_STATIC) ≈ loss_function(ST_MODEL, ST_BATCH, strat_1, ST_STATIC)  atol=1f-6
    end

end

# ---------------------------------------------------------------------------
# Noise scale behaviour
# ---------------------------------------------------------------------------

@testset "noise_scale effect" begin

    strat_noisy = TrainingStrategy([1], [10], 10.0)
    losses = [loss_function(ST_MODEL, ST_BATCH, strat_noisy, ST_STATIC) for _ in 1:10]

    @testset "noisy losses are finite and non-negative" begin
        @test all(isfinite, losses)
        @test all(>=(0), losses)
    end

    @testset "noise causes variation between calls" begin
        @test !all(==(losses[1]), losses)
    end

    @testset "zero noise gives deterministic loss" begin
        strat_clean = TrainingStrategy([1], [10], 0.0)
        l1 = loss_function(ST_MODEL, ST_BATCH, strat_clean, ST_STATIC)
        l2 = loss_function(ST_MODEL, ST_BATCH, strat_clean, ST_STATIC)
        @test l1 == l2
    end

end
