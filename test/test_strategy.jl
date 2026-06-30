using Test
using Flux
using GraphNeuralNetworks
using Statistics

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
                      forcing = rand(Float32, ST_N_FORCING, ST_N_NODES),
                      static  = rand(Float32, ST_N_STATIC,  ST_N_NODES)))
end

# A batch of 3 consecutive graphs (supports up to 2-step rollout)
const ST_BATCH = [make_st_graph() for _ in 1:3]
const ST_MODEL = WflowGNN(ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1))

# ---------------------------------------------------------------------------
# TrainingStrategy constructor validation
# ---------------------------------------------------------------------------

@testset "TrainingStrategy constructor validation" begin

    @test TrainingStrategy([1, 2], [3, 4]) isa TrainingStrategy

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
# loss_function
# ---------------------------------------------------------------------------

@testset "loss_function" begin

    strat_1 = TrainingStrategy([1], [10])
    strat_2 = TrainingStrategy([2], [10])
    strat_2.current_steps = 2

    @testset "returns a finite non-negative Float32 (1-step)" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_1)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "returns a finite non-negative Float32 (2-step)" begin
        l = loss_function(ST_MODEL, ST_BATCH, strat_2)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "batch too short throws" begin
        strat_3 = TrainingStrategy([2], [1])
        strat_3.current_steps = 2
        short_batch = [make_st_graph(), make_st_graph()]   # length 2, need >= 3
        @test_throws ArgumentError loss_function(ST_MODEL, short_batch, strat_3)
    end

end

# ---------------------------------------------------------------------------
# one_step_loss
# ---------------------------------------------------------------------------

@testset "one_step_loss" begin

    @testset "returns a finite non-negative Float32" begin
        l = one_step_loss(ST_MODEL, ST_BATCH)
        @test l isa Float32
        @test isfinite(l)
        @test l >= 0
    end

    @testset "equals loss_function with current_steps=1" begin
        strat_1 = TrainingStrategy([1], [10])
        @test one_step_loss(ST_MODEL, ST_BATCH) ≈ loss_function(ST_MODEL, ST_BATCH, strat_1)  atol=1f-6
    end

end

# ---------------------------------------------------------------------------
# Noise scale behaviour
# ---------------------------------------------------------------------------

@testset "noise_scale effect" begin

    strat_noisy = TrainingStrategy([1], [10], 10.0)
    losses = [loss_function(ST_MODEL, ST_BATCH, strat_noisy) for _ in 1:10]

    @testset "noisy losses are finite and non-negative" begin
        @test all(isfinite, losses)
        @test all(>=(0), losses)
    end

    @testset "noise causes variation between calls" begin
        @test !all(==(losses[1]), losses)
    end

    @testset "zero noise gives deterministic loss" begin
        strat_clean = TrainingStrategy([1], [10], 0.0)
        l1 = loss_function(ST_MODEL, ST_BATCH, strat_clean)
        l2 = loss_function(ST_MODEL, ST_BATCH, strat_clean)
        @test l1 == l2
    end

end
