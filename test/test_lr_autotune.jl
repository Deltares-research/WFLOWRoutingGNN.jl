using Test

# Reuse the script helper directly; guarded main() prevents execution on include.
include(joinpath(@__DIR__, "..", "scripts", "lr_range_test.jl"))

@testset "recommend_lr robustness" begin
    n = 40
    lrs = collect(10 .^ range(-7, -1; length = n))

    # Synthetic smooth-loss curve with early hump, descent, then late rise.
    losses = Float64[]
    for i in 1:n
        if i <= 8
            push!(losses, 600.0 + 40.0 * i)
        elseif i <= 28
            push!(losses, 940.0 - 34.0 * (i - 8))
        else
            push!(losses, 260.0 + 35.0 * (i - 28))
        end
    end

    # Healthy plateau + one transient spike + sustained late growth.
    gnorms = fill(120.0, n)
    gnorms[12] = 1.0e7
    for i in 31:n
        gnorms[i] = 120.0 * (1.35 ^ (i - 30))
    end

    rec = recommend_lr(lrs, losses, gnorms)

    @test isfinite(rec.safe)
    @test rec.safe > 10.0 * minimum(lrs)
    @test !rec.near_lr_floor
    @test !rec.min10_near_floor
    @test rec.gnorm_cap > 10.0 * minimum(lrs)
end

@testset "aggregate_lr_recommendations" begin
    recs = [
        (; safe = 1e-3, gnorm_ref = 2.0e2, near_lr_floor = false, min_over_10 = 2e-3, min10_near_floor = false),
        (; safe = 9e-4, gnorm_ref = 1.8e2, near_lr_floor = false, min_over_10 = 1.7e-3, min10_near_floor = false),
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 1e-7, min10_near_floor = true),
    ]
    agg = aggregate_lr_recommendations(recs)
    @test agg.n_valid == 2
    @test agg.n_total == 3
    @test agg.safe ≈ 9.5e-4
    @test agg.gnorm_ref ≈ 190.0
    @test !agg.used_fallback

    agg_fallback = aggregate_lr_recommendations([
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 5e-3, min10_near_floor = false),
        (; safe = NaN,  gnorm_ref = NaN,   near_lr_floor = false, min_over_10 = 6e-3, min10_near_floor = false),
    ])
    @test agg_fallback.n_valid == 0
    @test agg_fallback.used_fallback
    @test agg_fallback.safe ≈ 5.5e-3

    agg_none = aggregate_lr_recommendations([
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 1e-7, min10_near_floor = true),
        (; safe = NaN,  gnorm_ref = NaN,   near_lr_floor = false, min_over_10 = NaN,  min10_near_floor = true),
    ])
    @test agg_none.n_valid == 0
    @test !agg_none.used_fallback
    @test !isfinite(agg_none.safe)
end

@testset "probe_training_strategy inherits configured loss" begin
    base = TrainingStrategy([1, 2, 5], [3, 3, 3], 0.05f0;
                            h_loss_weight = 0.7f0,
                            loss_type = :huber,
                            peak_delta = 0.3f0,
                            peak_lambda = 2.5f0,
                            peak_gamma = 1.2f0,
                            peak_w_max = 6.0f0)

    probe = probe_training_strategy(base, 1)
    @test probe.steps == [1]
    @test probe.durations == [1]
    @test probe.current_steps == 1
    @test probe.noise_scale == base.noise_scale
    @test probe.h_loss_weight == base.h_loss_weight
    @test probe.loss_type == base.loss_type
    @test probe.peak_delta == base.peak_delta
    @test probe.peak_lambda == base.peak_lambda
    @test probe.peak_gamma == base.peak_gamma
    @test probe.peak_w_max == base.peak_w_max
end
