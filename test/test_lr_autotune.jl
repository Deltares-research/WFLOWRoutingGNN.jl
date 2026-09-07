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
    @test rec.gnorm_cap > 10.0 * minimum(lrs)
end
