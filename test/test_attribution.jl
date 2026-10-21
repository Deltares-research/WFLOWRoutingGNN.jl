using Test
using WflowRoutingGNN
using Statistics

# Synthetic drainage tree (unit cell areas):
#
#   1 -> 2 -> 3 -> 4(pit)        7 -> 6 -> 5 -> 3        8 -> 2        9 (isolated pit)
#
const _AT_DOWN = [2, 3, 4, 0, 3, 5, 6, 2, 0]
const _AT_ACTIVE = fill(true, 9)
const _AT_AREA = fill(1.0, 9)
const _AT_UPAREA = Float64[1, 3, 7, 8, 3, 2, 1, 1, 1]

@testset "Subcatchment error attribution" begin

    @testset "cell_areas_km2" begin
        A = WflowRoutingGNN.cell_areas_km2([0.0], 1.0, 1.0, 2)
        @test size(A) == (2, 1)
        @test A[1, 1] ≈ 12364 rtol = 0.01          # 111.19 km x 111.19 km at the equator
        B = WflowRoutingGNN.cell_areas_km2([60.0], 1.0, -1.0, 1)
        @test B[1, 1] ≈ 0.5 * A[1, 1] rtol = 0.01  # cos(60°) = 0.5
    end

    @testset "network_outlets (Route B)" begin
        river = fill(true, 9)
        o = network_outlets(_AT_DOWN, river, _AT_UPAREA; min_area_km2 = 3)
        # sink 4, plus branch ends 2 and 5 above the significant confluence at 3
        @test sort(o) == [2, 4, 5]
        # high threshold: only the sink survives
        @test network_outlets(_AT_DOWN, river, _AT_UPAREA; min_area_km2 = 8) == [4]
        # off-river cells are never outlets
        river2 = copy(river); river2[2] = false
        @test !(2 in network_outlets(_AT_DOWN, river2, _AT_UPAREA; min_area_km2 = 3))
    end

    @testset "gauge_outlets (Route A)" begin
        R = fill(NaN, 3, 3)
        R[2] = 11; R[5] = 22; R[7] = 33; R[9] = 0
        river = fill(true, 9); river[7] = false
        cells, ids = gauge_outlets(R, river)
        @test cells == [2, 5]
        @test ids == [11, 22]
    end

    outlets = [2, 5, 4]
    tree = build_subcatchment_tree(_AT_DOWN, _AT_ACTIVE, outlets, _AT_AREA; uparea_km2 = _AT_UPAREA)

    @testset "tessellation and tree" begin
        @test tree.label == [1, 1, 3, 3, 2, 2, 2, 1, 0]
        @test tree.parent == [3, 3, 0]
        @test tree.children == [Int[], Int[], [1, 2]]
        @test tree.n_cells == [3, 3, 2]
        @test tree.inc_area_km2 == [3.0, 3.0, 2.0]
        @test tree.cum_area_km2 == [3.0, 3.0, 8.0]
        @test tree.check.boundary_ok
        @test tree.check.covered_ok
        @test tree.check.n_unassigned == 1
        @test tree.check.area_rel_err_max < 1e-12   # cumulative area == model upstream area
        @test WflowRoutingGNN.mainstem(tree) == [1, 3]
    end

    @testset "tessellation is exact for any outlet set" begin
        for outs in ([4], [3], [1, 4], [2, 3, 5], [8, 7, 4, 6])
            t = build_subcatchment_tree(_AT_DOWN, _AT_ACTIVE, outs, _AT_AREA)
            @test t.check.boundary_ok && t.check.covered_ok
            @test sum(t.n_cells) + t.check.n_unassigned == 9
        end
        @test_throws ArgumentError build_subcatchment_tree(_AT_DOWN, _AT_ACTIVE, [4, 4], _AT_AREA)
        @test_throws ErrorException WflowRoutingGNN.assign_subcatchments([2, 1], Int[], [true, true])
    end

    @testset "error decomposition" begin
        T = 6
        t  = collect(1.0:T)
        a  = [0.5, -1.0, 2.0, 0.0, 1.0, -0.5]    # dQ at outlet cell 2
        b  = [1.0, 1.0, -1.0, 0.5, 0.0, 2.0]     # dQ at outlet cell 5
        c  = [0.2, -0.4, 0.6, 0.1, -0.3, 0.0]    # error injected between inlets and outlet cell 4
        Q  = zeros(9, T); P = zeros(9, T)
        Q[2, :] = 3 .+ t;  Q[5, :] = 2 .+ t;  Q[4, :] = Q[2, :] .+ Q[5, :] .+ 10
        P[2, :] = Q[2, :] .+ a
        P[5, :] = Q[5, :] .+ b
        P[4, :] = Q[4, :] .+ a .+ b .+ c
        d = subcatch_error_decomposition(tree, P, Q)
        rms(x) = sqrt(mean(abs2, x))
        @test d[1].rms_adv == 0 && d[1].rms_local ≈ rms(a)             # headwater: all local
        @test d[2].rms_adv == 0 && d[2].rms_local ≈ rms(b)
        @test d[3].rms_adv ≈ rms(a .+ b)
        @test d[3].rms_local ≈ rms(c)
        @test d[3].rms_total ≈ rms(a .+ b .+ c)
        @test d[3].mean_local ≈ mean(c)
        @test d[3].mean_true_local ≈ 10
        @test d[3].rel_local ≈ rms(c) / 10
        @test d[3].local_over_total ≈ rms(c) / rms(a .+ b .+ c)
        @test d[3].n_steps == T
        @test d[1].rmse ≈ rms(a)

        # NaN at an inlet masks the step for the downstream outlet only
        P2 = copy(P); P2[2, 3] = NaN
        d2 = subcatch_error_decomposition(tree, P2, Q)
        @test d2[3].n_steps == T - 1
        @test d2[2].n_steps == T

        # overbank fraction
        qb = fill(NaN, 9); qb[4] = Q[4, 3]                              # exceeded at steps 4..6
        d3 = subcatch_error_decomposition(tree, P, Q; qbankfull = qb)
        @test d3[3].frac_overbank ≈ 3 / T
        @test isnan(d3[1].frac_overbank)
    end

    @testset "attributes" begin
        R = Dict("KsatVer" => Float64[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 "wflow_uparea" => _AT_UPAREA,
                 "wflow_reservoirareas" => Float64[0, 0, 1, 0, 0, 0, 0, 0, 1])
        river = fill(true, 9)
        at = Dict(WflowRoutingGNN.subcatch_attributes(tree, R, river))
        @test at["ksat_ver"] ≈ [mean([1, 2, 8]), mean([5, 6, 7]), mean([3, 4])]
        @test at["uparea_km2"] == _AT_UPAREA[outlets]
        @test at["reservoir_frac"] ≈ [0, 0, 0.5]                        # cell 3 is in region 3, cell 9 unassigned
        @test !haskey(at, "slope_missing")
        @test !haskey(at, "land_slope")                                  # absent variable is skipped
    end

    @testset "spearman_test" begin
        x = collect(1.0:8)
        s = spearman_test(x, x .^ 3)
        @test s.rho ≈ 1 && s.n == 8 && s.p < 0.01
        @test spearman_test(x, -x).rho ≈ -1
        @test isnan(spearman_test(x, fill(1.0, 8)).rho)               # constant input
        @test isnan(spearman_test(x[1:3], x[1:3]).rho)                # too few pairs
        # NaN pairs are dropped
        y = copy(x); y[2] = NaN
        @test spearman_test(x, y).n == 7
        # uncorrelated: large p
        z = [3.0, 1, 4, 1.5, 5, 9, 2, 6, 5.5, 3.5]
        w = [2.0, 7, 1, 8, 2.5, 8.5, 1.8, 2.8, 4.5, 9]
        @test spearman_test(z, w).p > 0.1
        # reproducible
        @test spearman_test(z, w).p == spearman_test(z, w).p

        rk = WflowRoutingGNN.rank_attributes(["a" => x, "b" => [3.0, 1, 4, 1.5, 5, 9, 2, 6], "c" => fill(1.0, 8)], x .^ 2)
        @test [r.attribute for r in rk] == ["a", "b"]                 # constant attribute dropped, sorted by |rho|
        @test all(r -> r.low_power, rk)
        @test rk[1].p_bonferroni ≈ min(1.0, rk[1].p * 2)
        # near-constant attribute (rounding-level spread) is dropped
        rk2 = WflowRoutingGNN.rank_attributes(["a" => x, "flat" => 2000.0 .- 1e-4 .* x], x .^ 2)
        @test [r.attribute for r in rk2] == ["a"]
    end
end
