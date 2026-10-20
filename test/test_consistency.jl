using Test
using NCDatasets
using WflowRoutingGNN

# Layer-free consistency checks of wflow data (src/consistency.jl).

const CS_MODEL_PATH = joinpath(@__DIR__, "..", "test_data", "test_model")
const CS_STATIC     = joinpath(CS_MODEL_PATH, "staticmaps.nc")
const CS_OUTPUT     = joinpath(CS_MODEL_PATH, "run_default", "output.nc")

# 3-node chain 1 -> 2 -> 3 plus a side branch 4 -> 3.
const CS_SRC = [1, 2, 4]
const CS_TGT = [2, 3, 3]

@testset "upstream_sum" begin
    q = Float32[1, 2, 3, 4]
    @test upstream_sum(q, CS_SRC, CS_TGT) == Float32[0, 1, 2 + 4, 0]

    Q = hcat(q, 10 .* q)
    U = upstream_sum(Q, CS_SRC, CS_TGT)
    @test U[:, 1] == upstream_sum(q, CS_SRC, CS_TGT)
    @test U[:, 2] == 10 .* U[:, 1]
    @test eltype(U) == Float32
end

@testset "graph_topology_checks" begin
    chk = graph_topology_checks(CS_SRC, CS_TGT, 4)
    @test chk.n_nodes == 4 && chk.n_edges == 3
    @test chk.headwaters == [1, 4]
    @test chk.outlets == [3]
    @test isempty(chk.isolated)
    @test chk.headwater_ok
    @test chk.n_single_upstream == 1                 # node 2 only
    @test chk.single_upstream_mismatches == 0

    iso = graph_topology_checks([1], [2], 3)         # node 3 is isolated
    @test iso.isolated == [3]
end

@testset "mb_scheme_residuals" begin
    n, T, dt = 4, 6, 3600.0
    wl = Float32[50, 60, 70, 80]
    q  = Float32[1 + 0.1i + 0.05t for i in 1:n, t in 1:T]
    iw = Float32[0.2 + 0.01i * t for i in 1:n, t in 1:T]
    kw = (; edge_src = CS_SRC, edge_tgt = CS_TGT, wl, dt)

    # Build h exactly from each scheme's recursion (large h0: the zero floor is inactive).
    function build_h(scheme)
        h = fill(2000f0, n, T)
        for t in 1:(T - 1)
            uq = upstream_sum(q[:, scheme === :lagged_upstream ? t : t + 1], CS_SRC, CS_TGT)
            iwt = scheme === :fully_implicit ? iw[:, t + 1] : iw[:, t]
            h[:, t + 1] = h[:, t] .+ Float32(dt) ./ wl .* (uq .+ iwt .- q[:, t + 1])
        end
        return h
    end

    for scheme in MB_SCHEMES
        h = build_h(scheme)
        res = mb_scheme_residuals(q, h, iw; kw..., scheme)
        @test size(res.residual) == (n, T - 1) && size(res.h_pred) == (n, T - 1)
        @test maximum(abs, res.residual) < 1f-2          # Float32 round-off of the recursion
        # a wrong discretisation leaves a clear residual
        other = scheme === :semi_implicit ? :lagged_upstream : :semi_implicit
        @test maximum(abs, mb_scheme_residuals(q, h, iw; kw..., scheme = other).residual) > 1f-1
    end

    # fully implicit == θ = 1 of the post-hoc residual (no floor active)
    h  = build_h(:fully_implicit)
    r1 = mb_scheme_residuals(q, h, iw; kw..., scheme = :fully_implicit).residual
    r2 = mb_residual_diagnostics(q, h, iw; edge_src = CS_SRC, edge_tgt = CS_TGT, wl, dt, θ = 1).residual
    @test r1 ≈ r2 atol = 1f-2

    # the zero floor on h_pred
    hneg = fill(0f0, n, T)
    res = mb_scheme_residuals(zeros(Float32, n, T), hneg, -ones(Float32, n, T);
                              kw..., scheme = :semi_implicit)
    @test all(==(0f0), res.h_pred)

    # invalid wl -> NaN rows
    wl_bad = copy(wl); wl_bad[2] = NaN32
    rb = mb_scheme_residuals(q, h, iw; edge_src = CS_SRC, edge_tgt = CS_TGT, wl = wl_bad, dt,
                             scheme = :fully_implicit).residual
    @test all(isnan, rb[2, :]) && all(isfinite, rb[[1, 3, 4], :])

    @test_throws ArgumentError mb_scheme_residuals(q, h, iw; kw..., scheme = :bogus)
    @test_throws ArgumentError mb_scheme_residuals(q, h[:, 1:3], iw; kw..., scheme = :semi_implicit)
    @test_throws ArgumentError mb_scheme_residuals(q[:, 1:1], h[:, 1:1], iw[:, 1:1];
                                                   kw..., scheme = :semi_implicit)
end

@testset "residual_summary" begin
    r = Float32[0 1; -1 NaN; 3 0]
    s = residual_summary(r; ref = Float32[2 2; 2 NaN; 2 2])
    @test s.n == 6 && s.n_finite == 5
    @test s.mean ≈ 0.6
    @test s.max_abs == 3
    @test s.mae ≈ 1.0
    @test s.rmse ≈ sqrt(11 / 5)
    @test s.frac_exact ≈ 0.4
    @test s.median_abs_ref == 2
    @test residual_summary(fill(NaN32, 2, 2)) === nothing
    @test residual_summary(r).median_abs_ref === nothing
end

@testset "wflow_output_consistency (synthetic)" begin
    n, T = 4, 5
    layout = (; n_nodes = n, edge_src = CS_SRC, edge_tgt = CS_TGT,
              width = Float32[2, 2, 2, 2], length_m = Float32[10, 10, 10, 10])
    q  = Float32[1 + i + t for i in 1:n, t in 1:T]
    h  = Float32[0.5 + 0.1i for i in 1:n, t in 1:T]
    vars = Dict{String,Any}(v => nothing for v in WFLOW_CONSISTENCY_VARS)
    vars["river_q"]         = q
    vars["river_flow_q_av"] = q
    vars["river_h"]         = h
    vars["river_flow_h"]    = h
    vars["river_flow_qin_av"]   = upstream_sum(q, CS_SRC, CS_TGT)      # same-timestep sum
    vars["river_flow_storage"]  = h .* (layout.width .* layout.length_m)
    vars["river_bc_abstraction"] = zeros(Float32, n, T)

    checks = Dict(c.id => c for c in wflow_output_consistency(vars, layout))
    @test sort(collect(keys(checks))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    @test maximum(abs, checks[1].residual) == 0               # q_av == q
    @test maximum(abs, checks[2].residual) == 0
    @test maximum(abs, checks[9].residual) == 0               # same-timestep convention holds
    @test maximum(abs, checks[8].residual) > 0                # lagged does not
    @test maximum(abs, checks[12].residual) == 0
    @test checks[5].residual == zeros(Float32, n, T)          # ≈ 0 check passes the field through
    @test checks[3].residual === nothing                      # river_inwater absent
    @test checks[4].residual === nothing                      # external inflow absent
    @test checks[7].residual === nothing                      # storage_prev absent
    @test size(checks[8].residual) == (n, T - 1)
end

@testset "river_layout / river_mask_consistency (test model)" begin
    layout = river_layout(CS_STATIC, CS_OUTPUT)
    n = layout.n_nodes
    @test n > 0
    @test length(layout.rows) == length(layout.cols) == length(layout.width) == n
    @test layout.dt == get_timestep(CS_OUTPUT)
    @test all(1 .<= layout.edge_src .<= n) && all(1 .<= layout.edge_tgt .<= n)
    @test length(layout.edge_src) == length(layout.edge_tgt)

    # same node order/topology as preprocessing
    src_raw, tgt_raw = ldd_to_graph(CS_STATIC, "river")
    @test layout.node_ids[layout.edge_src] == src_raw
    @test layout.node_ids[layout.edge_tgt] == tgt_raw

    mc = river_mask_consistency(CS_STATIC, layout)
    @test mc.mask_count == mc.n_nodes == n
    @test mc.n_edges == mc.ldd_to_river
    @test isempty(mc.missed_edges)
    @test mc.ldd_to_river + mc.ldd_to_nonriver + mc.ldd_is_pit + mc.ldd_missing == n

    topo = graph_topology_checks(layout.edge_src, layout.edge_tgt, n)
    @test topo.headwater_ok && topo.single_upstream_mismatches == 0

    NCDataset(CS_OUTPUT, "r") do ds
        q = read_river_series(ds, layout, "river_q")
        @test size(q, 1) == n
        @test size(q, 2) == size(ds["river_q"], 3)
        @test read_river_series(ds, layout, "no_such_variable") === nothing

        # node-order data agrees with the preprocessing node order (column-major ids)
        raw = ds["river_q"][:, :, 1]
        i = 1
        r, c = layout.rows[i], layout.cols[i]
        r_out = layout.dim1_flip ? layout.nrows - r + 1 : r
        c_out = layout.dim2_flip ? layout.ncols - c + 1 : c
        @test q[i, 1] == (ismissing(raw[r_out, c_out]) ? NaN32 : Float32(raw[r_out, c_out])) ||
              (isnan(q[i, 1]) && ismissing(raw[r_out, c_out]))
    end
end

@testset "mb_scheme_residuals on wflow output (test model)" begin
    layout = river_layout(CS_STATIC, CS_OUTPUT)
    NCDataset(CS_OUTPUT, "r") do ds
        q  = read_river_series(ds, layout, "river_q")
        h  = read_river_series(ds, layout, "river_h")
        iw = read_river_series(ds, layout, "river_inwater")
        wl = layout.width .* layout.length_m
        res = Dict(s => mb_scheme_residuals(q, h, iw; edge_src = layout.edge_src,
                                            edge_tgt = layout.edge_tgt, wl, dt = layout.dt,
                                            scheme = s) for s in MB_SCHEMES)
        sums = Dict(s => residual_summary(res[s].residual) for s in MB_SCHEMES)
        @test all(!isnothing, values(sums))
        # wflow routes with a same-timestep upstream sum: the lagged scheme is the worst
        @test sums[:semi_implicit].rmse <= sums[:lagged_upstream].rmse
        @test sums[:fully_implicit].rmse <= sums[:lagged_upstream].rmse
    end
end
