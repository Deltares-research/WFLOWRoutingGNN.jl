using Test
using Flux
using GraphNeuralNetworks
using SparseArrays
using LinearAlgebra
using Random
using MLUtils
using TOML

# ── Muskingum-continuity hybrid (reservoir backbone + continuity readout) ─────────────

# Small branching river: 2→1, 3→1, 4→2, 5→2, 6→3, 7→6, 8→7  (node 1 = outlet).
const HY_N   = 8
const HY_SRC = [2, 3, 4, 5, 6, 7, 8]
const HY_TGT = [1, 1, 2, 2, 3, 6, 7]
const HY_A   = sparse(HY_TGT, HY_SRC, ones(Float32, length(HY_SRC)), HY_N, HY_N)
const HY_DT  = 86400.0f0

hy_vars()    = DOMAIN_VARS["river"]
hy_n_state() = length(hy_vars()["state"])

function hy_inputs(; N = HY_N)
    a  = Float32.(range(50, 400; length = N)) .* 1f6          # upstream area [m²]
    ph = fill(0.5f0, N) .* a ./ 1f5                            # a/(w·l)
    stats(μ, σ) = (mean = μ isa Number ? fill(Float32(μ), N) : Float32.(μ),
                   std  = fill(Float32(σ), N))
    # K_i = μ_h/μ_q ranges over ≈ 0.1 … 2.3 timesteps
    norm_stats = Dict("river_q" => stats(2f-6, 1f-6),
                      "river_h" => stats(range(0.02, 0.4; length = N), 0.03),
                      "river_inwater" => stats(5f-6, 1f-6))
    postscale = Dict("river_q" => a, "river_h" => ph, "river_inwater" => a)
    return norm_stats, postscale
end

function hy_build(; bound = 0.5, smooth = false)
    norm_stats, postscale = hy_inputs()
    return build_muskingum_continuity(norm_stats, postscale, HY_SRC, HY_TGT, HY_DT;
                                      residual_bound = bound, smooth_h_floor = smooth,
                                      h_floor_softness = 0.05)
end

function hy_model(mb)
    ms = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1,
                       mb_formulation = "muskingum_continuity")
    A_self = sparse(vcat(HY_TGT, 1:HY_N), vcat(HY_SRC, 1:HY_N),
                    ones(Float32, length(HY_SRC) + HY_N), HY_N, HY_N)
    return WflowGNN(ms, mb, A_self)
end

hy_graph(N = HY_N) = rand_graph(N, 2N)

@testset "muskingum-continuity hybrid" begin

    @testset "inverse operators" begin
        N = HY_N
        λ = Float32.(range(0.5, 4; length = N))
        P = WflowRoutingGNN._tree_inverse_operator(HY_SRC, HY_TGT, N, λ, 1f0 .+ λ)
        L = Diagonal(1f0 .+ λ) - HY_A * Diagonal(λ)
        @test Matrix(L) * Matrix(P) ≈ Matrix{Float32}(I, N, N) atol = 1f-5
        @test all(≥(0f0), nonzeros(P))
        @test diag(Matrix(P)) ≈ 1f0 ./ (1f0 .+ λ)

        R = WflowRoutingGNN._tree_inverse_operator(HY_SRC, HY_TGT, N, ones(Float32, N), ones(Float32, N))
        @test Matrix(R) * Matrix(I - HY_A) ≈ Matrix{Float32}(I, N, N) atol = 1f-6
        # R[i, j] = 1 exactly when j is i or upstream of i
        @test R[1, 8] == 1f0 && R[8, 1] == 0f0 && R[2, 3] == 0f0

        @test_throws ArgumentError WflowRoutingGNN._tree_inverse_operator(
            [1, 2], [2, 1], 2, ones(Float32, 2), ones(Float32, 2))
    end

    @testset "contraction: spectrum below 1, powers decay" begin
        mb = hy_build()
        c  = backbone_contraction(mb)
        @test all(0 .< c .< 1)
        @test diag(Matrix(mb.P)) ≈ c
        # P is triangular in topological order: its eigenvalues are its diagonal.
        ev = eigvals(Matrix(mb.P))
        @test maximum(abs, ev) ≈ maximum(c) atol = 1f-4
        @test norm(Matrix(mb.P)^200) < 1f-6
    end

    @testset "build aborts on invalid settings" begin
        @test_throws ArgumentError hy_build(bound = -0.1)
        @test_throws ArgumentError hy_build(bound = NaN)
    end

    @testset "formulation selection and settings round trip" begin
        ms = ModelSettings(domain = "river", mb_formulation = "muskingum_continuity",
                           mb_residual_bound = 0.25f0)
        @test mb_formulation_kind(ms) == :muskingum_continuity
        path = tempname() * ".toml"
        save_model_settings(path, ms)
        ms2 = load_model_settings(path)
        rm(path)
        @test ms2.mb_formulation == "muskingum_continuity"
        @test ms2.mb_residual_bound == 0.25f0
        @test ModelSettings(domain = "river").mb_residual_bound == 0.5f0   # default
        @test ModelSettings(domain = "river", enforce_mass_balance = false,
                            mb_formulation = "muskingum_continuity") |> mb_formulation_kind == :none
    end

    @testset "interface" begin
        mb = hy_build()
        @test mb isa WflowRoutingGNN.MassBalanceFormulation
        @test WflowRoutingGNN.has_mass_balance(mb)
        @test WflowRoutingGNN.decoder_outputs(mb, 2) == 1
        @test WflowRoutingGNN.extra_decoder_dim(mb) == 0
        @test WflowRoutingGNN.mb_h_loss_weight(mb, :absolute) === nothing
        model = hy_model(mb)
        @test size(model.decoder.weight, 1) == 1
        @test Flux.trainable(mb) == (;)   # physics constants, not optimised
    end

    @testset "continuity holds exactly for the readout" begin
        mb = hy_build()
        g  = hy_graph()
        N  = HY_N
        ns, ps = hy_inputs()
        pq, ph = ps["river_q"]', ps["river_h"]'
        state   = vcat(randn(Float32, 1, N), 1f0 .+ 0.2f0 .* randn(Float32, 1, N))
        fnext   = ones(Float32, length(hy_vars()["forcing"]), N)
        Δ       = 0.3f0 .* randn(Float32, 1, N)
        out     = mb(g, state, fnext, Δ)
        @test size(out) == (2, N)
        # denormalise
        μq, σq = ns["river_q"].mean', ns["river_q"].std'
        μh, σh = ns["river_h"].mean', ns["river_h"].std'
        μi, σi = ns["river_inwater"].mean', ns["river_inwater"].std'
        S0 = pq .* (state[2:2, :] .* σh .+ μh)
        S1 = pq .* (out[2:2, :] .* σh .+ μh)
        Q  = pq .* (out[1:1, :] .* σq .+ μq)
        I1 = (fnext[1:1, :] .* σi .+ μi) .* pq
        up = (HY_A * Q')'
        resid = (S1 .- S0) ./ HY_DT .- (up .+ I1 .- Q)
        # nodes whose whole upstream cone is unfloored satisfy continuity to rounding
        Rm = Matrix(WflowRoutingGNN._tree_inverse_operator(
            HY_SRC, HY_TGT, N, ones(Float32, N), ones(Float32, N)))
        scale = maximum(abs, I1) + maximum(abs, Q)
        cone_ok = [all(Q[1, j] > 0 for j in 1:N if Rm[i, j] != 0) for i in 1:N]
        @test any(cone_ok)
        @test maximum(abs, resid[1, cone_ok]) / scale < 1f-4
    end

    @testset "zero residual bound reproduces the pure reservoir backbone" begin
        mb = hy_build(bound = 0.0)
        g  = hy_graph()
        N  = HY_N
        ns, ps = hy_inputs()
        pq = ps["river_q"]'
        # positive storage and inflow so the floors are inactive
        state = vcat(zeros(Float32, 1, N), 2f0 .* ones(Float32, 1, N))
        fnext = ones(Float32, length(hy_vars()["forcing"]), N)
        out   = mb(g, state, fnext, randn(Float32, 1, N))   # Δ ignored when bound = 0
        μq, σq = ns["river_q"].mean', ns["river_q"].std'
        μh, σh = ns["river_h"].mean', ns["river_h"].std'
        S1 = pq .* (out[2:2, :] .* σh .+ μh)
        Q  = pq .* (out[1:1, :] .* σq .+ μq)
        @test all(Q .> 0)
        # linear reservoir: S_i = K_i · Q_i
        @test vec(S1) ≈ vec(Q) .* mb.K rtol = 1f-3
    end

    @testset "q is a readout: the encoder never sees the q row" begin
        mb    = hy_build()
        model = hy_model(mb)
        g     = hy_graph()
        N     = HY_N
        static  = randn(Float32, length(hy_vars()["static"]), N)
        forcing = randn(Float32, length(hy_vars()["forcing"]), N)
        state   = randn(Float32, 2, N)
        state2  = copy(state); state2[1, :] .+= 100f0         # change only the q row
        @test model(g, state, forcing, static, forcing) == model(g, state2, forcing, static, forcing)
        gq = Flux.gradient(s -> sum(model(g, s, forcing, static, forcing)), state)[1]
        @test all(iszero, gq[1, :])
        @test any(!iszero, gq[2, :])
    end

    @testset "bounded state under saturated residuals (BIBO)" begin
        mb    = hy_build()
        model = hy_model(mb)
        g     = hy_graph()
        N     = HY_N
        static  = randn(Float32, length(hy_vars()["static"]), N)
        forcing = abs.(randn(Float32, length(hy_vars()["forcing"]), N))
        # drive the decoder to huge outputs: the residual must stay bounded by tanh
        big = fmap(p -> p isa AbstractArray ? 50f0 .* p : p, model.decoder)
        model = WflowGNN(model.encoder, model.processor, big, model.mass_balance, model.augment_mb)
        state = vcat(zeros(Float32, 1, N), zeros(Float32, 1, N))
        for _ in 1:400
            state = model(g, state, forcing, static, forcing)
        end
        @test all(isfinite, state)
        @test maximum(abs, state[2, :]) < 1f3
    end

    @testset "batched forward matches per-graph forward" begin
        mb    = hy_build()
        B     = 3
        mbB   = precompute_batched(mb, B)
        @test mbB.batch_size == B
        @test size(mbB.P_batched) == (B * HY_N, B * HY_N)
        g1    = hy_graph()
        gb    = MLUtils.batch(fill(g1, B))
        nf    = length(hy_vars()["forcing"])
        state = randn(Float32, 2, B * HY_N)
        fnext = randn(Float32, nf, B * HY_N)
        Δ     = randn(Float32, 1, B * HY_N)
        ob    = mbB(gb, state, fnext, Δ)
        for k in 1:B
            cols = ((k - 1) * HY_N + 1):(k * HY_N)
            @test ob[:, cols] ≈ mb(g1, state[:, cols], fnext[:, cols], Δ[:, cols]) rtol = 1f-4
        end
        # reshape fallback (no precomputed block-diagonal)
        @test mb(gb, state, fnext, Δ) ≈ ob rtol = 1f-4
    end

    @testset "training step and checkpoint round trip" begin
        mb    = hy_build()
        model = hy_model(mb)
        g     = hy_graph()
        N     = HY_N
        static  = randn(Float32, length(hy_vars()["static"]), N)
        forcing = abs.(randn(Float32, length(hy_vars()["forcing"]), N))
        state   = randn(Float32, 2, N)
        target  = randn(Float32, 2, N)
        loss, grads = Flux.withgradient(m -> Flux.mse(m(g, state, forcing, static, forcing), target), model)
        @test isfinite(loss)
        @test all(isfinite, grads[1].decoder.weight)
        @test any(!iszero, grads[1].decoder.weight)

        model2 = hy_model(hy_build())
        Flux.loadmodel!(model2, Flux.state(model))
        o1 = model(g, state, forcing, static, forcing)
        o2 = model2(g, state, forcing, static, forcing)
        @test o1 == o2
    end
end

@testset "formulation-agnostic stability headline (rollout_contraction)" begin
    N      = HY_N
    nf     = length(hy_vars()["forcing"])
    static = randn(Float32, length(hy_vars()["static"]), N)
    mkg(; state = randn(Float32, 2, N)) =
        rand_graph(N, 2N; ndata = (; state, forcing = randn(Float32, nf, N)))
    batch = [mkg(), mkg()]
    rc    = WflowRoutingGNN.rollout_contraction

    @testset "carried state and roles per formulation" begin
        mb = hy_build()
        @test WflowRoutingGNN.carried_state_rows(mb, 2) == 2:2
        @test WflowRoutingGNN.carried_state_rows(WflowRoutingGNN.NoMassBalance(), 2) == 1:2
        @test WflowRoutingGNN.mb_state_roles(mb) == (; state = "river_h", readout = "river_q")
        @test WflowRoutingGNN.mb_state_roles(WflowRoutingGNN.NoMassBalance()).state == "all"
        @test WflowRoutingGNN.MB_STATE_ROLES[:theta_kinematic_wave] ==
              (; state = "river_q", readout = "river_h")
    end

    @testset "finite, non-negative, deterministic, RNG-free" begin
        for mbf in (hy_build(), WflowRoutingGNN.NoMassBalance())
            ms = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1,
                               mb_formulation = mbf isa WflowRoutingGNN.NoMassBalance ?
                                                "none" : "muskingum_continuity")
            A_self = sparse(vcat(HY_TGT, 1:HY_N), vcat(HY_SRC, 1:HY_N),
                            ones(Float32, length(HY_SRC) + HY_N), HY_N, HY_N)
            model = WflowGNN(ms, mbf, A_self)
            ρ = rc(model, batch, static)
            @test ρ isa Float32 && isfinite(ρ) && ρ ≥ 0f0
            Random.seed!(1); a = rc(model, batch, static)
            Random.seed!(2); b = rc(model, batch, static)
            @test a == b == ρ
        end
    end

    @testset "matches the analytic backbone gain when the residual is off" begin
        # bound = 0 and an inactive depth floor: h_new = D⁻¹ P D h (+ const), D = diag(pq·σ_h),
        # so ρ̂ = ‖D⁻¹ P D δ‖ / ‖δ‖ for the injected pattern δ.
        mb     = hy_build(bound = 0.0)
        model  = hy_model(mb)
        ns, ps = hy_inputs()
        state  = vcat(randn(Float32, 1, N), fill(3f0, 1, N))
        g1 = rand_graph(N, 2N; ndata = (; state, forcing = ones(Float32, nf, N)))
        g2 = rand_graph(N, 2N; ndata = (; state, forcing = ones(Float32, nf, N)))
        δ  = 1f-2 .* Float32[isodd((UInt32(i) * 0x9e3779b1) >> 15) ? 1 : -1 for i in 1:N]
        D  = ps["river_q"] .* ns["river_h"].std
        expected = norm((Matrix(mb.P) * (D .* δ)) ./ D) / norm(δ)
        @test rc(model, [g1, g2], static) ≈ expected rtol = 1f-3
        @test expected > 0
    end
end