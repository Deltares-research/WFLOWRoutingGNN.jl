using Test
using Flux
using GraphNeuralNetworks
using SparseArrays

# ── Constants derived from DOMAIN_VARS["river"] ────────────────────────────
const GNN_DOMAIN    = "river"
const GNN_VARS      = DOMAIN_VARS[GNN_DOMAIN]
const GNN_N_STATE   = length(GNN_VARS["state"])
const GNN_N_FORCING = length(GNN_VARS["forcing"])
const GNN_N_STATIC  = length(GNN_VARS["static"])
const GNN_IN_DIM    = GNN_N_STATE + GNN_N_FORCING + GNN_N_STATIC
const GNN_HIDDEN    = 16
const GNN_NLAYERS   = 2
const GNN_N_NODES   = 10
const GNN_N_EDGES   = 20

@testset "_activation_name" begin

    @testset "known activations return their name" begin
        @test WflowRoutingGNN._activation_name(swish)    == "swish"
        @test WflowRoutingGNN._activation_name(relu)     == "relu"
        @test WflowRoutingGNN._activation_name(identity) == "identity"
    end

    @testset "unknown activation throws ArgumentError" begin
        @test_throws ArgumentError WflowRoutingGNN._activation_name(x -> x^2)
        @test_throws ArgumentError WflowRoutingGNN._activation_name(cos)
    end

end

@testset "ModelSettings TOML round-trip" begin

    s = ModelSettings(domain          = GNN_DOMAIN,
                      hidden_dim      = GNN_HIDDEN,
                      nlayers         = GNN_NLAYERS,
                      enc_activation  = relu,
                      proc_activation = tanh,
                      mb_theta        = 0.5f0,
                      mb_augment_decoder = true,
                      include_log_upstream_area = true,
                      mb_smooth_h_floor = true,
                      mb_h_floor_softness = 0.02f0,
                      mb_sigma_h_floor = 0.5f0,
                      mb_sigma_q_floor = 0.1f0,
                      mb_sigma_inwater_floor = 0.2f0)

    path = tempname() * ".toml"
    save_model_settings(path, s)
    s2 = load_model_settings(path)
    rm(path)

    @test s2.domain          == s.domain
    @test s2.hidden_dim      == s.hidden_dim
    @test s2.nlayers         == s.nlayers
    @test s2.enc_activation  === s.enc_activation
    @test s2.proc_activation === s.proc_activation
    @test s2.mb_theta        == s.mb_theta
    @test s2.mb_augment_decoder == s.mb_augment_decoder
    @test s2.include_log_upstream_area == s.include_log_upstream_area
    @test s2.mb_smooth_h_floor == s.mb_smooth_h_floor
    @test s2.mb_h_floor_softness == s.mb_h_floor_softness
    @test s2.mb_sigma_h_floor == s.mb_sigma_h_floor
    @test s2.mb_sigma_q_floor == s.mb_sigma_q_floor
    @test s2.mb_sigma_inwater_floor == s.mb_sigma_inwater_floor

end

@testset "WflowGNN construction from ModelSettings" begin

    s = ModelSettings(domain     = GNN_DOMAIN,
                      hidden_dim = GNN_HIDDEN,
                      nlayers    = GNN_NLAYERS)
    m = WflowGNN(s)

    @testset "encoder shape" begin
        @test size(m.encoder.weight) == (GNN_HIDDEN, GNN_IN_DIM)
    end

    @testset "processor has correct number of layers" begin
        @test length(m.processor.layers) == GNN_NLAYERS
    end

    @testset "decoder shape" begin
        @test size(m.decoder.weight) == (GNN_N_STATE, GNN_HIDDEN)
    end

end

@testset "WflowGNN optional log-upstream-area feature" begin
    s = ModelSettings(domain = GNN_DOMAIN,
                      hidden_dim = GNN_HIDDEN,
                      nlayers = GNN_NLAYERS,
                      include_log_upstream_area = true)
    m = WflowGNN(s)

    @test size(m.encoder.weight) == (GNN_HIDDEN, GNN_IN_DIM + 1)

    static = rand(Float32, GNN_N_STATIC + 1, GNN_N_NODES)
    g = rand_graph(GNN_N_NODES, GNN_N_EDGES,
                   ndata = (state   = rand(Float32, GNN_N_STATE,   GNN_N_NODES),
                            forcing = rand(Float32, GNN_N_FORCING, GNN_N_NODES)))
    out = m(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing)
    @test size(out) == (GNN_N_STATE, GNN_N_NODES)
    @test all(isfinite, out)
end

@testset "WflowGNN forward pass" begin

    s = ModelSettings(domain     = GNN_DOMAIN,
                      hidden_dim = GNN_HIDDEN,
                      nlayers    = GNN_NLAYERS)
    m = WflowGNN(s)

    static = rand(Float32, GNN_N_STATIC, GNN_N_NODES)
    g = rand_graph(GNN_N_NODES, GNN_N_EDGES,
                   ndata = (state   = rand(Float32, GNN_N_STATE,   GNN_N_NODES),
                            forcing = rand(Float32, GNN_N_FORCING, GNN_N_NODES)))

    out = m(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing)

    @testset "output has shape (n_state, n_nodes)" begin
        @test size(out) == (GNN_N_STATE, GNN_N_NODES)
    end

    @testset "output is a Float32 array" begin
        @test eltype(out) == Float32
    end

    @testset "output differs from input state (model is not identity)" begin
        @test out != g.ndata.state
    end

end

@testset "WflowGNN mass-balance augmented decoder" begin

    N = GNN_N_NODES
    # Simple chain graph: node i+1 flows into node i (A_routing[i,j]=1 ⇔ j→i).
    src_r = collect(2:N)
    tgt_r = collect(1:(N - 1))
    A_routing = sparse(tgt_r, src_r, ones(Float32, N - 1), N, N)
    # SparseConv adjacency: routing edges + self-loops.
    A_self = sparse(vcat(tgt_r, collect(1:N)), vcat(src_r, collect(1:N)),
                    ones(Float32, (N - 1) + N), N, N)

    pq = fill(2.0f0, N)
    ph = fill(0.5f0, N)
    mb = WflowRoutingGNN.ThetaKinematicWave(
        pq, ph, ph ./ pq,
        fill(0.0f0, N), fill(1.0f0, N),
        fill(0.0f0, N), fill(1.0f0, N),
        fill(0.0f0, N), fill(1.0f0, N),
        86400.0f0,
        A_routing, nothing, 0, 1.0f0,
        false, 0.05f0,
    )

    s_base = ModelSettings(domain = GNN_DOMAIN, hidden_dim = GNN_HIDDEN,
                           nlayers = GNN_NLAYERS)
    s_aug  = ModelSettings(domain = GNN_DOMAIN, hidden_dim = GNN_HIDDEN,
                           nlayers = GNN_NLAYERS, mb_augment_decoder = true)

    m_base = WflowGNN(s_base, mb, A_self)
    m_aug  = WflowGNN(s_aug,  mb, A_self)

    @testset "decoder input widths" begin
        @test size(m_base.decoder.weight, 2) == GNN_HIDDEN
        @test size(m_aug.decoder.weight, 2)  == GNN_HIDDEN + WflowRoutingGNN.MB_DECODER_FEATURES
        @test m_aug.augment_mb
        @test !m_base.augment_mb
    end

    g = rand_graph(N, GNN_N_EDGES,
                   ndata = (state   = rand(Float32, GNN_N_STATE,   N),
                            forcing = rand(Float32, GNN_N_FORCING, N)))
    static = rand(Float32, GNN_N_STATIC, N)

    @testset "mb_decoder_features shapes" begin
        uq, he = WflowRoutingGNN.mb_decoder_features(mb, g, g.ndata.state, g.ndata.forcing)
        @test size(uq) == (1, N)
        @test size(he) == (1, N)
        @test all(isfinite, uq)
        @test all(isfinite, he)
    end

    @testset "augmented forward pass keeps exact mass-balance output shape" begin
        out = m_aug(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing)
        @test size(out) == (GNN_N_STATE, N)
        @test eltype(out) == Float32
        @test all(isfinite, out)
    end

    @testset "mass-balance forward floors propagated q to non-negative" begin
        m_floor = deepcopy(m_base)
        m_floor.decoder.weight .= 0f0
        m_floor.decoder.bias   .= -100f0  # force large negative Δq at every node

        state_in = copy(g.ndata.state)
        state_in[1:1, :] .= 0.1f0
        out = m_floor(g, state_in, g.ndata.forcing, static, g.ndata.forcing)

        # With pq=2, μq=0, σq=1 in this fixture, physical floor at 0 maps to q_norm=0.
        @test all(out[1:1, :] .>= 0f0)
        @test maximum(abs, out[1:1, :]) <= 1f-6
    end

    @testset "gradient flows through the augmented decoder" begin
        gs = gradient(m_aug) do model
            sum(model(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing))
        end
        @test gs[1].decoder.weight !== nothing
        @test all(isfinite, gs[1].decoder.weight)
    end

end

@testset "mass-balance formulation hierarchy" begin

    N = GNN_N_NODES
    src_r = collect(2:N)
    tgt_r = collect(1:(N - 1))
    A_routing = sparse(tgt_r, src_r, ones(Float32, N - 1), N, N)
    A_self = sparse(vcat(tgt_r, collect(1:N)), vcat(src_r, collect(1:N)),
                    ones(Float32, (N - 1) + N), N, N)

    ph = fill(0.5f0, N)
    pq = fill(2.0f0, N)
    mb = WflowRoutingGNN.ThetaKinematicWave(
        pq, ph, ph ./ pq,
        fill(0.0f0, N), fill(1.0f0, N),
        fill(0.0f0, N), fill(1.0f0, N),
        fill(0.0f0, N), fill(1.0f0, N),
        86400.0f0, A_routing, nothing, 0, 1.0f0, false, 0.05f0)

    s = ModelSettings(domain = GNN_DOMAIN, hidden_dim = GNN_HIDDEN, nlayers = GNN_NLAYERS)
    g = rand_graph(N, GNN_N_EDGES,
                   ndata = (state   = rand(Float32, GNN_N_STATE,   N),
                            forcing = rand(Float32, GNN_N_FORCING, N)))
    static = rand(Float32, GNN_N_STATIC, N)
    nomb   = WflowRoutingGNN.NoMassBalance()

    @testset "type hierarchy" begin
        @test nomb isa WflowRoutingGNN.MassBalanceFormulation
        @test mb   isa WflowRoutingGNN.MassBalanceFormulation
        @test WflowRoutingGNN.MassBalanceLayer === WflowRoutingGNN.ThetaKinematicWave
        @test !WflowRoutingGNN.has_mass_balance(nomb)
        @test WflowRoutingGNN.has_mass_balance(mb)
    end

    @testset "models never hold `nothing`" begin
        @test WflowGNN(s).mass_balance isa WflowRoutingGNN.NoMassBalance
        @test WflowGNN(s, A_self).mass_balance isa WflowRoutingGNN.NoMassBalance
        @test WflowGNN(s, mb).mass_balance === mb
        @test WflowGNN(s, nomb, A_self).mass_balance isa WflowRoutingGNN.NoMassBalance
    end

    @testset "dispatch interface" begin
        @test WflowRoutingGNN.decoder_outputs(nomb, GNN_N_STATE) == GNN_N_STATE
        @test WflowRoutingGNN.decoder_outputs(mb, GNN_N_STATE)   == 1
        @test WflowRoutingGNN.extra_decoder_dim(nomb) == 0
        @test WflowRoutingGNN.extra_decoder_dim(mb)   == WflowRoutingGNN.MB_DECODER_FEATURES
        @test WflowRoutingGNN.extra_decoder_features(nomb, g, g.ndata.state, g.ndata.forcing) == ()
        feats = WflowRoutingGNN.extra_decoder_features(mb, g, g.ndata.state, g.ndata.forcing)
        @test length(feats) == WflowRoutingGNN.MB_DECODER_FEATURES

        raw = randn(Float32, 1, N)
        @test WflowRoutingGNN.floor_state(nomb, g, raw) === raw
        @test WflowRoutingGNN.floor_state(mb, g, raw) ==
              WflowRoutingGNN._floor_q_norm_nonnegative(mb, g, raw)
        # pq=2, μq=0, σq=1 ⇒ physical floor at 0 is normalised 0
        @test all(WflowRoutingGNN.floor_state(mb, g, raw) .>= 0f0)

        state = g.ndata.state
        forcing = g.ndata.forcing
        Δ2 = randn(Float32, GNN_N_STATE, N)
        @test WflowRoutingGNN.apply_formulation(nomb, g, state, forcing, forcing, Δ2) == state .+ Δ2
        Δ1 = randn(Float32, 1, N)
        out = WflowRoutingGNN.apply_formulation(mb, g, state, forcing, forcing, Δ1)
        q_new = WflowRoutingGNN.floor_state(mb, g, state[1:1, :] .+ Δ1)
        @test out == vcat(q_new, mb(g, state, forcing, forcing, q_new))
    end

    @testset "NoMassBalance forward is the residual update" begin
        m = WflowGNN(s, A_self)
        @test size(m.decoder.weight, 1) == GNN_N_STATE
        out = m(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing)
        h = m.processor(g, m.encoder(vcat(g.ndata.state, g.ndata.forcing, static)))
        @test out == g.ndata.state .+ m.decoder(h)
    end

    @testset "mb_augment_decoder is ignored without extra decoder features" begin
        s_aug = ModelSettings(domain = GNN_DOMAIN, hidden_dim = GNN_HIDDEN,
                              nlayers = GNN_NLAYERS, mb_augment_decoder = true)
        m = WflowGNN(s_aug, A_self)
        @test !m.augment_mb
        @test size(m.decoder.weight, 2) == GNN_HIDDEN
    end

    @testset "NoMassBalance is field-less and device/batch transparent" begin
        m = WflowGNN(s, A_self)
        @test fieldcount(typeof(m.mass_balance)) == 0
        @test Flux.cpu(m).mass_balance isa WflowRoutingGNN.NoMassBalance
        @test precompute_batched(m, 3).mass_balance isa WflowRoutingGNN.NoMassBalance
        @test isempty(Flux.trainable(m.mass_balance))
    end

    @testset "optimiser setup and gradients work with NoMassBalance" begin
        m = WflowGNN(s, A_self)
        opt = Flux.setup(Flux.Adam(1f-3), m)
        gs = gradient(mm -> sum(mm(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing)), m)
        @test all(isfinite, gs[1].decoder.weight)
        Flux.update!(opt, m, gs[1])
        @test all(isfinite, m.decoder.weight)
    end

    @testset "old checkpoints (mass_balance = nothing) still load" begin
        m_src = WflowGNN(s, A_self)
        m_dst = WflowGNN(s, A_self)
        st = Flux.state(m_src)
        st_old = merge(st, (mass_balance = nothing,))
        Flux.loadmodel!(m_dst, st_old)
        @test m_dst.decoder.weight == m_src.decoder.weight
        @test m_dst.encoder.weight == m_src.encoder.weight
        # Current-format state loads too.
        m_dst2 = WflowGNN(s, A_self)
        Flux.loadmodel!(m_dst2, st)
        @test m_dst2.decoder.weight == m_src.decoder.weight
    end

    @testset "θ-kinematic-wave state loads round-trip" begin
        m_src = WflowGNN(s, mb, A_self)
        m_dst = WflowGNN(s, mb, A_self)
        Flux.loadmodel!(m_dst, Flux.state(m_src))
        @test m_dst.decoder.weight == m_src.decoder.weight
    end

end

@testset "mb_formulation setting" begin

    @testset "mb_formulation_kind mapping" begin
        k(; kw...) = mb_formulation_kind(ModelSettings(; domain = "river", kw...))
        @test k() == :theta_kinematic_wave
        @test k(enforce_mass_balance = false) == :none
        @test k(mb_formulation = "none") == :none
        @test k(mb_formulation = "theta_kinematic_wave") == :theta_kinematic_wave
        @test k(mb_formulation = "none", enforce_mass_balance = false) == :none
        @test mb_formulation_kind(ModelSettings(domain = "land")) == :none
        @test_throws ArgumentError k(mb_formulation = "bogus")
    end

    @testset "TOML round-trip and back-compat default" begin
        s = ModelSettings(domain = GNN_DOMAIN, mb_formulation = "none")
        path = tempname() * ".toml"
        save_model_settings(path, s)
        @test load_model_settings(path).mb_formulation == "none"
        # A settings file written before the key existed defaults to the θ-kinematic wave.
        d = WflowRoutingGNN.TOML.parsefile(path)
        delete!(d, "mb_formulation")
        open(path, "w") do io
            WflowRoutingGNN.TOML.print(io, d)
        end
        @test load_model_settings(path).mb_formulation == "theta_kinematic_wave"
        rm(path)
    end

end
