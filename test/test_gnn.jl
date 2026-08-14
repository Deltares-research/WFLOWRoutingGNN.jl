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
                      mb_augment_decoder = true)

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
    mb = WflowRoutingGNN.MassBalanceLayer(
        pq, ph, ph ./ pq,
        0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 86400.0f0,
        A_routing, nothing, 0, 1.0f0,
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

    @testset "gradient flows through the augmented decoder" begin
        gs = gradient(m_aug) do model
            sum(model(g, g.ndata.state, g.ndata.forcing, static, g.ndata.forcing))
        end
        @test gs[1].decoder.weight !== nothing
        @test all(isfinite, gs[1].decoder.weight)
    end

end
