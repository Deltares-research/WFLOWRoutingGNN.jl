using Test
using Flux
using GraphNeuralNetworks

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
                      proc_activation = tanh)

    path = tempname() * ".toml"
    save_model_settings(path, s)
    s2 = load_model_settings(path)
    rm(path)

    @test s2.domain          == s.domain
    @test s2.hidden_dim      == s.hidden_dim
    @test s2.nlayers         == s.nlayers
    @test s2.enc_activation  === s.enc_activation
    @test s2.proc_activation === s.proc_activation

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

    g = rand_graph(GNN_N_NODES, GNN_N_EDGES,
                   ndata = (state   = rand(Float32, GNN_N_STATE,   GNN_N_NODES),
                            forcing = rand(Float32, GNN_N_FORCING, GNN_N_NODES),
                            static  = rand(Float32, GNN_N_STATIC,  GNN_N_NODES)))

    out = m(g)

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
