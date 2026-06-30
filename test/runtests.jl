using Test
using WflowRoutingGNN

@testset "Tests for WflowRoutingGNN" begin
    include("test_preprocess.jl")
    include("test_gnn.jl")
    include("test_strategy.jl")
    include("test_training.jl")
end