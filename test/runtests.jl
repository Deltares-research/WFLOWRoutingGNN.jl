using Test
using WflowRoutingGNN

@testset "Tests for WflowRoutingGNN" begin
    include("test_preprocess.jl")
    include("test_gnn.jl")
    include("test_hparsearch.jl")
    include("test_strategy.jl")
    include("test_lr_autotune.jl")
    include("test_consistency.jl")
    include("test_training.jl")
    include("test_pipeline.jl")
    include("test_analysis.jl")
    include("test_attribution.jl")
    include("test_stages.jl")
end