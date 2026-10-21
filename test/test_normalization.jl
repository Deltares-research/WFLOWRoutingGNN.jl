using Test
using Flux
using GraphNeuralNetworks
using Statistics
using WflowRoutingGNN

# Selectable nonlinear input scaling: scheme maths, fitted stats, the encoder-only
# layer, config/persistence and the mass-balance-unchanged guarantee.

const NZ_STATICMAPS = joinpath(@__DIR__, "..", "test_data", "test_model", "staticmaps.nc")
const NZ_OUTPUT     = joinpath(@__DIR__, "..", "test_data", "test_model", "run_default", "output.nc")

@testset "normalization schemes" begin
    @testset "forward/inverse round trip" begin
        x = Float32[-3, -0.5, 0, 0.2, 1, 40]
        for sc in WflowRoutingGNN.NORMALIZATION_SCHEMES
            xs = sc === :log1p_zscore ? abs.(x) : x
            y  = nonlinear_forward.(sc, xs, 0.7f0)
            @test all(isfinite, y)
            @test nonlinear_inverse.(sc, y, 0.7f0) ≈ xs rtol = 1f-4 atol = 1f-5
        end
        @test nonlinear_forward(:log1p_zscore, -2f0, 1f0) == 0f0      # negatives clamp
        @test nonlinear_forward(:signed_log_zscore, -2f0, 1f0) ≈ -nonlinear_forward(:signed_log_zscore, 2f0, 1f0)
        @test_throws ArgumentError nonlinear_forward(:nope, 1f0, 1f0)
        @test_throws ArgumentError nonlinear_inverse(:nope, 1f0, 1f0)
    end

    @testset "parse_normalization" begin
        p = parse_normalization(Dict("river_inwater" => "asinh_zscore", "river_q" => "zscore",
                                     "river_h" => :log1p_zscore), "river")
        @test p == Dict("river_inwater" => :asinh_zscore, "river_h" => :log1p_zscore)  # zscore dropped
    end

    @testset "parse_normalization rejects bad input" begin
        @test_throws ArgumentError parse_normalization(Dict("river_inwater" => "sqrt"), "river")
        @test_throws ArgumentError parse_normalization(Dict("not_a_var" => "asinh_zscore"), "river")
        @test_throws ArgumentError parse_normalization(Dict("river_length" => "asinh_zscore"), "river")  # static
        @test parse_normalization(Dict{String,String}(), "river") == Dict{String,Symbol}()
    end

    @testset "nonlinear_stats" begin
        fit = abs.(randn(Float32, 5, 400)) .* 1f-8
        fit[2, 1:50] .= 0f0                           # zeros are ignored for the scale
        fit[3, 7]    = NaN32
        for sc in WflowRoutingGNN.NORMALIZATION_SCHEMES
            sc === :zscore && continue
            s, μ, σ = nonlinear_stats(sc, fit; per_node = true)
            @test length(s) == length(μ) == length(σ) == 5
            @test all(>(0), s) && all(>(0), σ)
            y = [nonlinear_forward(sc, v, s[3]) for v in fit[3, :] if isfinite(v)]
            @test mean(y) ≈ μ[3] rtol = 1f-4
            @test std(y)  ≈ σ[3] rtol = 1f-4
            sg, μg, σg = nonlinear_stats(sc, fit; per_node = false)
            @test sg isa Float32 && μg isa Float32 && σg isa Float32
        end
        # Degenerate inputs keep the defaults (no NaN/zero scale).
        s0, μ0, σ0 = nonlinear_stats(:asinh_zscore, zeros(Float32, 2, 10); per_node = true)
        @test s0 == [1f0, 1f0] && μ0 == [0f0, 0f0] && σ0 == [1f0, 1f0]
    end
end

# One shared build of the plain and the nonlinear-inwater preprocessing.
const NZ_NORM = Dict("river_inwater" => "asinh_zscore", "river_q" => "signed_log_zscore")
const NZ_PLAIN = build_wflow_graph(NZ_STATICMAPS, NZ_OUTPUT, "river")
const NZ_NL    = build_wflow_graph(NZ_STATICMAPS, NZ_OUTPUT, "river"; normalization = NZ_NORM)

@testset "preprocessing leaves the affine data untouched" begin
    gp, sp, _, pp, stat_p = NZ_PLAIN
    gn, sn, _, pn, stat_n = NZ_NL
    @test length(gp) == length(gn)
    @test all(i -> gp[i].ndata.state == gn[i].ndata.state &&
                   gp[i].ndata.forcing == gn[i].ndata.forcing, eachindex(gp))
    @test stat_p == stat_n
    @test pp == pn
    for v in keys(sp)
        @test sp[v].mean == sn[v].mean
        @test sp[v].std  == sn[v].std
    end
    @test !any(s -> haskey(s, :scheme), values(sp))
    @test sn["river_inwater"].scheme === :asinh_zscore
    @test sn["river_q"].scheme === :signed_log_zscore
    @test !haskey(sn["river_h"], :scheme)
    n = gp[1].num_nodes
    for v in ("river_inwater", "river_q")        # per-node vectors
        @test length(sn[v].scale) == length(sn[v].nl_mean) == length(sn[v].nl_std) == n
    end
    @test_throws ArgumentError build_wflow_graph(NZ_STATICMAPS, NZ_OUTPUT, "river";
                                                 normalization = Dict("river_inwater" => "bad"))
end

@testset "EncoderInputTransform" begin
    graphs, norm_stats, _, postscale, static_arr = NZ_NL
    g0 = graphs[1]
    n  = g0.num_nodes
    ms = ModelSettings(domain = "river", normalization = Dict{String,String}(NZ_NORM))
    tf = encoder_input_transform(ms, norm_stats, n)
    @test tf isa EncoderInputTransform
    @test encoder_input_transform(ModelSettings(domain = "river"), NZ_PLAIN[2], n) === nothing

    vars = DOMAIN_VARS["river"]
    tv   = vcat(vars["state"], vars["forcing"])
    nrow = length(tv) + length(vars["static"])
    X = vcat(g0.ndata.state, g0.ndata.forcing, static_arr)
    Y = tf(X)
    @test size(Y) == size(X)
    @test all(isfinite, Y)

    # Untouched rows pass through; selected rows follow the documented formula.
    sel = [findfirst(==(v), tv) for v in keys(NZ_NORM)]
    other = setdiff(1:nrow, sel)
    @test Y[other, :] == X[other, :]
    for (v, sc) in (("river_inwater", :asinh_zscore), ("river_q", :signed_log_zscore))
        i, st = findfirst(==(v), tv), norm_stats[v]
        phys = X[i, :] .* st.std .+ st.mean
        ref  = (nonlinear_forward.(sc, phys, st.scale) .- st.nl_mean) ./ st.nl_std
        @test Y[i, :] ≈ ref rtol = 1f-4 atol = 1f-5
    end

    # Batched input (B·n columns) == per-graph application, repeated.
    B = 3
    @test tf(repeat(X, 1, B)) ≈ repeat(Y, 1, B)
    @test_throws DimensionMismatch tf(X[:, 1:n-1])

    # Non-trainable, but differentiable w.r.t. its input.
    @test isempty(Flux.trainables(tf))
    gx = Flux.gradient(x -> sum(abs2, tf(x)), X)[1]
    @test size(gx) == size(X) && all(isfinite, gx)
end

@testset "model: encoder-only, mass balance unchanged" begin
    bs = 2
    graphs, ns_plain, _, ps, static_arr = NZ_PLAIN
    _, ns_nl, _, _, _ = NZ_NL
    ms_plain = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 2)
    ms_nl    = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 2,
                             normalization = Dict{String,String}(NZ_NORM))
    m_plain = build_gnn_model(ms_plain, graphs, ns_plain, ps, NZ_OUTPUT, bs)
    m_nl    = build_gnn_model(ms_nl,    graphs, ns_nl,    ps, NZ_OUTPUT, bs)

    @test m_plain.encoder isa Union{Dense, Chain} && !(m_plain.encoder isa Chain &&
              m_plain.encoder[1] isa EncoderInputTransform)         # default model unchanged
    @test m_nl.encoder isa Chain && m_nl.encoder[1] isa EncoderInputTransform
    # The mass-balance layer is built from the affine stats only: identical constants.
    @test Flux.state(m_plain.mass_balance) == Flux.state(m_nl.mass_balance)

    g0 = graphs[1]
    out = m_nl(g0, g0.ndata.state, g0.ndata.forcing, static_arr, g0.ndata.forcing)
    @test size(out, 2) == g0.num_nodes && all(isfinite, out)

    # Checkpoint round trip (Flux.state -> loadmodel!) into a freshly built model.
    fresh = build_gnn_model(ms_nl, graphs, ns_nl, ps, NZ_OUTPUT, bs)
    Flux.loadmodel!(fresh, Flux.state(Flux.cpu(m_nl)))
    out2 = fresh(g0, g0.ndata.state, g0.ndata.forcing, static_arr, g0.ndata.forcing)
    @test out2 == out
end

@testset "normalization config + persistence" begin
    s = ModelSettings(domain = "river", normalization = Dict("river_inwater" => "log1p_zscore"))
    path = tempname() * ".toml"
    save_model_settings(path, s)
    s2 = load_model_settings(path)
    @test s2.normalization == Dict("river_inwater" => "log1p_zscore")
    @test occursin("river_inwater=log1p_zscore", sprint(show, s))

    # Plain settings omit the table; a legacy file reloads with plain z-score.
    save_model_settings(path, ModelSettings(domain = "river"))
    @test !haskey(WflowRoutingGNN.TOML.parsefile(path), "normalization")
    @test load_model_settings(path).normalization == Dict{String,String}()
    # A saved file with an invalid scheme is rejected at load.
    d = WflowRoutingGNN.TOML.parsefile(path)
    d["normalization"] = Dict("river_inwater" => "sqrt")
    open(path, "w") do io
        WflowRoutingGNN.TOML.print(io, d)
    end
    @test_throws ArgumentError load_model_settings(path)
    rm(path)

    cfg = Dict{String,Any}(
        "data"  => Dict{String,Any}("run_name" => "x", "runs_dir" => "r", "wflow_model_path" => "m",
                                    "train_frac" => 0.6, "val_frac" => 0.2),
        "model" => Dict{String,Any}("domain" => "river"),
        "train" => Dict{String,Any}("epochs" => 1, "batch_size" => 1, "lr_start" => 1e-3, "lr_final" => 1e-4),
    )
    _, ms0, _ = WflowRoutingGNN.settings_from_config(cfg, pwd())
    @test isempty(ms0.normalization)
    # `_set!` creates the missing [model.normalization] table (hparsearch key).
    WflowRoutingGNN._set!(cfg, "model.normalization.river_inwater", "asinh_zscore")
    _, ms1, _ = WflowRoutingGNN.settings_from_config(cfg, pwd())
    @test ms1.normalization == Dict("river_inwater" => "asinh_zscore")
    WflowRoutingGNN._set!(cfg, "model.normalization.river_inwater", "bogus")
    @test_throws ArgumentError WflowRoutingGNN.settings_from_config(cfg, pwd())
end
