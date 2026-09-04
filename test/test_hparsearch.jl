using Test
using TOML
using WflowRoutingGNN

@testset "hparsearch config parsing" begin
    mktempdir() do tmp
        cfg = Dict{String, Any}(
            "data" => Dict{String, Any}(
                "run_name" => "hps",
                "runs_dir" => "runs",
                "wflow_model_path" => "test_data/test_model",
                "train_frac" => 0.6,
                "val_frac" => 0.2,
            ),
            "model" => Dict{String, Any}(
                "domain" => "river",
                "hidden_dim" => 32,
                "nlayers" => 2,
            ),
            "train" => Dict{String, Any}(
                "epochs" => 4,
                "batch_size" => 2,
                "lr_start" => 1e-3,
                "lr_final" => 1e-4,
                # Intentionally omit lr_steps to verify fallback stays in sync.
                "grad_clip" => 1.0,
                "lr_warmup_epochs" => 2,
                "lr_peak_decay" => 0.8,
                "strategy" => Dict{String, Any}(
                    "steps" => [1, 2],
                    "durations" => [2, 2],
                    "noise_scale" => 0.0,
                    "loss_type" => "huber",
                    "peak_delta" => 0.7,
                    "peak_lambda" => 0.0,
                    "peak_gamma" => 1.5,
                    "peak_w_max" => 3.0,
                ),
            ),
            "hparsearch" => Dict{String, Any}(
                "search_type" => "box",
                "search_space" => Dict{String, Any}(
                    "train.grad_clip" => [1.0, 0.1],
                    "train.strategy.peak_lambda" => [0.0, 2.0],
                ),
            ),
        )

        cfg_path = joinpath(tmp, "hps_config.toml")
        open(cfg_path, "w") do io
            TOML.print(io, cfg)
        end

        parsed = TOML.parsefile(cfg_path)
        ds_a, ms_a, ts_a = WflowRoutingGNN.settings_from_config(parsed, dirname(cfg_path))
        ds_b, ms_b, ts_b = WflowRoutingGNN.parse_run_config(cfg_path)

        @test ts_a.lr_steps == 10
        @test ts_b.lr_steps == 10
        @test ts_a.grad_clip ≈ ts_b.grad_clip
        @test ts_a.lr_warmup_epochs == ts_b.lr_warmup_epochs
        @test ts_a.lr_peak_decay ≈ ts_b.lr_peak_decay
        @test ts_a.strategy.loss_type == :huber
        @test ts_b.strategy.loss_type == :huber
        @test ts_a.strategy.peak_delta ≈ 0.7f0
        @test ts_a.strategy.peak_gamma ≈ 1.5f0
        @test ts_a.strategy.peak_w_max ≈ 3.0f0
        @test ds_a.runs_dir == ds_b.runs_dir
        @test ds_a.wflow_model_path == ds_b.wflow_model_path
        @test ms_a.hidden_dim == ms_b.hidden_dim

        combos = WflowRoutingGNN._box_combinations(Dict{String, Vector{Any}}(
            "train.grad_clip" => Any[1.0, 0.1],
            "train.strategy.peak_lambda" => Any[0.0, 2.0],
        ))

        observed = Set{Tuple{Float32, Float32}}()
        for combo in combos
            d_combo = deepcopy(parsed)
            for (path, value) in combo
                WflowRoutingGNN._set!(d_combo, path, value)
            end
            _, _, ts = WflowRoutingGNN.settings_from_config(d_combo, dirname(cfg_path))
            push!(observed, (ts.grad_clip, ts.strategy.peak_lambda))
        end

        @test observed == Set([
            (1.0f0, 0.0f0),
            (1.0f0, 2.0f0),
            (0.1f0, 0.0f0),
            (0.1f0, 2.0f0),
        ])
    end
end
