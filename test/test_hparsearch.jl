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
                "include_log_upstream_area" => true,
            ),
            "train" => Dict{String, Any}(
                "epochs" => 4,
                "batch_size" => 2,
                "lr_start" => 1e-3,
                "lr_final" => 1e-4,
                "seed" => 123,
                # Intentionally omit lr_steps to verify fallback stays in sync.
                "grad_clip" => 1.0,
                "rollout_grad" => "detached",
                "lr_warmup_epochs" => 2,
                "lr_peak_decay" => 0.8,
                "strategy" => Dict{String, Any}(
                    "steps" => [1, 2],
                    "durations" => [2, 2],
                    "noise_scale" => 0.0,
                    "pushforward_tf_weight" => 0.25,
                    "loss_type" => "huber",
                    "peak_delta" => 0.7,
                    "peak_lambda" => 0.0,
                    "peak_gamma" => 1.5,
                    "peak_w_max" => 3.0,
                ),
            ),
            "eval" => Dict{String, Any}(
                "upstream_points" => 6,
                "subcatch_attribution" => false,
            ),
            "hparsearch" => Dict{String, Any}(
                "search_type" => "box",
                "repetitions" => 2,
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
        @test ts_a.strategy.pushforward_tf_weight ≈ 0.25f0
        @test ts_b.strategy.pushforward_tf_weight ≈ 0.25f0
        @test ts_a.upstream_points == 6
        @test ts_b.upstream_points == 6
        @test !ts_a.subcatch_attribution && !ts_b.subcatch_attribution   # [eval] key
        @test ts_a.subcatch_rank && ts_b.subcatch_rank                    # default
        @test ts_a.rollout_grad == :detached
        @test ts_b.rollout_grad == :detached
        @test ts_a.seed == 123
        @test ts_b.seed == 123
        @test ds_a.runs_dir == ds_b.runs_dir
        @test ds_a.wflow_model_path == ds_b.wflow_model_path
        @test ms_a.hidden_dim == ms_b.hidden_dim
        @test ms_a.include_log_upstream_area == true
        @test ms_b.include_log_upstream_area == true

        seeds = WflowRoutingGNN._resolve_replication_seeds(parsed["hparsearch"])
        @test seeds == [1, 2]

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

        @test WflowRoutingGNN._resolve_replication_seeds(Dict{String,Any}()) == Int[]
        @test WflowRoutingGNN._resolve_replication_seeds(Dict{String,Any}("seeds" => Any[3, 9])) == [3, 9]
        @test_throws ArgumentError WflowRoutingGNN._resolve_replication_seeds(
            Dict{String,Any}("seeds" => Any[1], "repetitions" => 2))
        @test_throws ArgumentError WflowRoutingGNN._resolve_replication_seeds(
            Dict{String,Any}("repetitions" => 0))

        # Repetitions-only mode: empty search space means one base combination.
        hps_empty = WflowRoutingGNN.HParSearchSettings(
            search_type = "box",
            search_space = Dict{String, Vector{Any}}(),
            seeds = [1, 2, 3],
        )
        combos_empty = WflowRoutingGNN._box_combinations(hps_empty.search_space)
        @test length(combos_empty) == 1
        @test isempty(first(combos_empty))
    end
end

@testset "hparsearch: stability headline and per-formulation companions" begin
    mktempdir() do tmp
        function mk_run(name; kind, stab)
            mkpath(joinpath(tmp, name, "metrics"))
            d = Dict{String, Any}(
                "formulation" => Dict{String, Any}("kind" => kind, "h_loss_weight" => 1.0),
                "stability" => stab,
                "mass_balance_residual" => Dict{String, Any}(
                    "volume_pbias" => 1.5, "volume_drift" => -2.0,
                    "pred" => Dict{String, Any}("rms" => 0.01)))
            open(joinpath(tmp, name, "metrics", "metrics.toml"), "w") do io
                TOML.print(io, d)
            end
            return joinpath(tmp, name)
        end
        rd = WflowRoutingGNN._read_run_scalar_metrics

        th = rd(mk_run("th1"; kind = "theta_kinematic_wave",
                       stab = Dict{String, Any}("rollout_contraction" => 0.8,
                                                "analytic_gain" => 0.5, "q_to_h_amp" => 1.2)))
        @test th.rollout_contraction == 0.8
        @test th.analytic_gain == 0.5 && th.q_to_h_amp == 1.2
        @test ismissing(th.backbone_rho_median) && ismissing(th.backbone_rho_max)
        @test th.volume_pbias == 1.5 && th.mb_residual_rms == 0.01
        @test th.h_loss_weight_effective == 1.0
        @test !hasproperty(th, :final_amp)

        hy = rd(mk_run("hy1"; kind = "muskingum_continuity",
                       stab = Dict{String, Any}("rollout_contraction" => 0.6,
                                                "backbone_rho_median" => 0.4,
                                                "backbone_rho_max" => 0.9)))
        @test hy.rollout_contraction == 0.6
        @test hy.backbone_rho_max == 0.9
        @test ismissing(hy.analytic_gain) && ismissing(hy.q_to_h_amp)

        gone = rd(joinpath(tmp, "no_such_run"))          # no metrics.toml
        @test isnan(gone.rollout_contraction) && ismissing(gone.analytic_gain)

        th2 = rd(mk_run("th2"; kind = "theta_kinematic_wave",
                        stab = Dict{String, Any}("rollout_contraction" => 0.6,
                                                 "analytic_gain" => 0.7, "q_to_h_amp" => 1.0)))
        base = (final_val_loss = 0.1, fixed_val_rmse = 0.2, peak_ratio_frac_gt2 = 0.0,
                pooled_kge = 0.5, river_h_nse = 0.5)
        row(combo, msc) = merge((combo_index = combo, run_name_base = "b$combo"), base, msc)
        rows = [row(1, th), row(1, th2), row(2, hy)]
        agg = WflowRoutingGNN._aggregate_repetition_rows(rows)
        @test length(agg) == 2
        @test agg[1].rollout_contraction_mean ≈ 0.7
        @test agg[1].analytic_gain_mean ≈ 0.6
        @test ismissing(agg[1].backbone_rho_max_mean)
        @test agg[2].backbone_rho_max_mean == 0.9
        @test ismissing(agg[2].analytic_gain_mean)
        @test !hasproperty(agg[1], :final_amp_mean)

        # CSV: a metric that does not apply is an empty cell, never NaN
        csv = joinpath(tmp, "results.csv")
        WflowRoutingGNN._write_hps_csv(csv, rows)
        lines = readlines(csv)
        header = split(lines[1], ",")
        @test "rollout_contraction" in header && !("final_amp" in header)
        cell(l, k) = split(lines[l], ","; keepempty = true)[findfirst(==(k), header)]
        @test cell(2, "backbone_rho_max") == "" && cell(2, "analytic_gain") == "0.5"
        @test cell(4, "analytic_gain") == "" && cell(4, "backbone_rho_max") == "0.9"
        @test !occursin("missing", join(lines, "\n"))

        # summary TOML: companions only for the combinations where they exist
        toml = joinpath(tmp, "summary.toml")
        WflowRoutingGNN._write_hps_summary_toml(toml, agg)
        c = TOML.parsefile(toml)["combinations"]
        @test c[1]["rollout_contraction"]["mean"] ≈ 0.7
        @test haskey(c[1], "analytic_gain") && !haskey(c[1], "backbone_rho_max")
        @test haskey(c[2], "backbone_rho_max") && !haskey(c[2], "analytic_gain")
        @test !haskey(c[1], "final_amp")
    end
end
@testset "hpar_search: resume, failure tolerance, incremental results" begin
    mktempdir() do tmp
        model_path = replace(abspath(joinpath(@__DIR__, "..", "test_data", "test_model")), '\\' => '/')
        function write_cfg(lrs; extra = "")
            cfg = joinpath(tmp, "hps.toml")
            write(cfg, """
                [data]
                run_name         = "hps"
                runs_dir         = "runs"
                wflow_model_path = "$model_path"
                train_frac       = 0.6
                val_frac         = 0.2

                [model]
                domain     = "river"
                hidden_dim = 8
                nlayers    = 1

                [train]
                epochs     = 1
                batch_size = 4
                lr_start   = 1e-3
                lr_final   = 1e-5
                seed       = 7
                device     = "cpu"

                [train.strategy]
                steps     = [1]
                durations = [1]

                [hparsearch]
                search_type = "box"
                $extra

                [hparsearch.search_space]
                "train.lr_start" = $lrs
                """)
            return cfg
        end
        runs  = joinpath(tmp, "runs")
        res   = joinpath(runs, "hps_hps_results.csv")
        fails = joinpath(runs, "hps_hps_failures.csv")
        results_rows() = readlines(res)[2:end]
        mtime_model(i) = mtime(joinpath(runs, "hps_hps$(i)", "model", "model.jld2"))

        # 1. run 2 fails (non-numeric lr): run 1 is kept, the failure is listed
        models = hpar_search(write_cfg("""[1e-3, "bad"]"""))
        @test length(models) == 1 && models[1] isa WflowRoutingGNN.WflowGNN
        @test length(results_rows()) == 1
        @test isfile(fails) && occursin("hps_hps02", read(fails, String))
        @test isfile(joinpath(runs, "hps_hps01", "metrics", "metrics.toml"))
        row1 = results_rows()[1]
        t1 = mtime_model("01")

        # 2. same search with run 2 fixed, `resume = true`: run 1 is reused, only run 2 trains
        models = hpar_search(write_cfg("[1e-3, 5e-4]"); resume = true)
        @test length(models) == 1
        @test mtime_model("01") == t1
        @test !isfile(fails)
        rows2 = results_rows()
        @test length(rows2) == 2
        # the reused row comes from metrics.toml (6 significant digits): equal up to that
        same(a, b) = (x = tryparse(Float64, a); y = tryparse(Float64, b);
                      x === nothing || y === nothing ? a == b : isapprox(x, y; rtol = 1e-4) ||
                      (isnan(x) && isnan(y)))
        c1, c2 = split(row1, ","; keepempty = true), split(rows2[1], ","; keepempty = true)
        @test length(c1) == length(c2) && all(same.(c1, c2))

        # 3. a changed setting invalidates its finished run (run 2 is unchanged -> reused);
        #    without `resume` every run is retrained
        models = hpar_search(write_cfg("[2e-3, 5e-4]"); resume = true)
        @test length(models) == 1
        @test mtime_model("01") > t1
        models = hpar_search(write_cfg("[2e-3, 5e-4]"))
        @test length(models) == 2

        # 4. continue_on_error = false propagates the failure
        @test_throws Exception hpar_search(write_cfg("""[1e-3, "bad"]""";
                                                     extra = "continue_on_error = false"))
        @test_throws ArgumentError hpar_search(write_cfg("[1e-3]"; extra = "continue_on_error = 1"))
    end
end

@testset "hpar_search helpers: completeness check and core metrics" begin
    mktempdir() do tmp
        ds = DataSettings(run_name = "r", runs_dir = tmp, train_frac = 0.6, val_frac = 0.2,
                          wflow_model_path = joinpath(@__DIR__, "..", "test_data", "test_model"))
        ms = ModelSettings(domain = "river")
        mk_ts(epochs) = TrainSettings(epochs = epochs, batch_size = 4, lr_start = 1f-3,
                                      lr_final = 1f-5, lr_steps = 2, device = :cpu,
                                      strategy = TrainingStrategy([1], [epochs], 0.0))
        ts = mk_ts(1)
        run_dir = joinpath(tmp, "r")
        complete = WflowRoutingGNN._hps_run_is_complete
        @test !complete(run_dir, ds, ms, ts)

        mkpath(joinpath(run_dir, "model")); mkpath(joinpath(run_dir, "metrics"))
        mkpath(joinpath(run_dir, "output"))
        write(joinpath(run_dir, "metrics", "metrics.toml"), """
            [run]
            n_params = 12
            train_duration_s = 1.5
            val_n_timesteps = 7
            [loss]
            final_val_rollout = 0.25
            """)
        @test !complete(run_dir, ds, ms, ts)               # no output NetCDFs yet
        for f in ("train_pred.nc", "train_true.nc", "val_pred.nc", "val_true.nc")
            touch(joinpath(run_dir, "output", f))
        end
        @test !complete(run_dir, ds, ms, ts)               # no saved settings yet
        save_data_settings(joinpath(run_dir, "model", "data_settings.toml"), ds)
        save_model_settings(joinpath(run_dir, "model", "model_settings.toml"), ms)
        save_train_settings(joinpath(run_dir, "model", "train_settings.toml"), ts)
        @test complete(run_dir, ds, ms, ts)
        @test !complete(run_dir, ds, ms, mk_ts(2))

        m = WflowRoutingGNN._read_run_core_metrics(run_dir)
        @test m.n_params == 12 && m.val_n_timesteps == 7
        @test m.final_val_loss == 0.25 && m.train_duration_s == 1.5
        @test isnan(m.final_train_loss) && isnan(m.val_rollout_duration_s)
    end
end