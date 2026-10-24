using Test
using TOML
using WflowRoutingGNN

# Stage type hierarchy (`run_stage`) and the `scripts/pipeline.jl` entry points.
# `pl_settings` / `PL_MODEL_PATH` come from test_pipeline.jl.

const ST_SCRIPT = joinpath(@__DIR__, "..", "scripts", "pipeline.jl")

@testset "stage_from_name" begin
    @test stage_from_name("prep") isa PreprocessStage
    @test stage_from_name("preprocess") isa PreprocessStage
    @test stage_from_name("train") isa TrainStage
    @test stage_from_name("infer") isa InferStage
    @test stage_from_name("analyze") isa AnalyzeStage
    @test stage_from_name(" ALL ") isa AllStages
    @test all(s -> s isa Stage, values(STAGE_NAMES))
    err = try stage_from_name("bogus"); nothing catch e; e end
    @test err isa ArgumentError
    @test all(n -> occursin(n, err.msg), ("prep", "train", "infer", "analyze", "all"))
end

@testset "run_stage: stages standalone == AllStages" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        ds_all = DataSettings(run_name = "all", runs_dir = tmp, wflow_model_path = PL_MODEL_PATH,
                              train_frac = ds.train_frac, val_frac = ds.val_frac)
        dir_st  = joinpath(tmp, "pl")
        dir_all = joinpath(tmp, "all")
        vars    = WflowRoutingGNN.DOMAIN_VARS[ms.domain]["state"]

        # Later stages need the artifacts of earlier ones.
        @test_throws ArgumentError run_stage(TrainStage(),   StageContext(ds, ms, ts))
        @test_throws ArgumentError run_stage(InferStage(),   StageContext(ds, ms, ts))
        @test_throws ArgumentError run_stage(AnalyzeStage(), StageContext(ds, ms, ts))

        # One stage per fresh context: everything is handed over through disk.
        ctx = run_stage(PreprocessStage(), StageContext(ds, ms, ts))
        @test ctx.prep isa PrepBundle && ctx.train === nothing
        @test isfile(joinpath(dir_st, "prep", "preprocessed.jld2"))

        ctx = run_stage(TrainStage(), StageContext(ds, ms, ts))
        @test ctx.train.model isa WflowGNN && ctx.infer === nothing
        @test isfile(joinpath(dir_st, "model", "model.jld2"))

        ctx = run_stage(InferStage(), StageContext(ds, ms, ts))
        @test ctx.infer.val_n_timesteps > 0
        @test isfile(joinpath(dir_st, "output", "val_pred.nc"))
        @test !isfile(joinpath(dir_st, "metrics", "metrics.toml"))

        ctx = run_stage(AnalyzeStage(), StageContext(ds, ms, ts))
        @test ctx.analysis.spatial_summary !== nothing
        @test isfile(joinpath(dir_st, "metrics", "metrics.toml"))
        @test isfile(joinpath(dir_st, "output", "spatial_metrics.nc"))

        # The whole pipeline in one context (in-memory hand-over).
        ctx_all = run_stage(AllStages(), StageContext(ds_all, ms, ts))
        @test ctx_all.prep !== nothing && ctx_all.train !== nothing &&
              ctx_all.infer !== nothing && ctx_all.analysis !== nothing
        @test isfile(joinpath(dir_all, "metrics", "metrics.toml"))

        # Same seed -> same training history and same predictions either way.
        @test load_train_summary(joinpath(dir_st, "model")).losses.train_rollout ==
              load_train_summary(joinpath(dir_all, "model")).losses.train_rollout
        for f in ("train_pred.nc", "val_pred.nc", "val_true.nc")
            a = WflowRoutingGNN.load_regrid_nc(joinpath(dir_st,  "output", f), vars)
            b = WflowRoutingGNN.load_regrid_nc(joinpath(dir_all, "output", f), vars)
            @test all(v -> isequal(a[v], b[v]), vars)
        end

        # An empty AllStages is the identity.
        c0 = StageContext(ds, ms, ts)
        @test run_stage(AllStages(()), c0) === c0
    end
end

@testset "run_wflow_gnn composes the stages" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        model, metrics = run_wflow_gnn(ds, ms, ts)
        @test model isa WflowGNN
        @test metrics.n_params > 0
        @test metrics.val_n_timesteps > 0
        @test isfile(joinpath(tmp, "pl", "metrics", "metrics.toml"))
    end
end

@testset "scripts/pipeline.jl" begin
    mktempdir() do tmp
        cfg = joinpath(tmp, "config.toml")
        write(cfg, """
            [data]
            run_name         = "cli"
            runs_dir         = "runs"
            wflow_model_path = "$(replace(abspath(PL_MODEL_PATH), '\\' => '/'))"
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
            """)

        # Library entry point used by the script (name or Stage).
        ctx = run_stage_from_toml("prep", cfg)
        @test ctx.prep isa PrepBundle
        @test isfile(joinpath(tmp, "runs", "cli", "prep", "preprocessed.jld2"))
        @test_throws ArgumentError run_stage_from_toml("bogus", cfg)
        @test_throws ArgumentError run_stage_from_toml("prep", joinpath(tmp, "missing.toml"))
        ctx = run_stage_from_toml(TrainStage(), cfg)
        @test isfile(joinpath(tmp, "runs", "cli", "model", "model.jld2"))

        # Argument validation of the CLI (rejected before the package is loaded).
        julia = Base.julia_cmd()
        proj  = abspath(joinpath(@__DIR__, ".."))
        cli_exit(args...) = run(pipeline(ignorestatus(`$julia --project=$proj $ST_SCRIPT $args`);
                                         stdout = devnull, stderr = devnull)).exitcode
        @test cli_exit() == 1
        @test cli_exit("prep") == 1
        @test cli_exit("prep", joinpath(tmp, "missing.toml")) == 1
    end
end

@testset "run_wflow_gnn(; resume): skips finished stages" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        root = joinpath(tmp, "pl")
        mt(f) = mtime(joinpath(root, f))
        done = WflowRoutingGNN._stage_done
        dirs = WflowRoutingGNN._run_dirs(ds)
        @test !done(AnalyzeStage(), dirs) && !done(AllStages(), dirs)

        model, m0 = run_wflow_gnn(ds, ms, ts)
        @test all(s -> done(s, dirs), AllStages().stages)

        # everything finished: nothing is rerun, results come from disk
        t_model, t_metrics = mt("model/model.jld2"), mt("metrics/metrics.toml")
        model1, m1 = run_wflow_gnn(ds, ms, ts; resume = true)
        @test model1 isa WflowGNN
        @test mt("model/model.jld2") == t_model && mt("metrics/metrics.toml") == t_metrics
        @test m1.n_params == m0.n_params && m1.val_n_timesteps == m0.val_n_timesteps
        @test m1.final_val_loss ≈ m0.final_val_loss rtol = 1e-4

        # analyze artifact missing: only that stage reruns, the model is not retrained
        rm(joinpath(root, "metrics", "metrics.toml"))
        model2, m2 = run_wflow_gnn(ds, ms, ts; resume = true)
        @test isfile(joinpath(root, "metrics", "metrics.toml"))
        @test mt("model/model.jld2") == t_model
        @test m2.n_params == m0.n_params

        # train artifact missing: train and everything after it rerun
        rm(joinpath(root, "model", "model.jld2"))
        run_wflow_gnn(ds, ms, ts; resume = true)
        @test isfile(joinpath(root, "model", "model.jld2"))
        @test mt("model/model.jld2") > t_model
    end
end