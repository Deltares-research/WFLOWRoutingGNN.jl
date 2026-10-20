using Test
using WflowRoutingGNN

# Stage-4-only entry points working on an existing run directory:
# resolve_staticmaps, spatial_eval_run, regen_mb_diagnostics, plot_ldd.

const AN_MODEL_PATH = joinpath(@__DIR__, "..", "test_data", "test_model")

an_settings(tmp; enforce_mb = true) = (
    DataSettings(run_name = "an", runs_dir = tmp, wflow_model_path = AN_MODEL_PATH,
                 train_frac = 0.6, val_frac = 0.2),
    ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1,
                  enforce_mass_balance = enforce_mb),
    TrainSettings(epochs = 1, batch_size = 4, lr_start = 1f-3, lr_final = 1f-5,
                  lr_steps = 1, seed = 7, device = :cpu,
                  strategy = TrainingStrategy([1, 2], [1, 1], 0.0)),
)

@testset "resolve_staticmaps" begin
    mktempdir() do tmp
        # Direct hit.
        @test resolve_staticmaps(AN_MODEL_PATH, tmp) ==
              joinpath(AN_MODEL_PATH, "staticmaps.nc")

        # Fallback: <repo>/models/<name>/staticmaps.nc for <repo>/experiments/<run>.
        run_dir = joinpath(tmp, "experiments", "run")
        mkpath(run_dir)
        mkpath(joinpath(tmp, "models", "some_model"))
        fallback = joinpath(tmp, "models", "some_model", "staticmaps.nc")
        touch(fallback)
        @test resolve_staticmaps("/not/here/some_model/", run_dir) == fallback
        @test resolve_staticmaps("C:\\elsewhere\\some_model", run_dir) == fallback

        @test_throws ErrorException resolve_staticmaps("/not/here/other_model", run_dir)
    end
end

@testset "spatial_eval_run / regen_mb_diagnostics on an existing run" begin
    mktempdir() do tmp
        ds, ms, ts = an_settings(tmp)
        run_dir = joinpath(tmp, "an")
        prep = preprocess_run(ds, ms, ts)
        tr   = train_run(prep, ds, ms, ts)
        infer_run(tr.model, prep, ms, ts; dirs = (; output = joinpath(run_dir, "output")),
                  splits = (:train, :val))

        @testset "spatial_eval_run" begin
            for split in ("val", "train")
                res = spatial_eval_run(run_dir; split)
                @test res.metrics !== nothing
                @test res.ramp !== nothing                 # river_q is a state variable
                @test isfile(joinpath(run_dir, "output",  "$(split)_spatial_metrics.nc"))
                @test isfile(joinpath(run_dir, "metrics", "$(split)_spatial_metrics.csv"))
                @test isfile(joinpath(run_dir, "plots",   "$(split)_spatial_metrics.png"))
                @test isfile(joinpath(run_dir, "metrics", "$(split)_q_overpred_vs_ramp.csv"))
                @test isfile(joinpath(run_dir, "plots",   "$(split)_q_overpred_vs_ramp.png"))
            end
            @test_throws ArgumentError spatial_eval_run(joinpath(tmp, "nope"))
            @test_throws ArgumentError spatial_eval_run(run_dir; split = "bogus")
        end

        @testset "regen_mb_diagnostics (mass-balance layer)" begin
            out = regen_mb_diagnostics(run_dir)
            @test out !== nothing
            @test haskey(out, :volume_budget)
            for f in ("mb_diagnostics.csv", "mb_floor_fraction_by_node.csv",
                      "mb_stiffness_by_node.csv", "volume_budget.csv")
                @test isfile(joinpath(run_dir, "metrics", f))
            end
            for f in ("mb_diagnostics.png", "volume_budget.png")
                @test isfile(joinpath(run_dir, "plots", f))
            end
        end

        @testset "regen_mb_diagnostics needs the prep bundle" begin
            rm(joinpath(run_dir, "prep"); recursive = true)
            @test_throws ArgumentError regen_mb_diagnostics(run_dir)
        end
    end

    @testset "regen_mb_diagnostics without a mass-balance layer" begin
        mktempdir() do tmp
            ds, ms, ts = an_settings(tmp; enforce_mb = false)
            prep = preprocess_run(ds, ms, ts)
            train_run(prep, ds, ms, ts)
            @test regen_mb_diagnostics(joinpath(tmp, "an")) === nothing
        end
    end
end

@testset "plot_ldd" begin
    mktempdir() do tmp
        png = joinpath(tmp, "ldd.png")
        fig = plot_ldd(joinpath(AN_MODEL_PATH, "staticmaps.nc"); path = png)
        @test fig !== nothing
        @test isfile(png) && filesize(png) > 0
    end
end
