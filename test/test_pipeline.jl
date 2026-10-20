using Test
using JLD2
using WflowRoutingGNN

# Stages 1 (preprocess) and 2 (train) on the small test model.

const PL_MODEL_PATH = joinpath(@__DIR__, "..", "test_data", "test_model")

pl_settings(tmp; loss_type = :mse, steps = [1, 2]) = (
    DataSettings(run_name = "pl", runs_dir = tmp, wflow_model_path = PL_MODEL_PATH,
                 train_frac = 0.6, val_frac = 0.2),
    ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1),
    TrainSettings(epochs = 2, batch_size = 4, lr_start = 1f-3, lr_final = 1f-5,
                  lr_steps = 2, seed = 7, device = :cpu,
                  strategy = TrainingStrategy(steps, fill(1, length(steps)), 0.0;
                                              loss_type = loss_type)),
)

@testset "preprocess_run / PrepBundle" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        run_dir = joinpath(tmp, "pl")
        prep = preprocess_run(ds, ms, ts)

        @test prep isa PrepBundle
        @test prep.prep_schema == PREP_SCHEMA_VERSION
        @test prep.wflow_schema == ds.wflow_schema
        @test prep.domain == "river"
        @test prep.nhorizon == 3
        @test prep.peak_stats === nothing          # mse loss: not computed
        @test prep.dt == get_timestep(prep.output_file)
        @test length(prep.edge_src) == length(prep.edge_tgt)
        @test length(prep.grid.rows) == prep.n_nodes == length(prep.grid.cols)

        # Artifact set: new prep/ file plus the historical dataset/grid files.
        @test isfile(joinpath(run_dir, "prep", "preprocessed.jld2"))
        for f in ("grid", "train", "val", "test")
            @test isfile(joinpath(run_dir, "output", "data", "$f.jld2"))
        end
        @test isfile(joinpath(run_dir, "metrics", "normalized_tail_diagnostics.toml"))

        # Round trip through disk.
        loaded = load_prep(run_dir)
        for f in (:prep_schema, :wflow_schema, :domain, :staticmaps_file, :output_file,
                  :train_frac, :val_frac, :nhorizon, :dt, :times, :n_nodes, :edge_src,
                  :edge_tgt, :grid, :norm_stats, :postscale, :static_arr, :peak_stats)
            @test getfield(loaded, f) == getfield(prep, f)
        end
        for s in (:train, :val, :test)
            a, b = getfield(prep.dataset, s), getfield(loaded.dataset, s)
            @test length(a) == length(b)
            @test a[1][1].ndata.state == b[1][1].ndata.state
            @test a[end][end].ndata.forcing == b[end][end].ndata.forcing
        end

        # Version check and missing bundle.
        file = joinpath(run_dir, "prep", "preprocessed.jld2")
        d = JLD2.load(file)
        d["prep_schema"] = PREP_SCHEMA_VERSION + 1
        JLD2.save(file, d)
        @test_throws ArgumentError load_prep(run_dir)
        @test_throws ArgumentError load_prep(joinpath(tmp, "does_not_exist"))
    end

    @testset "peak stats follow the loss" begin
        mktempdir() do tmp
            ds, ms, ts = pl_settings(tmp; loss_type = :huber)
            prep = preprocess_run(ds, ms, ts)
            @test prep.peak_stats !== nothing
            @test length(prep.peak_stats.q.u) == prep.n_nodes
            @test length(prep.peak_stats.h.s) == prep.n_nodes
        end
    end
end

@testset "train_run reads only the PrepBundle" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        run_dir = joinpath(tmp, "pl")
        preprocess_run(ds, ms, ts)

        # Resume from disk alone.
        prep = load_prep(run_dir)
        out  = train_run(prep, ds, ms, ts)
        @test out.model isa WflowGNN
        @test out.train_duration_s > 0
        @test length(out.losses.train_rollout) == ts.epochs

        for f in ("model.jld2", "data_settings.toml", "model_settings.toml",
                  "train_settings.toml", "norm_stats.toml")
            @test isfile(joinpath(run_dir, "model", f))
        end
        @test isfile(joinpath(run_dir, "metrics", "losses.csv"))

        # Same seed + same bundle -> identical training history.
        out2 = train_run(prep, ds, ms, ts)
        @test out2.losses.train_rollout == out.losses.train_rollout

        # Mismatched curriculum horizon / missing peak stats are rejected.
        _, _, ts_long = pl_settings(tmp; steps = [1, 2, 3])
        @test_throws ArgumentError train_run(prep, ds, ms, ts_long)
        _, _, ts_huber = pl_settings(tmp; loss_type = :huber)
        @test_throws ArgumentError train_run(prep, ds, ms, ts_huber)
        @test_throws ArgumentError train_run(prep, ds,
            ModelSettings(domain = "land", hidden_dim = 8, nlayers = 1), ts)
    end
end
