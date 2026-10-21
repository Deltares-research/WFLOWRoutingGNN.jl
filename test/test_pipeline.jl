using Test
using JLD2
using Dates
using TOML
using WflowRoutingGNN

# Stages 1 (preprocess), 2 (train), 3 (infer) and 4 (analyze) on the small test model.

const PL_MODEL_PATH = joinpath(@__DIR__, "..", "test_data", "test_model")

pl_settings(tmp; loss_type = :mse, steps = [1, 2], daterange = nothing) = (
    DataSettings(run_name = "pl", runs_dir = tmp, wflow_model_path = PL_MODEL_PATH,
                 train_frac = 0.6, val_frac = 0.2),
    ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1),
    TrainSettings(epochs = 2, batch_size = 4, lr_start = 1f-3, lr_final = 1f-5,
                  lr_steps = 2, seed = 7, device = :cpu, val_daterange = daterange,
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

@testset "gather_nodes inverts regrid" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        prep = preprocess_run(ds, ms, ts)
        vars = WflowRoutingGNN.DOMAIN_VARS[ms.domain]["state"]
        T    = 4
        x    = randn(Float32, length(vars), prep.n_nodes, T)
        grids = regrid(x, prep.grid, ms.domain)
        @test isequal(gather_nodes(grids, prep.grid, ms.domain), x)

        # Inactive cells stay NaN and are ignored by the gather.
        active = falses(prep.grid.nrows, prep.grid.ncols)
        for (r, c) in zip(prep.grid.rows, prep.grid.cols)
            active[r, c] = true
        end
        @test all(isnan, grids[first(vars)][.!active, :])

        # Missing variable / wrong raster size are rejected.
        @test_throws ArgumentError gather_nodes(Dict{String, Array{Float32,3}}(),
                                                prep.grid, ms.domain)
        bad = Dict(v => zeros(Float32, 1, 1, T) for v in vars)
        @test_throws ArgumentError gather_nodes(bad, prep.grid, ms.domain)
    end
end

@testset "infer_run / analyze_run stages" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp; daterange = (DateTime(1900), DateTime(2100)))
        run_dir = joinpath(tmp, "pl")
        prep = preprocess_run(ds, ms, ts)
        tr   = train_run(prep, ds, ms, ts)
        vars = WflowRoutingGNN.DOMAIN_VARS[ms.domain]["state"]
        out  = joinpath(run_dir, "output")

        # train_run persists what metrics.toml needs.
        summ = load_train_summary(joinpath(run_dir, "model"))
        @test summ.n_params == tr.n_params > 0
        @test summ.train_duration_s == tr.train_duration_s
        @test summ.losses.train_rollout == tr.losses.train_rollout

        # --- Stage 3: infer ---
        inf = infer_run(tr.model, prep, ms, ts; dirs = (; output = out))
        @test inf.val_rollout_duration_s > 0
        for split in ("train", "val", "val_daterange"), kind in ("pred", "true")
            @test isfile(joinpath(out, "$(split)_$(kind).nc"))
        end
        @test !isfile(joinpath(out, "test_pred.nc"))
        # No analysis products from the inference stage.
        @test !isfile(joinpath(out, "spatial_metrics.nc"))

        p_direct, t_direct = evaluate_trajectory(
            tr.model, prep.dataset.val, prep.norm_stats, ms.domain, prep.static_arr;
            device = :cpu, postscale = prep.postscale)
        @test inf.val_n_timesteps == size(p_direct, 3)
        val_pred = joinpath(out, "val_pred.nc")
        val_true = joinpath(out, "val_true.nc")
        @test isequal(gather_nodes(WflowRoutingGNN.load_regrid_nc(val_pred, vars),
                                   prep.grid, ms.domain), p_direct)
        @test isequal(gather_nodes(WflowRoutingGNN.load_regrid_nc(val_true, vars),
                                   prep.grid, ms.domain), t_direct)

        # Time axis: val frame i is graph (n_train + 1 + i) of the series.
        n_train = length(prep.dataset.train)
        vt = WflowRoutingGNN.load_regrid_times(val_pred)
        @test vt == [prep.times[clamp(n_train + 1 + i, 1, length(prep.times))]
                     for i in 1:size(p_direct, 3)]

        @test_throws ArgumentError infer_run(tr.model, prep, ms, ts; dirs = (; output = out),
                                             splits = (:bogus,))

        # load_model reproduces the trained weights.
        m2 = load_model(prep, ms, joinpath(run_dir, "model"))
        p2, _ = evaluate_trajectory(m2, prep.dataset.val, prep.norm_stats, ms.domain,
                                    prep.static_arr; device = :cpu, postscale = prep.postscale)
        @test p2 == p_direct
        @test_throws ArgumentError load_model(prep, ms, joinpath(tmp, "nope"))

        # --- Stage 4: analyze (from the NetCDFs; no model needed) ---
        dirs = (; output = out, plots = joinpath(run_dir, "plots"),
                  metrics = joinpath(run_dir, "metrics"), model = joinpath(run_dir, "model"))
        res = analyze_run(prep, ms, ts; dirs, infer = inf)
        @test res.spatial_summary !== nothing
        @test res.river_q_perf !== nothing
        @test res.volume_budget === nothing          # no mass-balance layer
        for f in ("spatial_metrics.csv", "q_overprediction_vs_ramp.csv", "metrics.toml")
            @test isfile(joinpath(dirs.metrics, f))
        end
        # One CSV per selected node (`_node<i>_r<k>` suffix) when upstream_points > 1.
        mfiles = readdir(dirs.metrics)
        @test any(f -> startswith(f, "downstream_timeseries_node") ||
                       f == "downstream_timeseries.csv", mfiles)
        @test any(f -> startswith(f, "downstream_timeseries_daterange"), mfiles)
        @test isfile(joinpath(out, "spatial_metrics.nc"))
        @test isfile(joinpath(dirs.plots, "validation.mp4"))
        @test isfile(joinpath(dirs.plots, "validation_daterange.mp4"))

        toml = TOML.parsefile(joinpath(dirs.metrics, "metrics.toml"))
        @test toml["run"]["n_params"] == tr.n_params
        @test toml["run"]["val_n_timesteps"] == inf.val_n_timesteps
        @test haskey(toml["run"], "val_rollout_duration_s")

        # Standalone analysis without timing info omits only the rollout duration.
        rm(joinpath(dirs.metrics, "metrics.toml"))
        analyze_run(prep, ms, ts; dirs)
        toml2 = TOML.parsefile(joinpath(dirs.metrics, "metrics.toml"))
        @test !haskey(toml2["run"], "val_rollout_duration_s")
        @test toml2["run"]["n_params"] == tr.n_params

        # Checkpoint-style call: no metrics.toml.
        rm(joinpath(dirs.metrics, "metrics.toml"))
        analyze_run(prep, ms, ts; dirs, metrics_toml = false)
        @test !isfile(joinpath(dirs.metrics, "metrics.toml"))

        # Missing inference outputs.
        empty = (; output = joinpath(tmp, "empty_out"), plots = joinpath(tmp, "empty_plots"),
                   metrics = joinpath(tmp, "empty_metrics"))
        @test_throws ArgumentError analyze_run(prep, ms, ts; dirs = empty)
    end
end

@testset "mb_residual_diagnostics (layer-free)" begin
    # Chain 1 → 2 → 3 with lateral inflow; h built exactly by the θ recursion.
    n, T = 3, 6
    src, tgt = [1, 2], [2, 3]
    wl = Float32[100, 200, 400]
    dt = 10f0
    q  = Float32[1 2 3 4 3 2; 2 3 5 6 5 4; 3 4 6 9 8 6]
    iw = Float32[0.5 0.5 1 1 0.5 0.5; 0 0.2 0.2 0.4 0.4 0; 0.1 0.1 0.1 0.3 0.3 0.1]
    A  = [0 0 0; 1 0 0; 0 1 0]
    net = Float32.(A) * q .+ iw .- q
    for θ in (1f0, 0.5f0, 0f0)
        h = zeros(Float32, n, T)
        h[:, 1] .= 1f0
        for k in 2:T
            h[:, k] = h[:, k-1] .+ dt ./ wl .* (θ .* net[:, k] .+ (1f0 - θ) .* net[:, k-1])
        end
        r = mb_residual_diagnostics(q, h, iw; edge_src = src, edge_tgt = tgt, wl, dt, θ)
        @test size(r.residual) == (n, T - 1)
        @test maximum(abs, r.residual) < 1f-5
        @test r.summary.n == n * (T - 1)
        @test all(abs.(r.volume_residual) .< 1f-3)

        # A perturbation of h[2, 3] appears as -δ at step 2 and +δ at step 3.
        hp = copy(h); hp[2, 3] += 0.25f0
        rp = mb_residual_diagnostics(q, hp, iw; edge_src = src, edge_tgt = tgt, wl, dt, θ)
        @test rp.residual[2, 2] ≈ 0.25f0 atol = 1f-5
        @test rp.residual[2, 3] ≈ -0.25f0 atol = 1f-5
        @test rp.residual[[1, 3], :] ≈ r.residual[[1, 3], :] atol = 1f-5
        @test rp.volume_residual[2] ≈ 0.25f0 * wl[2] atol = 1f-3
        @test rp.rms_node[2] > rp.rms_node[1]
    end

    # Invalid channel area -> NaN residual for that node, counted as not valid.
    h0 = ones(Float32, n, T)
    r = mb_residual_diagnostics(q, h0, iw; edge_src = src, edge_tgt = tgt,
                                wl = Float32[100, NaN, 0], dt, θ = 1f0)
    @test all(isnan, r.residual[2:3, :])
    @test r.n_valid_node == [T - 1, 0, 0]
    @test isnan(r.rms_node[2])

    @test_throws ArgumentError mb_residual_diagnostics(q, h0[:, 1:3], iw;
        edge_src = src, edge_tgt = tgt, wl, dt)
    @test_throws ArgumentError mb_residual_diagnostics(q, h0, iw;
        edge_src = src, edge_tgt = tgt, wl, dt, θ = 1.5f0)
    @test_throws ArgumentError mb_residual_diagnostics(q[:, 1:1], h0[:, 1:1], iw[:, 1:1];
        edge_src = src, edge_tgt = tgt, wl, dt)
end

@testset "analyze_run: residual for every model, layer diagnostics by formulation" begin
    mktempdir() do tmp
        ds, ms, ts = pl_settings(tmp)
        run_dir = joinpath(tmp, "pl")
        prep = preprocess_run(ds, ms, ts)
        tr   = train_run(prep, ds, ms, ts)
        inf  = infer_run(tr.model, prep, ms, ts; dirs = (; output = joinpath(run_dir, "output")))
        dirs = (; output = joinpath(run_dir, "output"), plots = joinpath(run_dir, "plots"),
                  metrics = joinpath(run_dir, "metrics"), model = joinpath(run_dir, "model"))
        vars = WflowRoutingGNN.DOMAIN_VARS[ms.domain]["state"]
        qi, hi = findfirst(==("river_q"), vars), findfirst(==("river_h"), vars)

        # No model supplied: the residual is still produced, layer diagnostics are not.
        res = analyze_run(prep, ms, ts; dirs, infer = inf)
        @test res.mb_residual !== nothing
        @test res.mb_floor === nothing && res.mb_stiffness === nothing
        @test isfile(joinpath(dirs.metrics, "mb_residual.csv"))
        @test isfile(joinpath(dirs.metrics, "mb_residual_by_node.csv"))
        @test !isfile(joinpath(dirs.metrics, "mb_floor_fraction_by_node.csv"))
        toml = TOML.parsefile(joinpath(dirs.metrics, "metrics.toml"))
        @test haskey(toml, "mass_balance_residual")
        @test haskey(toml["mass_balance_residual"]["pred"], "rms")
        @test haskey(toml["mass_balance_residual"]["reference"], "rms")
        @test !haskey(toml, "mass_balance_floor")
        rows = readlines(joinpath(dirs.metrics, "mb_residual.csv"))
        @test length(rows) == inf.val_n_timesteps          # header + (T-1) steps
        @test length(readlines(joinpath(dirs.metrics, "mb_residual_by_node.csv"))) ==
              prep.n_nodes + 1

        # With the MB model: the layer-free residual of the predicted fields agrees
        # with the layer-internal h_raw (pred_h − h_raw is the floor correction only).
        res_m = analyze_run(prep, ms, ts; dirs, model = tr.model, infer = inf)
        @test res_m.mb_floor !== nothing && res_m.volume_budget !== nothing
        @test res_m.mb_residual.pred.rms ≈ res.mb_residual.pred.rms
        toml_m = TOML.parsefile(joinpath(dirs.metrics, "metrics.toml"))
        @test haskey(toml_m, "mass_balance_floor")
        @test haskey(toml_m, "mass_balance_stiffness")
        @test haskey(toml_m, "volume_budget")

        diags = rollout_mb_diagnostics(tr.model, prep.dataset.val, prep.static_arr)
        # Feed the layer's own q, h and lateral-inflow series so only the equation is
        # compared. (The NetCDF comes from a separate rollout; once the lateral inflow
        # is applied this tiny model is stiff enough that its roundoff differences to
        # the diagnostics rollout are amplified to metres in h.)
        iw = diags.inwater
        wl = prep.postscale["river_q"] ./ prep.postscale["river_h"]
        r = mb_residual_diagnostics(diags.pred_q, diags.pred_h, iw;
                                    edge_src = prep.edge_src, edge_tgt = prep.edge_tgt,
                                    wl, dt = prep.dt, θ = ms.mb_theta)
        # The residual of step t-1 -> t equals h[t] - h_raw[t] up to the roundoff of the
        # two evaluation orders.
        expected = diags.pred_h[:, 2:end] .- diags.h_raw[:, 2:end]
        ok = isfinite.(expected) .& isfinite.(r.residual)
        scale = max(1f0, maximum(abs, diags.pred_h[isfinite.(diags.pred_h)]))
        @test maximum(abs.(r.residual[ok] .- expected[ok])) < 1f-3 * scale

        # A model without a mass-balance layer: residual only.
        ms_none = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1,
                                mb_formulation = "none")
        tr_none = train_run(prep, ds, ms_none, ts)
        @test !WflowRoutingGNN.has_mass_balance(tr_none.model.mass_balance)
        @test rollout_mb_diagnostics(tr_none.model, prep.dataset.val, prep.static_arr) === nothing
        inf_n = infer_run(tr_none.model, prep, ms_none, ts; dirs = (; output = dirs.output))
        res_n = analyze_run(prep, ms_none, ts; dirs, model = tr_none.model, infer = inf_n)
        @test res_n.mb_residual !== nothing && isfinite(res_n.mb_residual.pred.rms)
        @test res_n.mb_floor === nothing && res_n.mb_stiffness === nothing &&
              res_n.volume_budget === nothing
        toml_n = TOML.parsefile(joinpath(dirs.metrics, "metrics.toml"))
        @test haskey(toml_n, "mass_balance_residual")
        @test !haskey(toml_n, "mass_balance_floor")
        @test !haskey(toml_n, "volume_budget")
    end
end

@testset "muskingum_continuity hybrid: train -> infer -> analyze end to end" begin
    mktempdir() do tmp
        ds, _, ts = pl_settings(tmp)
        ms  = ModelSettings(domain = "river", hidden_dim = 8, nlayers = 1,
                            mb_formulation = "muskingum_continuity")
        run_dir = joinpath(tmp, "pl")
        prep = preprocess_run(ds, ms, ts)
        tr   = train_run(prep, ds, ms, ts)
        @test tr.model.mass_balance isa MuskingumContinuityHybrid
        @test all(isfinite, tr.losses.train_rollout)
        @test load_model_settings(joinpath(run_dir, "model", "model_settings.toml")).mb_formulation ==
              "muskingum_continuity"

        inf  = infer_run(tr.model, prep, ms, ts; dirs = (; output = joinpath(run_dir, "output")))
        dirs = (; output = joinpath(run_dir, "output"), plots = joinpath(run_dir, "plots"),
                  metrics = joinpath(run_dir, "metrics"), model = joinpath(run_dir, "model"))
        res = analyze_run(prep, ms, ts; dirs, model = tr.model, infer = inf)
        @test res.mb_residual !== nothing && isfinite(res.mb_residual.pred.rms)
        @test res.volume_budget === nothing && res.mb_stiffness === nothing
        csv = joinpath(dirs.metrics, "mb_backbone_contraction_by_node.csv")
        @test isfile(csv)
        @test length(readlines(csv)) == prep.n_nodes + 1
        @test rollout_mb_diagnostics(tr.model, prep.dataset.val, prep.static_arr) === nothing
        @test all(0 .< backbone_contraction(tr.model.mass_balance) .< 1)
    end
end