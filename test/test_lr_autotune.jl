using Test
using TOML
using WflowRoutingGNN

@testset "recommend_lr robustness" begin
    n = 40
    lrs = collect(10 .^ range(-7, -1; length = n))

    # Synthetic smooth-loss curve with early hump, descent, then late rise.
    losses = Float64[]
    for i in 1:n
        if i <= 8
            push!(losses, 600.0 + 40.0 * i)
        elseif i <= 28
            push!(losses, 940.0 - 34.0 * (i - 8))
        else
            push!(losses, 260.0 + 35.0 * (i - 28))
        end
    end

    # Healthy plateau + one transient spike + sustained late growth.
    gnorms = fill(120.0, n)
    gnorms[12] = 1.0e7
    for i in 31:n
        gnorms[i] = 120.0 * (1.35 ^ (i - 30))
    end

    rec = recommend_lr(lrs, losses, gnorms)

    @test isfinite(rec.safe)
    @test rec.safe > 10.0 * minimum(lrs)
    @test !rec.near_lr_floor
    @test !rec.min10_near_floor
    @test rec.gnorm_cap > 10.0 * minimum(lrs)
end

@testset "aggregate_lr_recommendations" begin
    recs = [
        (; safe = 1e-3, gnorm_ref = 2.0e2, near_lr_floor = false, min_over_10 = 2e-3, min10_near_floor = false),
        (; safe = 9e-4, gnorm_ref = 1.8e2, near_lr_floor = false, min_over_10 = 1.7e-3, min10_near_floor = false),
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 1e-7, min10_near_floor = true),
    ]
    agg = aggregate_lr_recommendations(recs)
    @test agg.n_valid == 2
    @test agg.n_total == 3
    @test agg.safe ≈ 9.5e-4
    @test agg.gnorm_ref ≈ 190.0
    @test !agg.used_fallback

    agg_fallback = aggregate_lr_recommendations([
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 5e-3, min10_near_floor = false),
        (; safe = NaN,  gnorm_ref = NaN,   near_lr_floor = false, min_over_10 = 6e-3, min10_near_floor = false),
    ])
    @test agg_fallback.n_valid == 0
    @test agg_fallback.used_fallback
    @test agg_fallback.safe ≈ 5.5e-3

    agg_none = aggregate_lr_recommendations([
        (; safe = 1e-7, gnorm_ref = 1.0e6, near_lr_floor = true, min_over_10 = 1e-7, min10_near_floor = true),
        (; safe = NaN,  gnorm_ref = NaN,   near_lr_floor = false, min_over_10 = NaN,  min10_near_floor = true),
    ])
    @test agg_none.n_valid == 0
    @test !agg_none.used_fallback
    @test !isfinite(agg_none.safe)
end

@testset "probe_training_strategy inherits configured loss" begin
    base = TrainingStrategy([1, 2, 5], [3, 3, 3], 0.05f0;
                            h_loss_weight = 0.7f0,
                            loss_type = :huber,
                            peak_delta = 0.3f0,
                            peak_lambda = 2.5f0,
                            peak_gamma = 1.2f0,
                            peak_w_max = 6.0f0)

    probe = probe_training_strategy(base, 1)
    @test probe.steps == [1]
    @test probe.durations == [1]
    @test probe.current_steps == 1
    @test probe.noise_scale == base.noise_scale
    @test probe.h_loss_weight == base.h_loss_weight
    @test probe.loss_type == base.loss_type
    @test probe.peak_delta == base.peak_delta
    @test probe.peak_lambda == base.peak_lambda
    @test probe.peak_gamma == base.peak_gamma
    @test probe.peak_w_max == base.peak_w_max
end

@testset "update_train_table keeps comments, edits only [train]" begin
    text = """
    [data]
    lr_start = 9.0   # not touched

    [train]
    epochs = 5
    lr_start = 1e-3   # keep this comment
    batch_size = 4

    [train.strategy]
    lr_final = 7.0
    """
    out = update_train_table(text, Dict("lr_start" => "0.5", "grad_clip" => "2.0"))
    lines = split(out, '\n')
    @test occursin("lr_start = 9.0   # not touched", out)           # [data] untouched
    @test occursin("lr_start = 0.5  # keep this comment", out)
    @test occursin("grad_clip = 2.0", out)
    @test occursin("lr_final = 7.0", out)                       # other table untouched
    ti = findfirst(l -> occursin("[train]", l), lines)
    gi = findfirst(l -> occursin("grad_clip", l), lines)
    si = findfirst(l -> occursin("[train.strategy]", l), lines)
    @test ti < gi < si                                           # inserted inside [train]
    @test_throws ErrorException update_train_table("[data]\na = 1\n", Dict("x" => "1"))
end

@testset "tuning toolkit on the test model (CPU)" begin
    model_path = joinpath(@__DIR__, "..", "test_data", "test_model")
    mktempdir() do tmp
        cfg = joinpath(tmp, "config.toml")
        write(cfg, """
        [data]
        run_name = "tune"
        runs_dir = "runs"
        wflow_model_path = "$(replace(model_path, '\\' => '/'))"
        train_frac = 0.6
        val_frac = 0.2

        [model]
        domain = "river"
        hidden_dim = 8
        nlayers = 1

        [train]
        epochs = 2
        batch_size = 4
        lr_start = 0.001   # peak
        lr_final = 1e-5
        device = "cpu"

        [train.strategy]
        steps = [1, 2]
        durations = [1, 1]
        noise_scale = 0.0
        """)
        ds, ms, ts = parse_run_config(cfg)
        gd = load_tuning_data(ds, ms)
        @test length(gd.graphs) > 3

        setup = make_model_loader(gd, ds, ms, ts; horizon = 1, batch_size = 1000,
                                  seed = 1, shuffle = false)
        @test setup.strategy.steps == [1]
        @test setup.batch_size <= 1000                     # clamped to the number of windows

        lrs, losses, gnorms = lr_range_test(setup; num_steps = 6, lr_min = 1e-6, lr_max = 1e-3)
        @test 1 <= length(lrs) <= 6
        @test length(losses) == length(gnorms) == length(lrs)
        @test issorted(lrs)
        @test all(isfinite, losses)
        csv = write_lr_curve_csv(joinpath(tmp, "curve.csv"), lrs, losses, gnorms)
        @test countlines(csv) == length(lrs) + 1

        s2 = make_model_loader(gd, ds, ms, ts; horizon = 2, batch_size = 2, seed = 2, shuffle = false)
        g = gradient_growth_probe(s2; num_steps = 3, warmup = 1)
        @test isfinite(g) && g > 0

        fit = estimate_batch_size(gd, ds, ms, ts; horizon = 1, candidates = [1, 2])
        @test fit in (1, 2)

        # autotune with every estimate overridden: only the config is rewritten.
        out = autotune(cfg; batch_size = 3, lr_start = 2e-3, lr_peak_decay = 0.5, grad_clip = 1.5)
        @test out == joinpath(tmp, "config_autotuned.toml")
        tuned = TOML.parsefile(out)["train"]
        @test tuned["batch_size"] == 3
        @test tuned["lr_start"] ≈ 2e-3
        @test tuned["lr_final"] ≈ 2e-5
        @test tuned["lr_peak_decay"] ≈ 0.5
        @test tuned["grad_clip"] ≈ 1.5
        @test TOML.parsefile(cfg)["train"]["lr_start"] ≈ 1e-3      # original untouched
        @test occursin("# peak", read(out, String))                 # comment preserved

        # in_place keeps a backup.
        autotune(cfg; batch_size = 2, lr_start = 1e-3, lr_peak_decay = 0.7, in_place = true)
        @test isfile(cfg * ".bak")
        @test TOML.parsefile(cfg)["train"]["batch_size"] == 2
    end
end