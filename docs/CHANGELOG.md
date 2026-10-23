This document serves as a concrete changelog of the code, summarizing briefly the changes of each commit.

## 2026-10-23

- **Formulation-aware stability and conservation metrics (TODO §3h).** New
        `rollout_contraction` (ρ̂, the measured one-step gain of the carried state under
        teacher forcing; `q` rows for theta, `h` for the hybrid, all rows without a mass
        balance) is evaluated per epoch for every model (`losses.val_contraction`, shown in
        the progress meter) next to `h_loss_weight`; the training start logs which of
        `river_q` / `river_h` is the carried state. `metrics.toml` gains `[formulation]` and
        `[stability]` (headline plus presence-based companions: theta `analytic_gain` /
        `q_to_h_amp`, hybrid `backbone_rho_median` / `backbone_rho_max`) and layer-free
        `volume_pbias` / `volume_drift` in `[mass_balance_residual]` (new
        `field_volume_bias`). `final_mb_gain` is no longer written for the hybrid. The hpar
        results CSV / summary replace the `final_amp` columns by `rollout_contraction` and the
        companions (empty cell / absent key when not applicable) and add `mb_formulation`,
        `mb_residual_rms`, `volume_pbias`, `volume_drift`, `h_loss_weight_effective`.
        Tests: [test/test_mb_hybrid.jl](../test/test_mb_hybrid.jl),
        [test/test_pipeline.jl](../test/test_pipeline.jl),
        [test/test_hparsearch.jl](../test/test_hparsearch.jl).
- **Subcatchment error attribution in the analyze stage (TODO §3g).** `analyze_run` (and
        therefore every pipeline and `hpar_search` run) now writes the attribution products
        `val_subcatch_error_map.png`, `_mainstem.png`, `_attributes.png`,
        `_error_attribution.csv` and `_attribute_ranking.csv` (and `val_daterange_subcatch_*`
        for the date-range window). One `subcatch_error_attribution` pass per rollout feeds
        both these and the 3f outlet timeseries; best effort, skipped for a diverged rollout.
        New `TrainSettings` fields `subcatch_attribution` (default `true`) and `subcatch_rank`
        (default `true`, the permutation ranking), read from `[eval]` (or `[train]`), saved in
        `train_settings.toml` and sweepable (`eval.subcatch_attribution`). New export
        `write_subcatch_error_attribution`, shared with `subcatch_error_attribution_run`.
        Tests: [test/test_attribution.jl](../test/test_attribution.jl).

## 2026-10-21

- **Mass-balance redesign: `muskingum_continuity` formulation (TODO "Mass-balance
        redesign").** New `MuskingumContinuityHybrid` (`build_muskingum_continuity`) behind
        `mb_formulation = "muskingum_continuity"`: storage `h` is the carried state, advanced
        by an implicit-Euler reservoir-cascade backbone (sparse `P`, spectrum `1/(1+dt/K) < 1`)
        plus a bounded learned residual (`mb_residual_bound`, default 0.5), and `q` is the
        exact continuity readout `(Iâˆ’A)â»Â¹(I_lat âˆ’ dS/dt)`, never fed back (`encoder_state`
        zeroes the `q` row in the encoder input). Wired into `_loss_components`,
        `_mb_amplification`, `_rollout_mb_diagnostics`, `layer_mb_analysis`
        (`metrics/mb_backbone_contraction_by_node.csv`), `MB_FORMULATIONS`,
        `ModelSettings.mb_residual_bound`, `settings_from_config` and the templates (incl. a
        `model.mb_formulation` sweep example). Default formulation and checkpoints unchanged.
        Tests: [test/test_mb_hybrid.jl](../test/test_mb_hybrid.jl), a pipeline case in
        [test/test_pipeline.jl](../test/test_pipeline.jl).- **Selectable nonlinear input scaling (TODO "Input feature normalization").** New
	`[model.normalization]` config table (`variable = scheme`; `zscore` default,
	`log1p_zscore`, `signed_log_zscore`, `asinh_zscore`) for time-varying
	state/forcing variables, exposed as `ModelSettings.normalization` and sweepable in the
	hpar search (`model.normalization.river_inwater`; `_set!` now creates missing
	intermediate tables). Encoder-input-only: `build_wflow_graph(...; normalization)`
	fits `scheme/scale/nl_mean/nl_std` on the training window (also in `norm_stats.toml`)
	and the model prepends a non-trainable `EncoderInputTransform` only when a scheme is
	set, so stored data, mass balance, loss and de-normalization are unchanged. New exports
	`nonlinear_forward/inverse`, `nonlinear_stats`, `parse_normalization`,
	`encoder_input_transform`, `with_input_transform`. Tests:
	[test/test_normalization.jl](../test/test_normalization.jl).
- **Outlet timeseries at the attribution outlets (TODO §3f).** `analyze_run` now also
	writes `plots/outlet_timeseries*.png` and `metrics/outlet_timeseries*.csv`
	(and `_daterange` variants): predicted vs true discharge at the whole-basin outlet
	and the retained subcatchment-attribution outlets, best effort (skipped with a notice
	for no `river_q`, a diverged rollout or missing `staticmaps.nc` inputs).
	`plot_downstream_timeseries` gained a `nodes` kwarg, `subcatch_error_attribution` a
	`rank = false` kwarg; new helpers `nodes_for_cells`, `attribution_outlet_cells`,
	`rollout_diverged`; `analyze_run` gained `outlet_gauge_var` / `outlet_min_area_km2`.

- **Subcatchment error attribution (TODO §3e).** New post-hoc diagnostic
	[src/attribution.jl](../src/attribution.jl) with CLI
	[scripts/subcatch_error_attribution.jl](../scripts/subcatch_error_attribution.jl):
	tessellates the domain by outlet cells (gauge raster or LDD-derived confluences),
	builds the outlet tree, reports per-outlet skill and the advected-vs-local split
	`e_local = dQ(o) - sum dQ(inlets)`, and Spearman-ranks static attributes against
	local error. Writes `metrics/<split>_subcatch_{error_attribution,attribute_ranking}.csv`
	and three plots. `resolve_staticmaps` now also finds `models/` for runs nested
	one level deeper (`experiments/<group>/<run>`). Tests in
	[test/test_attribution.jl](../test/test_attribution.jl).

- **Completed the 4-stage pipeline + mass-balance code restructuring** (TODO
	"Code restructuring", R1–R7). `run_wflow_gnn` is now
	`run_stage(AllStages(), …)` over four independently-runnable stages —
	**preprocess → train → infer → analyze** — dispatched through a `Stage` type
	hierarchy in [src/run.jl](../src/run.jl) and driven by the new
	[scripts/pipeline.jl](../scripts/pipeline.jl) `<prep|train|infer|analyze|all>`
	CLI. Stages exchange on-disk artifacts (`prep/preprocessed.jld2`, `model/`,
	`output/*.nc`, `metrics/`+plots); a stage uses an earlier stage's in-memory
	result when composed, else loads its predecessor's artifacts, so each runs
	standalone. Golden-run artifact sets are identical to the pre-refactor fused
	path (verified on `sava_small_v081`, CPU+GPU).
- **Mass-balance code separated from the generic GNN.** New
	[src/sparse_ops.jl](../src/sparse_ops.jl) (the `_topology_mul` rrule +
	CuSPARSE/CPU sparse helpers) and [src/massbalance.jl](../src/massbalance.jl)
	(the layer, `build_massbalance`, diagnostics). Formulations are now a
	`MassBalanceFormulation` dispatch hierarchy — `ThetaKinematicWave` (wraps the
	old `MassBalanceLayer`, kept as a compat alias) and `NoMassBalance` (null
	object replacing the `nothing` sentinel). The forward pass collapses to
	`decoder_outputs`/`extra_decoder_features`/`apply_formulation`/`floor_state`
	dispatch; every `isnothing(mass_balance)` branch across
	`gnn.jl`/`run.jl`/`rollout.jl`/`training.jl`/`strategy.jl` is gone. Selected via
	`mb_formulation` (`"theta_kinematic_wave"` default / `"none"`);
	`enforce_mass_balance` and old `mass_balance = nothing` checkpoints still load.
- **Layer-free, post-hoc mass-balance analysis.** `mb_residual_diagnostics`
	([src/postprocess.jl](../src/postprocess.jl)) recomputes the kinematic-wave
	residual from predicted q/h NetCDF fields + staticmaps, so MB analysis works on
	**any** trained model; the layer-internal diagnostics dispatch on the
	formulation (no-op for `NoMassBalance`). New artifacts `metrics/mb_residual.csv`,
	`metrics/mb_residual_by_node.csv`, `[mass_balance_residual]` in `metrics.toml`.
- **Scripts consolidated into `src/`.** [src/analysis.jl](../src/analysis.jl)
	(`analyze_run`, `resolve_staticmaps`, `spatial_eval_run`, `regen_mb_diagnostics`),
	[src/consistency.jl](../src/consistency.jl) (the layer-free MB routing/residual
	core of `check_mass_balance*.jl`), and [src/tune.jl](../src/tune.jl) (LR-range
	test / batch-size / autotune toolkit). `scripts/*` are now thin CLIs;
	`smoke_test_sparse.jl` moved into the test suite; `plot_ldd` promoted to
	`src/plot.jl`; one-off scripts archived under `scripts/oneoff/`.
- **Fixed lateral-inflow (inwater) scaling in `ThetaKinematicWave`** (R5b):
	`inwater_phys` now includes the `postscale["river_inwater"]` upstream-area
	factor that preprocessing divides out before z-scoring (new `mb_inwater_unscale`
	toggle, default `true`; old checkpoints default to the legacy behaviour). Also
	fixed `_rollout_mb_diagnostics` passing the predicted (not pre-step) state to
	`mb_diagnostics` — this changes `h_raw`/floor/`net_flux`/volume-budget numbers
	for all runs (prior `docs/EXPERIMENTS.md` values are not comparable).
- Performance acceptance gate passed: `benchmark_message_passing` +
	`benchmark_rollout` show no regression beyond noise vs. baseline `829c432`
	(see [PERFORMANCE.md](PERFORMANCE.md), STATUS AS OF 21-10-2026). The
	`_topology_mul` rrule, `SparseConv` three-path dispatch + `precompute_batched`,
	CuSPARSE CSR conversions and CPU-build-before-`gpu` ordering are byte-for-byte
	preserved.

## 2026-09-19

- Fixed a hard-cap bug in the smooth `river_h` floor. `_nonnegative_floor` in
	[src/gnn.jl](../src/gnn.jl) computed the softplus as
	`β·log1p(exp(clamp(x/β, -40, 40)))`; the overflow guard also clamped the
	*positive* side, capping depth at `β·40 = softness·40` (e.g. **2.0 m** at
	`h_floor_softness = 0.05`) on every node. This produced the strict 0↔2 rail
	seen in E12 `river_h` (outlet had 105/222 samples pinned at exactly 2.0; nodes
	74/152/183 also capped at 2.0 despite different per-node σ/μ/postscale).
	Replaced with the overflow-safe identity `softplus(z) = max(z,0) + log1p(exp(-|z|))`
	(exp argument always ≤ 0, so no overflow; large depths pass through as `β·z = x`
	with no ceiling). **Implication:** E12 (`mb_smooth_h_floor = true`) stability /
	boundedness results are confounded by the old cap and must be re-run.

## 2026-09-11

- Fixed autotune ignoring the configured loss (commit `4731de3`, "fix autotune
	with huber loss"): `make_model_loader` in
	[scripts/lr_range_test.jl](../scripts/lr_range_test.jl) now builds its probe
	strategy via `probe_training_strategy`, forwarding the full loss config
	(`loss_type`, `peak_delta`, `peak_lambda`, `peak_gamma`, `peak_w_max`) instead
	of dropping everything but `noise_scale`/`h_loss_weight`. The effective loss is
	logged, and `test/test_lr_autotune.jl` asserts the strategy inherits
	`loss_type` from `ts.strategy`. Standing limitation (unchanged): the range test
	is teacher-forced / horizon-1 and blind to multi-step rollout collapse — treat
	the autotuned LR as an upper bound.
- Added the system water-volume validation diagnostic (item 2d), computed for
	BOTH prediction and ground truth. `volume_budget_diagnostics` in
	[src/rollout.jl](../src/rollout.jl) reduces the per-node physical series into
	storage `V_pred(t)` / `V_true(t)` (`Σ h·w·l`, with `w·l = postscale_q/postscale_h`)
	and budget-closure series (cumulative inflow/outflow, `ΔV`) for both
	trajectories, using the `argmax(upstream_area)` single-outlet convention.
	Headline scalars `volume_pbias` and `volume_drift` are written to the run
	metrics TOML via `evaluate_and_write` ([src/run.jl](../src/run.jl)), and
	`plot_volume_budget` ([src/plot.jl](../src/plot.jl)) renders the storage + budget
	rows with a companion CSV.
- Added longitudinal upstream timeseries plots (item 2e). `plot_downstream_timeseries`
	([src/plot.jl](../src/plot.jl)) now selects `K` percentile-spanning active nodes
	over `upstream_area` (outlet → headwater), renders per-node series with a
	per-panel catchment inset, and exports per-node CSVs. `K` is configurable via
	`[eval].upstream_points` (default 5), threaded through
	[src/run.jl](../src/run.jl) and persisted on `TrainSettings`.

## 2026-09-07

- Made the peak-weighted Huber loss GPU-safe: `peak_weighted_huber_loss`,
	`peak_weight_matrix` and `_resolve_peak_thresholds` now allocate fallback
	weight/threshold arrays on the same device as their inputs (device-matched
	`similar`/`fill!` helpers) instead of host-side `ones`/`Float32[...]`. Fixes a
	CUDA `KernelError: passing non-bitstype argument` that aborted `:huber`
	hparsearch runs on GPU.
- Added a CUDA regression for the Huber loss in `test/test_strategy.jl` (skipped
	when CUDA is unavailable). Canonical suite passes (`test/runtests.jl`: 395/395).
- Closed the rollout-path discrepancy investigation: a same-start regression
	confirms the batched fixed-horizon anchor path and the date-range trajectory
	path agree to tolerance when seeded from the same initial state; both run from
	the restored best-epoch checkpoint. The residual divergence is a genuine model
	instability, not a code-path mismatch. See DECISIONS.md.

## 2026-09-04

- Fixed hparsearch config-parser drift by extracting a shared parser helper
	(`settings_from_config`) in `src/run.jl` and reusing it from `src/hparsearch.jl`.
- Added regression coverage for search-space override propagation in
	`test/test_hparsearch.jl` (included in `test/runtests.jl`).
- Verified `train.grad_clip` search overrides propagate to saved
	`train_settings.toml` (2-combo check: `1.0` and `0.1`).
- Added peak-weighted Huber support in training config parsing and execution:
	`loss_type`, `peak_delta`, `peak_lambda`, `peak_gamma`, `peak_w_max`, and
	per-node `peak_node_stats` threading through `run_wflow_gnn`/`train_model!`.
- Added Tier-1 peak diagnostics in training (`peak_epoch_diagnostics`,
	`peak_weight_matrix`) and logging/history outputs.
- Added Tier-2 `river_q` performance reporting (`river_q_performance_metrics`)
	and wiring into run evaluation + metrics TOML output.
- Completed item 2c fixed-horizon diagnostics: per-anchor arrays + anchor
	start-flow percentile tagging + persisted `metrics/fixed_horizon_anchors.csv`.
- Added per-epoch anti-masking fixed-horizon guards in training history:
	`val_peak_ratio_frac_gt2` and `val_fixed_rmse_highflow`; wired into
	fixed-horizon CSV output and run `metrics.toml` summary.
- Updated training tests for the new fixed-horizon guard histories; canonical
	suite passes (`test/runtests.jl`: 390/390).

## Pre-2026-09-04 (baseline snapshot)

*This entry back-fills the foundational state that existed before the
commit-by-commit log above began. It is a summary of already-shipped work, not a
single commit; the full reasoning lives in [DECISIONS.md](DECISIONS.md) and the
research notes under `docs/notes/`.*

- Encode–process–decode GNN with a physics decoder; neighbour aggregation via
	`SparseConv` cuSPARSE SpMM, with a precomputed block-diagonal adjacency
	(`precompute_batched`) for batched training (built on CPU before `Flux.gpu`).
- Hard mass-balance layer (`MassBalanceLayer` in [src/gnn.jl](../src/gnn.jl)):
	decoder emits only `Δq` (`out_dim = 1`), `river_h` derived analytically from
	the fully-implicit kinematic-wave balance each step (verified vs real Wflow,
	median |residual| ≈ 0). Configurable `mb_theta` (default `1.0` = fully
	implicit). Positivity floors applied in physical units inside the layer.
- Gradient weighting `h_loss_weight = σ_h/(dt·σ_q)` to equalise `q`/`h`
	magnitudes (runtime-derived, not persisted).
- Custom `_topology_mul` adjoint returning `NoTangent()` for the adjacency — the
	out-of-memory fix that avoids a dense `∂A` (~17 GB at batched `(B·N)²`) — plus
	`Functors.@functor` field restriction and explicit `Flux.gpu`/`Flux.cpu`
	overloads for the sparse topology matrices.
- Rollout-length curriculum (`steps = [1,2,5,8,10]`) with an SGDR-style
	warm-restart LR schedule (`lr_warmup_epochs`, `lr_peak_decay`) and BPTT via
	`Zygote.checkpointed`; gradient clipping + adaptive backoff on
	non-finite/unstable epochs.
- Ensemble (batched) rollout path for GPU throughput (~2.5× per-member gain).
- Autotune path (`scripts/autotune_train.jl`): horizon-1 LR range test +
	batch-size estimator.