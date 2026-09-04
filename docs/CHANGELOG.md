This document serves as a concrete changelog of the code, summarizing briefly the changes of each commit.

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