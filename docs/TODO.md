Concrete todos of changes and features to be implemented.

Distilled from the `slurm-246431` (full-basin divergence) post-mortem and the
peak-discharge research notes. Full detail and parameter-estimation recipes in
[notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md). Each item should land
behind a toggle so it can be A/B-tested against current behaviour.

---

# Code restructuring — 4-stage pipeline + mass-balance separation

See [DECISIONS.md](DECISIONS.md) → PROPOSED 19-10-2026 and the full design in
[notes/pipeline_and_massbalance_redesign.md](notes/pipeline_and_massbalance_redesign.md).
Split the monolithic `run_wflow_gnn` into four independently-runnable stages that
exchange on-disk artifacts, separate MB code from the generic GNN, and prefer
multiple dispatch over `if/else`. Do the steps **in order** (each is individually
testable and reversible); after each, tests must stay green and a small golden run
(`test_sava_small_v081`) must reproduce the current artifact set.

## R1. Extract shared/MB code into their own files (pure code move)

- [x] New `src/sparse_ops.jl`: move `_topology_mul` + its `ChainRulesCore.rrule`
      verbatim (included first so both `massbalance.jl` and `gnn.jl` can use it).
      **Do not alter** — this is the top autodiff-memory optimisation.
      (Also hosts the shared `_to_cusparse`/`_to_cpu_sparse` helpers, which both
      `SparseConv` and the MB layer's `gpu`/`cpu` overloads use.)
- [x] New `src/massbalance.jl`: move `MassBalanceLayer` + its `@layer`/`@functor`/
      `gpu`/`cpu` overloads, the forward operator, `mb_decoder_features`,
      `mb_diagnostics`, `_floor_q_norm_nonnegative`, `_nonnegative_floor`, and
      `precompute_batched(::MassBalanceLayer, …)` out of `src/gnn.jl`.
- [x] Move the inlined MB-construction block from `build_gnn_model`
      ([src/run.jl](../src/run.jl)) into an MB factory `build_massbalance(...)` in
      `src/massbalance.jl`. (Takes scalars `θ`/`smooth_h_floor`/`h_floor_softness`
      rather than `ModelSettings`, since `ModelSettings` is defined later in
      `gnn.jl`.)
- [x] Update `include`/`export` order in
      [src/WflowRoutingGNN.jl](../src/WflowRoutingGNN.jl); `gnn.jl` keeps only the
      generic `SparseConv`/`WflowGNN`/encoder-processor-decoder plumbing.
- Touch: `src/sparse_ops.jl` (new), `src/massbalance.jl` (new), `src/gnn.jl`,
      `src/run.jl`, `src/WflowRoutingGNN.jl`. No behaviour change; tests green.
- [x] R1 validation (2026-10-19): full test suite 4472/4472 pass. Golden run
      (`sava_small_v081`, seed 42, 5 epochs, CPU and GPU) vs. pre-R1 `HEAD`:
      predictions (`river_q`/`river_h`, train+val), model weights, losses and
      metrics CSVs are bit-identical on both devices. Rollout benchmark
      (300 steps, CPU/GPU, single/ensemble, 2 interleaved reps) shows
      before/after differences within run-to-run noise.

## R2. Mass-balance formulations as a dispatch hierarchy

- [x] Add `abstract type MassBalanceFormulation`; `struct NoMassBalance` (null
      object replacing the `nothing` sentinel) and `struct ThetaKinematicWave`
      wrapping the current `MassBalanceLayer` fields.
      (`const MassBalanceLayer = ThetaKinematicWave` kept as a compat alias.)
- [x] Replace the branch points with dispatch methods: `decoder_outputs(f, n_state)`,
      `apply_formulation(f, g, state, forcing, forcing_next, Δ)`,
      `extra_decoder_features(f, g, state, forcing)` (returns `()` for
      `NoMassBalance`), `floor_state(f, g, raw)` (identity for `NoMassBalance`).
      (Plus `extra_decoder_dim`, `has_mass_balance`, `mb_h_loss_weight`.)
- [x] Make `WflowGNN.mass_balance :: MassBalanceFormulation` (never `nothing`);
      collapse the forward pass to encode → process → `decoder(vcat(h,
      extra_decoder_features(...)...))` → `apply_formulation(...)`.
      (`WflowGNN{M}` is parametric on the formulation.)
- [x] Keep the `θ == 1` fast path as a cheap inner scalar branch (do **not**
      over-dispatch a scalar); keep it bit-identical.
- [x] Remove every `isnothing(mass_balance)` branch across `gnn.jl`, `run.jl`,
      `rollout.jl`, `training.jl` (and `strategy.jl`).
- [x] Map `ModelSettings` → formulation via a `mb_formulation` key in
      `settings_from_config` (keep `enforce_mass_balance` working for back-compat).
      (`mb_formulation` ∈ {`"theta_kinematic_wave"` (default), `"none"`};
      `mb_formulation_kind(ms)` resolves it, `enforce_mass_balance = false` forces
      `"none"`.) Old checkpoints with `mass_balance = nothing` still load.
- Touch: `src/massbalance.jl`, `src/gnn.jl`, `src/run.jl`, `src/rollout.jl`,
      `src/training.jl`, `src/strategy.jl`. Assert **bit-identical** outputs vs
      pre-refactor on a fixed seed.
- [x] R2 validation (2026-10-19): full test suite 4515/4515 pass. Golden run
      (`sava_small_v081`, seed 42, 5 epochs) vs. post-R1 code, four variants
      (CPU; GPU; GPU `enforce_mass_balance=false`; GPU `mb_theta=0.5` +
      `mb_augment_decoder=true`): model weights, train/val predictions and all
      metrics CSV/TOML are bit-identical (only timing fields and the new
      `mb_formulation` key in `model_settings.toml` differ). Interleaved
      forward / fwd+bwd / train-step / 300-step rollout benchmarks (CPU+GPU,
      2 reps) are within run-to-run noise of the pre-R2 code.

## R3. Split preprocessing into its own stage + artifact

- [ ] `preprocess_run(ds, ms)`: `build_wflow_graph` + `make_horizon_dataset` +
      peak-node stats + normalized-tail diagnostics.
- [ ] Serialize a versioned `PrepBundle` to a new `prep/` dir
      (`preprocessed.jld2` + the existing dataset/`grid.jld2`): `norm_stats`,
      `postscale`, `static_arr`, `dt`, graph topology (edge index, `A_routing`),
      schema id, **and the node↔cell mapping** (`grid`) — the inverse-gather key
      stage 4 uses. Tag with a `prep_schema` version alongside `wflow_schema`.
- [ ] Factor `train_run(prep, ms, ts)` out of `run_wflow_gnn` so it reads only
      `prep/` (never the raw wflow outputs) and writes the current `model/` set.
- Touch: `src/pipeline.jl` (new) or `src/run.jl`, `src/preprocess.jl`.

## R4. Split inference from analysis

- [ ] `infer_run(model, prep, split, val_daterange)`: rollout (`evaluate_trajectory`)
      + `regrid` + write NetCDF **only**. Output `<split>_pred.nc`/`<split>_true.nc`
      as gridded `(lon,lat,time)` NetCDF matching wflow's `output.nc` layout as
      closely as possible (grid, variable names, time axis). No metrics, no node
      dump.
- [ ] `analyze_run(output, prep, model-settings)`: everything currently in
      `evaluate_and_write` **after** the rollout — spatial metrics + summary,
      downstream timeseries (+ inset), overprediction-vs-ramp,
      `river_q_performance_metrics`, validation movie, MB diagnostics, volume
      budget, `metrics.toml`. Recover node arrays by gathering river cells from the
      NetCDFs via the prep-owned node↔cell mapping (handle inactive/NaN-fill cells;
      reproduce `ldd_to_graph` column-major node order exactly).
- [ ] Compose `run_wflow_gnn = analyze_run ∘ infer_run ∘ train_run ∘ preprocess_run`;
      verify artifact-set equality against a golden pre-refactor run.
- [ ] Point `hpar_search` + checkpoint-full-eval at `infer_run`+`analyze_run`
      instead of the fused `evaluate_and_write`.
- Touch: `src/pipeline.jl`/`src/run.jl`, `src/postprocess.jl`, `src/hparsearch.jl`.

## R5. Layer-free, post-hoc mass-balance residual diagnostic

- [ ] Add `mb_residual_diagnostics` that recomputes the kinematic-wave residual
      `r(t) = h[t+1] − h[t] − dt/(w·l)·(ΣQ_up + Q_iw − Q_out)` **from predicted
      q/h fields** (NetCDF) + `staticmaps` constants + `ldd_to_graph` routing —
      works for **any** model regardless of whether an MB layer was present.
- [ ] Dispatch the existing layer-internal diagnostics (`rollout_mb_diagnostics`:
      net_flux, h_raw-before-floor, floor fraction, stiffness) on the formulation
      type: real method for MB formulations, no-op for `NoMassBalance`.
- [ ] Stage 4 always runs the residual diagnostic; runs the layer-internal one
      only for MB models. `metrics.toml` MB fields degrade gracefully (residual
      for all; floor/stiffness only for MB).
- Touch: `src/massbalance.jl` (or `src/analysis.jl`), `src/postprocess.jl`,
      `src/run.jl`.

## R6. Consolidate `scripts/` into `src/` + thin CLIs

- [ ] Promote post-hoc helpers `resolve_staticmaps` + `load_regrid_nc` (inline in
      [scripts/spatial_eval.jl](../scripts/spatial_eval.jl)) into the analysis
      layer; stage 4, `spatial_eval`, and `subcatch_error_attribution` share one
      implementation.
- [ ] Promote the layer-free MB routing/residual core from
      [scripts/check_mass_balance.jl](../scripts/check_mass_balance.jl)
      (`compute_upstream_q`, 3-scheme comparison) and
      [scripts/check_mass_balance_details.jl](../scripts/check_mass_balance_details.jl)
      (`upstream_sum`, 9 consistency checks) into `src/` — this is the prototype
      for R5's `mb_residual_diagnostics`; have both scripts call the promoted core.
- [ ] Promote the LR/batch tuning toolkit (`lr_range_test`, `gradient_growth_probe`,
      `recommend_lr`, `aggregate_lr_recommendations`, `estimate_batch_size`,
      `autotune`) from [scripts/lr_range_test.jl](../scripts/lr_range_test.jl) +
      [scripts/autotune_train.jl](../scripts/autotune_train.jl) into a new
      `src/tune.jl` public API; scripts become thin CLIs.
- [ ] Retarget thin drivers to the new stage API: `train.jl`→`train_run`,
      `hparsearch.jl`, `spatial_eval.jl`+`regen_mb_diagnostics.jl`→"stage 4 only"
      (`scripts/pipeline.jl analyze`).
- [ ] Update the benchmark/profiler harnesses (`benchmark_message_passing`,
      `benchmark_rollout`, `benchmark_inference_vs_train`, `diag_overhead`,
      `profile_rollout`, `typecheck_forward`) to the new module/stage API so they
      keep compiling — they are the §4/§5–7 perf acceptance gate.
- [ ] Migrate `smoke_test_sparse.jl` into `test/test_gnn.jl` (exercises the sparse
      batching paths the perf optimisations depend on); factor `plot_ldd.jl`'s
      network drawing into `src/plot.jl` (reused by the timeseries network inset);
      archive remaining one-offs under `scripts/oneoff/`.
- Touch: `src/analysis.jl`/`src/postprocess.jl`, `src/massbalance.jl`,
      `src/tune.jl` (new), `src/plot.jl`, `scripts/*`, `test/test_gnn.jl`.

## R7. Staged CLI entry point

- [ ] `scripts/pipeline.jl <stage> <config.toml>` with
      `stage ∈ {prep, train, infer, analyze, all}`, dispatched via a `Stage` type
      hierarchy (`run_stage(::Preprocess, cfg)` … one method per stage; `all` folds
      over the four) — not an `if stage == …` ladder.
- [ ] Decide `run_dir` layout: prefer **additively** introducing `prep/` over
      renaming existing dirs, to avoid breaking evaluation/performance-agent and
      plotting-script consumers; audit consumers first.
- Touch: `scripts/pipeline.jl` (new), `src/pipeline.jl`.

## R — performance acceptance gate (run around the whole restructuring)

- [ ] Run `scripts/benchmark_message_passing.jl` + `scripts/benchmark_rollout.jl`
      **before and after** and confirm no regression. Preserve byte-for-byte: the
      `_topology_mul` rrule, `SparseConv` three-path dispatch + `precompute_batched`
      (both `SparseConv` and MB), CuSPARSE CSR (`Int32` colptr) conversions,
      zero-copy per-node `reshape` when `n_rep == 1`, explicit `gpu`/`cpu`
      overloads with Functors restricted to trainable fields, and CPU-build of the
      block-diagonal before `Flux.gpu`.

---

# Enforce non-negative predicted discharge (physics floor)

From E3: the free-rollout produced **unphysical negative discharge** (daterange
`river_q_pred` min ≈ −6 m³/s across cells) as the model collapsed toward zero.
River discharge cannot be negative.

- **Current code (partial floor already exists — do NOT re-implement it).** The
  MB layer already floors its *internal* physical discharge at zero
  (`q_phys_new = max.(0f0, …)`, [src/gnn.jl](../src/gnn.jl#L281), since commit
  `5cb4922`), and `rollout_mb_diagnostics` reports that floored copy. The gap is
  that this floored value does **not** propagate: the MB layer returns only `h`,
  so the **state/decoder q that carries to the next step is un-floored**, and
  `fixed_horizon_metrics` reconstructs its reported q directly from `q_norm`
  **without** a floor ([src/rollout.jl](../src/rollout.jl#L483)) — hence E3's
  negative anchor/daterange q.
- [x] Floor the **propagating** predicted `river_q` (the decoder/state q, not
      just the internal h-path copy) so negative flow cannot feed the next step —
      e.g. `softplus`/`relu`/`max(·,0)` on the reconstructed absolute flow, chosen
      so it does not break the MB water-balance derivation of `river_h` or its
      gradients.
- [x] Floor the reported q in `fixed_horizon_metrics`
      ([src/rollout.jl](../src/rollout.jl#L483)) consistently with the state floor.
- [ ] Verify it does not mask instability (a floored-but-still-collapsing model
      should still be detectable via PBIAS / peak_ratio, not hidden by clamping).
- Touch: `src/gnn.jl` (decoder / state q), `src/rollout.jl`
  (`fixed_horizon_metrics`), a test asserting `q ≥ 0`.

---

## 2b. Validation metrics — remaining items

Tier 1 (peak diagnostics: `peak_epoch_diagnostics`) and Tier 2
(`river_q_performance_metrics`: KGE, r/α/β, MAE, PBIAS, event peak + FHV, pooled)
are **done and wired in** (see [CHANGELOG.md](CHANGELOG.md) 2026-09-04).
Remaining:

- [ ] **Per-gauge (not pooled) reporting** — blocked: no "gauge" concept exists
      in the codebase yet; needs a design decision (which nodes count as gauges)
      before implementing. `river_q_performance_metrics` currently reports pooled
      (all-river-node) metrics only.
- [ ] **Tier 3 detection metrics (POD / FAR / CSI)** — `event_detection_metrics`
      exists as a unit-tested helper but is **not wired in**; deferred until a
      detection/exceedance head exists in the codebase.
- Touch: `src/postprocess.jl`, `src/run.jl` (when unblocked).

## 3. Pushforward / detached-rollout trick (toggle)

- **Definitions (both cut the BPTT tape to O(1) via stop-gradient — this is the
  `Flux.ignore_derivatives` primitive already used in `src/strategy.jl`, NOT a
  forward-mode/JVP linearisation of `f_θ`):**
  - **`:pushforward` (Brandstetter et al. 2022).** Unroll `k` steps with the
    *actual* nonlinear `f_θ` but under stop-gradient, then take **one**
    gradient-carrying step from that self-generated state and score **only that
    final step**: `x_{t+k-1} = detach(f_θ∘…∘f_θ(x_t))`, loss on
    `f_θ(x_{t+k-1})` vs `y_{t+k}`. The differentiated step sees a
    distribution-shifted (self-rolled) input → teaches robustness to the model's
    own rollout error. Supervises the endpoint only.
  - **`:detached` (truncation-length-1).** Detach the incoming state at **every**
    step, take one differentiable `f_θ` application, score **that** step, then
    detach its output before the next: `x̂_{t+1} = f_θ(detach(x_t))`, loss each
    step. Keeps per-step hydrograph supervision (matters for the dense daily q/h
    targets + peak-weighted loss) with O(1) tape depth. `k` is fixed at 1 — there
    is no configurable TBPTT window in this design.
- [x] `rollout_grad = :full_bptt | :pushforward | :detached`, default full-BPTT.
- [x] Detached (truncation-length-1) variant preferred: per-step loss, backprop
      one step, O(1) tape depth — should make gradient checkpointing unnecessary.
- [ ] Benchmark step time & peak memory vs. full-BPTT at `steps=10`.
- **Note:** fixes rollout-depth instability & training time, *not* the epoch-2
  single-step blow-up (that is items 1 & 2). Pair it with them.
- Touch: `src/strategy.jl`, `src/training.jl`, config.

## 3c. Hybrid pushforward + teacher-forced 1-step loss

**Motivation (E6).** Pure `pushforward` is the first lever to lower `amp`
(11.0→6.4) and give a finite fixed-horizon rollout, but it supervises **only the
detached endpoint**, so in deep curriculum phases the 1-step map and `river_h` are
never directly trained — E6 regressed `val_q_1step` 0.0144→0.165 and `river_h`
spatial NSE −7.9→−96. Decouple long-horizon self-correction from 1-step/h
supervision with a convex combination:
`L = α·L_pushforward + (1−α)·L_1step`.

- [x] Add the **teacher-forced** 1-step term: for `t in 1:k`, forward from the
      **ground-truth** state `batch[t].ndata.state` (NOT the rolled/detached
      state — that is the `:detached` signal, which E6 showed is a net negative,
      `amp`↑12.7) and score `step_loss` vs `targets[t]`. O(1) tape, no BPTT; reuse
      the existing `one_step_loss` machinery ([src/strategy.jl](../src/strategy.jl#L516)).
- [x] Only active when `rollout_grad = :pushforward`; keep `step_loss` shared so
      `loss_type` / `h_loss_weight` / peak params apply consistently to both terms.
- [x] Config knob `pushforward_tf_weight = (1−α)`, **default 0.0** (= today's pure
      pushforward exactly; no behaviour change for existing configs). Prefer a
      weight over a new enum value so `α=1` continuously recovers pure pushforward.
- [ ] Validate with an **α sweep** on the E6-pushforward base
      (`pushforward_tf_weight ∈ {0.0, 0.25, 0.5, 0.75}`). Win condition: recover
      `val_q_1step`/`river_h` toward full_bptt **while** `amp` stays ~6.4 and the
      fixed-horizon RMSE stays finite (i.e. break the E6 trade-off).
- **Prerequisite** for the deeper curriculum schedule (extending `steps` toward
  `eval_horizon`): decoupled 1-step/h supervision is what lets the deep phases be
  front-loaded without starving h.
- Touch: `src/strategy.jl` (`loss_function` pushforward branch), `src/training.jl`
  / `TrainingStrategy` (the weight field), config.

---

# Diagnostics & plotting

## 3e. Subcatchment error attribution (post-hoc; which subcatchments inject error)

**Why (see [DECISIONS.md](DECISIONS.md) → PROPOSED 19-10-2026 and
[notes/subcatchment_error_attribution.md](notes/subcatchment_error_attribution.md)).**
Tests the hypothesis that *some subcatchments contribute disproportionately to
downstream error* and that this correlates with WFLOW subcatchment attributes.
Raw per-outlet NSE conflates **locally injected** error with **advected** error
(discharge accumulates), so the key quantity is the **incremental** error each
subcatchment adds. Post-hoc like [scripts/spatial_eval.jl](../scripts/spatial_eval.jl)
— no retraining, primary method needs no forward pass. **New analysis script**
(e.g. `scripts/subcatch_error_attribution.jl`), reusing `resolve_staticmaps`,
`load_regrid_nc`, `ldd_to_graph`, `river_q_performance_metrics`.

- [ ] **Subcatchment definition + size filter.** Route (A, preferred): use
      WFLOW-native `wflow_subcatch` (or a gauge-set variant) with outlets from
      `wflow_gauges` / `idx_out`. Route (B, fallback): delineate incremental
      catchments from the LDD tree, splitting at confluences where both branches
      exceed an upstream-area threshold. Keep only subcatchments whose incremental
      `wflow_uparea` (or median outlet flow) exceeds a **configurable threshold**
      ("sufficiently large" — also dodges the dry-headwater floor regime where NSE
      is degenerate).
- [ ] **Subcatchment tree from LDD.** Build which subcatchment drains into which
      (immediate upstream sub-outlets per outlet) via `ldd_to_graph`. **Verify the
      partition is a proper tessellation** (every cell in exactly one subcatchment;
      all non-local inflow passes through tracked sub-outlets) — the incremental
      form below depends on it.
- [ ] **Per-outlet skill (baseline).** Extract pred/true `q` (and `h`) at each
      retained outlet; compute NSE / KGE(+r,α,β) / PBIAS / peak_ratio / RMSE via
      `river_q_performance_metrics`. This is total (local+advected) error, not yet
      an attribution.
- [ ] **Incremental (contributed) error — the crux.** For subcatchment `S` with
      outlet `o` and immediate upstream sub-outlets `{u_i}`,
      `e_local(S,t) = [Q_pred(o)−Q_true(o)] − Σ_i [Q_pred(u_i)−Q_true(u_i)]`.
      Report `rms(e_local)`, signed `mean(e_local)`, and a **specific injection**
      (`rms(e_local)` ÷ incremental Σ inwater or incremental area) so sizes are
      comparable. Also report the **advected-vs-local split** (`rms(Σ upstream
      error)` vs `rms(e_local)`) per outlet.
- [ ] **(Optional Tier-2 counterfactual — needs re-inference, still no
      retraining.)** Re-run the trained model with *true* upstream inflow
      substituted at `S`'s boundary; measure outlet error to isolate local skill
      exactly. Inference-time intervention on fixed weights.
- [ ] **Attribute aggregation (per subcatchment).** Outlet-cell value
      (`wflow_uparea`, `wflow_streamorder`, `RiverSlope`); incremental mean/median
      (`KsatVer`, `SoilThickness`, `M`, `f`, `PathFrac`, `WaterFrac`, `Slope`,
      `N_River`, `RiverSlope`); presence (reservoir `ResTrapEff>0`, glacier
      `wflow_glacierfrac>0`); nonlinearity (fraction of time `Q > qbankfull`).
- [ ] **Attribute ranking.** Spearman rank-correlate each attribute vs the
      local-error metric across subcatchments; scatter per attribute with ρ; rank
      by |ρ|. Report ρ CI/p; flag low power + multiple-comparison on small basins —
      run on **`sava_v081`** (many subcatchments) for a real test, not just
      `sava_small`.
- [ ] **Outputs (mirror `spatial_eval.jl`).** `metrics/<split>_subcatch_error_attribution.csv`
      (row per subcatchment: id, outlet node, incremental area, n_cells, outlet
      skill, `e_local` stats, specific injection, advected RMS, aggregated
      attributes); plots: subcatchment map coloured by local injection,
      advected-vs-local stacked bars along the main stem, attribute scatter panel.
- **Note:** partly unblocks §2b (`wflow_gauges*`/`wflow_subcatch*` is a ready-made
  gauge/subcatchment definition for per-gauge reporting).
- Touch: new `scripts/subcatch_error_attribution.jl` (+ small reusable helpers if
  `resolve_staticmaps`/`load_regrid_nc` are promoted out of
  `scripts/spatial_eval.jl` into `src/`); no source-model change for the primary
  method.

---

# Computational performance

Distilled from the 02-09-2026 benchmark log in
[PERFORMANCE.md](PERFORMANCE.md) (STATUS AS OF 02-09-2026) and its persisted
metrics
([experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml](../experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml)).
Each item should land behind a toggle / be A/B-tested with the existing
`scripts/typecheck_forward.jl`, `scripts/diag_overhead.jl`,
`scripts/benchmark_inference_vs_train.jl`, `scripts/benchmark_rollout.jl`.

## 4. Fix type instability in the hot forward path (highest-confidence win)

- **Evidence:** `fieldtype(SparseConv, :A)` and
  `fieldtype(MassBalanceLayer, :A_routing)` are `AbstractMatrix{Float32}`
  (abstract) though the stored value is concrete `SparseMatrixCSC`.
  `@code_warntype` shows `Body::ANY` with red
  `getproperty(l, :A)::ABSTRACTMATRIX{FLOAT32}` propagating `::ANY` through
  `neigh`/`net_flux`/`q_new`/`h_new`/`Δ`. One `SparseConv` forward measures
  **18 allocs / 331 KB / ~289 µs** (323-node CPU graph). Paid every layer,
  every step, in both rollout and training.
- [ ] Parameterise the adjacency field on `SparseConv`:
      `struct SparseConv{M,V,F,SA<:AbstractMatrix{Float32},SB} … A::SA;
      A_batched::SB end` (SB covers `Nothing`/concrete). Update the
      `Flux.gpu`/`Flux.cpu` overloads and `precompute_batched` to preserve the
      concrete types.
- [ ] Same treatment for `MassBalanceLayer.A_routing` / `A_routing_batched`
      (currently `AbstractMatrix{Float32}` / `Union{Nothing,AbstractMatrix}`).
- [ ] Verify the fix: `@code_warntype` red→blue (no `::ANY` on the layer and
      top-level `WflowGNN` forwards) and BenchmarkTools allocs drop, via
      `scripts/typecheck_forward.jl` before/after.
- [ ] Re-run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` to quantify the wall-clock gain
      (the cuSPARSE/BLAS SpMM may still dominate; record the delta regardless).
- Touch: `src/gnn.jl` (`SparseConv`, `MassBalanceLayer` structs + `gpu`/`cpu`/
  `precompute_batched`).

## 5. Gate per-batch training diagnostics to once-per-epoch

- **Evidence:** `training_diagnostic_overhead` — the logging-only forwards
  (`one_step_loss` unconditionally + `loss_components` + `mb_amplification`
  when a mass balance is present) cost **62.4 ms CPU / 25.6 ms GPU** vs a real
  train step of **61.9 / 15.5 ms**, i.e. **50.2% (CPU) / 62.2% (GPU)** of every
  batch at horizon 1. Fraction shrinks at deeper BPTT horizons but dominates the
  many horizon-1 epochs.
- [ ] In the `train_model!` batch loop, stop calling `one_step_loss`,
      `loss_components`, `mb_amplification` every batch; compute them **once per
      epoch on a single fixed batch** (mirror the pattern already documented for
      `peak_epoch_diagnostics`). Keep the per-epoch history fields populated.
- [ ] Put it behind a `TrainSettings` toggle (e.g. `diagnostics::Bool = true`)
      so the per-batch behaviour can be restored for debugging and A/B'd.
- [ ] Confirm ~1.3–2× shallow-horizon training speedup with
      `scripts/diag_overhead.jl` (and an end-to-end `@elapsed` on one epoch).
- Touch: `src/training.jl` (batch loop + `TrainSettings`), config in
  `src/run.jl`.

## 6. Re-baseline on the production model / refresh smoke model

- **Evidence:** the 02-09 numbers are a 323-node / 8-layer proxy on a 4 GB
  laptop GPU; `sava_v081` (8235 nodes) does not fit locally, and
  `test_sava_small` fails to load (4-field `WflowGNN` state predates the
  `augment_mb` field, `5b75540`).
- [ ] Run `scripts/benchmark_rollout.jl` +
      `scripts/benchmark_inference_vs_train.jl` on `sava_v081` (8235 nodes) on a
      ≥16 GB GPU to get the real surrogate-vs-Wflow inference number; persist to
      that run's `metrics/performance.toml` and add a STATUS entry to
      PERFORMANCE.md.
- [ ] Re-train / re-save `test_sava_small` with the current `WflowGNN` struct so
      a lightweight current-struct smoke model exists for future benchmarks.
- Touch: `scripts/*`, `docs/PERFORMANCE.md` (no source change).

## 7. (Lower priority) Fuse MassBalanceLayer broadcasts

- **Evidence (partly speculative):** GPU is launch-latency bound at ~323 nodes
  (single forward 1.71 ms → 0.25 ms/member under B=16). The MB layer issues many
  small separate-kernel broadcasts (`q_phys_new`, `upstream_q`, `net_flux`,
  `h_phys_new`, re-normalise).
- [ ] Fuse the elementwise broadcasts in the `MassBalanceLayer` forward into
      fewer kernels; keep the θ=1 fast path bit-identical.
- [ ] Quantify launch-count/time change with `CUDA.@profile` before committing.
- Touch: `src/gnn.jl` (`MassBalanceLayer` forward).