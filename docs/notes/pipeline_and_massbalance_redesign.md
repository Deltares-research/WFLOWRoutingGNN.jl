# Pipeline & mass-balance code redesign

A design note for restructuring the emulator into **four cleanly separable
stages** and **decoupling the mass-balance (MB) code from the generic GNN code**,
with multiple-dispatch MB formulations and a post-hoc MB analysis that works even
on models trained *without* an MB layer.

Status: IMPLEMENTED (shipped 2026-10-21 as TODO steps R1–R7; this note is kept
as the design record). See [DECISIONS.md](../DECISIONS.md) →
IMPLEMENTED 19-10-2026, [CHANGELOG.md](../CHANGELOG.md) 2026-10-21, and
[TODO.md](../TODO.md).

---

## 0. Goals & constraints (verbatim intent)

1. Split the workflow into 4 parts that **run independently** (given their input
   data exists): **(1) data preprocessing, (2) model training, (3) model
   inference, (4) model analysis**.
2. Running all 4 back-to-back must reproduce **all artifacts currently produced**
   by `run_wflow_gnn`.
3. **Separate the mass-balance layer code from the generic GNN code**, reflected
   in the code/file structure.
4. Make **toggling between MB formulations easy** (anticipating new formulations).
5. **MB analysis of model inference must be possible regardless of whether an MB
   layer was present** in the trained model.
6. Preserve the **computational-performance optimizations** already identified.
7. Stylistically prefer **multiple dispatch** over single functions with large
   `if/else` chains.

---

## 1. The four-stage pipeline

### 1.1 Current coupling (what blocks separation)

`run_wflow_gnn` ([src/run.jl](../../src/run.jl)) does everything in one function
and keeps the preprocessing products **in memory only**:

- `build_wflow_graph` → `graphs, norm_stats, grid, postscale, static_arr` — never
  serialized as a unit (only `norm_stats.toml` + the dataset `.jld2` splits land
  on disk, as *model* artifacts).
- training consumes those in-memory objects directly;
- **inference and analysis are fused** inside `evaluate_and_write`
  ([src/run.jl](../../src/run.jl) L422): a single loop rolls out *and*
  regrids *and* writes NetCDF *and* computes spatial metrics / downstream
  timeseries / MB diagnostics / volume budget.

So today you cannot re-run analysis without re-running inference, nor inference
without the in-memory preprocessing state.

### 1.2 Target: stages exchange **on-disk artifacts**

Each stage becomes a function with an explicit on-disk contract; a thin driver
runs any subset. The glue that makes this clean is that **stages 3→4 already have
a working post-hoc precedent**: [scripts/spatial_eval.jl](../../scripts/spatial_eval.jl)
reconstructs everything it needs from the written NetCDFs + `staticmaps.nc`.

```
                 ┌────────────────────────── stage artifacts on disk ──────────────────────────┐
 wflow outputs → │ (1) PREPROCESS → prep/   (2) TRAIN → model/   (3) INFER → output/  (4) ANALYZE → metrics/,plots/ │
 (staticmaps.nc, │   graphs/dataset           model.jld2           <split>_pred.nc      spatial_metrics.*            │
  output.nc)     │   norm_stats, postscale    settings .toml       <split>_true.nc      downstream_timeseries.*      │
                 │   static_arr, grid-map     losses.*             (gridded, wflow-   mb_diagnostics.*, movie …   │
                 │   (node↔cell lookup)                            faithful NetCDF)                                │
                 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Stage 1 — Preprocess (`preprocess_run`)
- **Input:** `DataSettings` + `ModelSettings` (for `include_log_upstream_area`,
  the σ-floors) + `staticmaps.nc`, `output.nc`.
- **Work:** `build_wflow_graph` + `make_horizon_dataset` + peak-node stats +
  normalized-tail diagnostics.
- **Output (new `prep/` dir):** the dataset splits (already `.jld2`), `grid.jld2`
  (already), **plus** a serialized `preprocessed.jld2` / TOMLs holding
  `norm_stats`, `postscale`, `static_arr`, `dt`, graph topology (edge index,
  `A_routing`), and the schema id. This is the missing artifact that currently
  only lives in memory.
- **The node↔cell mapping lives here (not in inference output).** The `grid`
  lookup (node index → raster `(row,col)`) is a pure function of
  `staticmaps`/LDD and the preprocessing node ordering (`ldd_to_graph`), so it is
  a *preprocessing* artifact. Stage 4 uses it to gather node arrays back from the
  gridded NetCDFs (§1.3). Because the mapping and the node order come from the
  same stage, there is no ordering-drift risk.
- **Guarantee:** deterministic given settings (+ seed); no model, no device.

#### Stage 2 — Train (`train_run`)
- **Input:** `prep/` + `ModelSettings` + `TrainSettings`.
- **Work:** `build_gnn_model` (MB layer built here from `prep/` stats) +
  `train_model!` + loss/amplification/fixed-horizon plots.
- **Output (`model/`):** `model.jld2`, the three settings TOMLs, `norm_stats.toml`,
  loss curves/CSVs. (Unchanged set.)
- **Guarantee:** reads only `prep/`; never re-reads raw wflow outputs.

#### Stage 3 — Infer (`infer_run`)
- **Input:** `model/` + `prep/` (for `static_arr`, `norm_stats`, `postscale`,
  `grid`) + split selector + optional `val_daterange`.
- **Work:** `evaluate_trajectory` rollout per split → `regrid` → write NetCDF.
  **Only rollout + regrid + write** — no metrics.
- **Output (`output/`):** `<split>_pred.nc`, `<split>_true.nc` — **gridded
  `(lon,lat,time)` NetCDF matching wflow's `output.nc` layout as closely as
  possible** (same cell grid, variable names, time axis). This is the single
  inference artifact; no raw node-array dump here (the node↔cell mapping needed
  to recover node order is a preprocessing artifact, §1.1/§1.3).
- **Guarantee:** pure forward model; reproducible; the *only* stage that needs the
  trained weights + device. Output is a faithful stand-in for a wflow run so
  downstream tooling can treat predicted and reference NetCDFs identically.

#### Stage 4 — Analyze (`analyze_run`)
- **Input:** `output/` NetCDFs + `staticmaps.nc` + `model/` settings + `prep/`
  (postscale/grid) — **no model forward pass** for the standard diagnostics.
- **Work:** spatial-error metrics + summary, downstream timeseries (+ network
  inset), overprediction-vs-ramp, `river_q_performance_metrics`, validation
  movie, **MB residual diagnostics (§4)**, volume budget, the per-run
  `metrics.toml`.
- **Output (`metrics/`, `plots/`):** the current analysis artifact set.
- **Guarantee:** runs on any already-trained model, exactly like `spatial_eval.jl`
  today. MB diagnostics computed **post-hoc from outputs**, not from a layer.

### 1.3 Back-to-back equivalence

A driver `run_wflow_gnn(ds, ms, ts)` becomes
`analyze_run(infer_run(train_run(preprocess_run(...))))` (threading the `run_dir`).
Equivalence to today holds because:
- the artifact set is a **superset** of today's (adds `prep/preprocessed.jld2`);
- `regrid` is a lossless node→cell scatter, so node-level metrics
  (`river_q_performance_metrics`, MB residual) are recovered from
  `<split>_pred.nc` by **gathering river cells back to node order using the
  preprocessing-owned node↔cell mapping** (`grid` in `prep/`). The mapping and
  the node ordering originate in the same stage (`ldd_to_graph`), so the inverse
  gather is exact and ordering-stable — no per-split node dump is needed.
- stages read the **same settings** via the existing `settings_from_config`
  single-source-of-truth, so no behavioral drift.

### 1.4 Entry points & dispatch

- One driver module (e.g. `src/pipeline.jl`) exporting `preprocess_run`,
  `train_run`, `infer_run`, `analyze_run`, and the composed `run_wflow_gnn`.
- CLI: a single `scripts/pipeline.jl <stage> <config.toml>` with
  `stage ∈ {prep, train, infer, analyze, all}`. Prefer dispatch over an
  `if stage == …` ladder:
  ```julia
  abstract type Stage end
  struct Preprocess <: Stage end; struct Train <: Stage end
  struct Infer <: Stage end;      struct Analyze <: Stage end
  run_stage(::Preprocess, cfg) = preprocess_run(cfg)
  # …one method per stage; `all` folds over (Preprocess(),Train(),Infer(),Analyze())
  ```
- `hpar_search` / checkpoint-full-eval call `infer_run`+`analyze_run` on each
  checkpoint dir instead of the fused `evaluate_and_write`.

---

## 2. Separating mass-balance from generic GNN code

### 2.1 File structure

Move all MB code out of [src/gnn.jl](../../src/gnn.jl) into a new
**`src/massbalance.jl`** (included *before* `gnn.jl`):

- `MassBalanceLayer` struct + `Flux.@layer`/`@functor`/`gpu`/`cpu` overloads;
- the forward operator, `mb_decoder_features`, `mb_diagnostics`,
  `_floor_q_norm_nonnegative`, `_nonnegative_floor`, `precompute_batched(::MB…)`;
- the MB-construction block currently inlined in `build_gnn_model`
  ([src/run.jl](../../src/run.jl) L190–250) → an MB **factory**
  `build_massbalance(formulation, ms, prep)`.

`gnn.jl` keeps only the generic `SparseConv`, `WflowGNN`, encoder/processor/
decoder plumbing, and the `_topology_mul` rrule (shared — it lives at the top of
gnn.jl and is used by both; keep it in a tiny `src/sparse_ops.jl` included first
so both `massbalance.jl` and `gnn.jl` can use it without ordering hacks).

### 2.2 Formulations as a dispatch hierarchy (replaces `nothing` + if-else)

Introduce an abstract type and make "no mass balance" a **first-class null
object** instead of `nothing`, so the forward pass dispatches instead of branching:

```julia
abstract type MassBalanceFormulation end

struct NoMassBalance <: MassBalanceFormulation end          # replaces `nothing`

struct ThetaKinematicWave{V<:AbstractVector{Float32}} <: MassBalanceFormulation
    # exactly the current MassBalanceLayer fields (postscale_q/h, μ/σ vectors,
    # dt, A_routing[_batched], batch_size, θ, smooth_h_floor, h_floor_softness)
end
# future: struct DiffusiveWave …, struct LocalInertial … <: MassBalanceFormulation
```

Dispatch points that today are `if/else` on `isnothing(m.mass_balance)` or on
`l.θ == 1f0` become methods:

| concern | today (branch) | redesign (dispatch) |
|---|---|---|
| decoder output width | `WflowGNN(...,1)` vs `out_dim` | `decoder_outputs(::NoMassBalance,ms)` / `(::ThetaKinematicWave,ms)` |
| forward state update | `if isnothing(m.mass_balance)` in `(m::WflowGNN)(…)` | `apply_formulation(f, g, state, forcing, forcing_next, Δ)` |
| implicit vs mixed θ | `if l.θ == 1f0` inside the layer | keep as a cheap inner branch *or* `NetFlux{:implicit}`/`{:mixed}` trait; θ is a scalar so a branch is fine — **don't over-dispatch** |
| decoder augmentation | `if m.augment_mb` | `extra_decoder_features(f, …)` returning `()` for `NoMassBalance` |
| per-step q floor | `_floor_q_norm_nonnegative` only in MB path | `floor_state(f, g, raw)` (identity for `NoMassBalance`) |

`WflowGNN.mass_balance :: MassBalanceFormulation` (never `nothing`); the forward
pass collapses to:

```julia
h = m.processor(g, m.encoder(x))
Δ = m.decoder(vcat(h, extra_decoder_features(m.mass_balance, g, state, forcing)...))
return apply_formulation(m.mass_balance, g, state, forcing, forcing_next, Δ)
```

where `apply_formulation(::NoMassBalance, …) = state .+ Δ` and the
`ThetaKinematicWave` method does floor-q + analytic h. This removes **every**
`isnothing(mass_balance)` branch across `gnn.jl`, `run.jl`, `rollout.jl`,
`training.jl`.

### 2.3 Guidance for new formulations (the "easy toggle" test)

Adding a formulation = (a) one `struct … <: MassBalanceFormulation`, (b) methods
for `decoder_outputs`, `extra_decoder_features`, `apply_formulation`,
`floor_state`, `build_<name>` factory, (c) a `ModelSettings` enum value
(`mb_formulation = "theta_kinematic" | "none" | …`) mapped in
`settings_from_config`. No edits to the generic GNN forward pass.

---

## 3. Post-hoc MB analysis independent of the MB layer

**Problem:** `rollout_mb_diagnostics(cpu_model, …)` ([src/run.jl](../../src/run.jl)
L501) calls `model.mass_balance`, so a model trained with `enforce_mass_balance
= false` (or a future non-MB formulation) cannot be mass-balance-analyzed.

**Design:** split MB diagnostics into two layers.

- **`mb_residual_diagnostics`** (NEW, post-hoc, layer-free): given
  `<split>_pred.nc` (river_q, river_h grids), `staticmaps.nc` (`wflow_ldd`,
  `wflow_uparea`→postscale_q, river width·length→postscale_h, `dt`), and the
  split times, recompute the kinematic-wave residual
  `r(t)=h_phys[t+1]−h_phys[t]−dt/(w·l)·(ΣQ_up+Q_iw−Q_out)` **directly from the
  predicted fields**. Reuses `ldd_to_graph` + the same A·q routing used by the
  layer, but takes *predicted q/h* as inputs rather than model internals. Works
  for **any** model (MB or not) — the residual simply measures how
  mass-conservative the predictions are. This is the mass-balance analog of the
  subcatchment post-hoc design.
- **`mb_layer_diagnostics`** (the current `rollout_mb_diagnostics`): kept, but
  invoked only when the formulation is an MB type (dispatch on
  `m.mass_balance::MassBalanceFormulation`; a no-op method for `NoMassBalance`).
  Gives the exact internal intermediates (net_flux, h_raw-before-floor, floor
  fraction) that only exist when a layer is present.

Stage 4 always runs `mb_residual_diagnostics` (post-hoc, universal) and
*additionally* runs `mb_layer_diagnostics` when the model carried an MB layer.
The per-run `metrics.toml` MB fields degrade gracefully: residual-based for all
models, layer-internal (floor fraction, stiffness) only for MB models.

---

## 4. Scripts consolidation (fold `scripts/` into the stages)

A full audit of `scripts/` (20 `.jl` files) shows most either belong *inside* the
four stages, are thin drivers that stay thin, or are benchmark/one-off tools. The
restructuring should **pull the reusable logic into `src/` and leave `scripts/`
as thin CLIs + benchmarks**. Classification and action:

### 4.1 Promote to `src/` (reusable logic currently inline in scripts)

- **Post-hoc analysis helpers** — `resolve_staticmaps`, `load_regrid_nc` (inline
  in [scripts/spatial_eval.jl](../../scripts/spatial_eval.jl)) are exactly what
  **stage 4** and the subcatchment-attribution note need. Promote into the
  analysis layer (`src/postprocess.jl` or a new `src/analysis.jl`) so stage 4,
  `spatial_eval`, and `subcatch_error_attribution` share one implementation.
- **Layer-free mass-balance routing/validation** —
  [scripts/check_mass_balance.jl](../../scripts/check_mass_balance.jl)
  (`compute_upstream_q`, the three-scheme discretisation comparison) and
  [scripts/check_mass_balance_details.jl](../../scripts/check_mass_balance_details.jl)
  (`upstream_sum`, the 9 consistency checks) already compute the kinematic-wave
  residual **from data, with no model layer** — i.e. they *are* the prototype for
  the §3 `mb_residual_diagnostics`. Promote this routing/residual core into
  `src/massbalance.jl` (or `src/analysis.jl`) and have both the scripts and stage
  4 call it. This is the single most valuable promotion: it directly delivers the
  "MB analysis independent of the MB layer" requirement.
- **LR / batch-size tuning machinery** — [scripts/lr_range_test.jl](../../scripts/lr_range_test.jl)
  and [scripts/autotune_train.jl](../../scripts/autotune_train.jl) define a large,
  genuinely reusable toolkit (`lr_range_test`, `gradient_growth_probe`,
  `recommend_lr`, `aggregate_lr_recommendations`, `estimate_batch_size`,
  `autotune`). Promote into a new `src/tune.jl` with a public API; the scripts
  become thin CLIs over it. (Fits the existing `src/hparsearch.jl` neighbourhood.)

### 4.2 Keep as thin CLI drivers (already correct — just retarget)

- [scripts/train.jl](../../scripts/train.jl), [scripts/hparsearch.jl](../../scripts/hparsearch.jl),
  [scripts/spatial_eval.jl](../../scripts/spatial_eval.jl),
  [scripts/regen_mb_diagnostics.jl](../../scripts/regen_mb_diagnostics.jl) already
  just orchestrate `src/` functions. After the split they should call the new
  **stage** entry points (`train_run`, `analyze_run`, …). `regen_mb_diagnostics`
  and `spatial_eval` effectively *become* "run stage 4 only" — i.e. they
  collapse into `scripts/pipeline.jl analyze`.

### 4.3 Keep as benchmarks / profilers (do not promote; use as the perf gate)

- [scripts/benchmark_message_passing.jl](../../scripts/benchmark_message_passing.jl),
  [scripts/benchmark_rollout.jl](../../scripts/benchmark_rollout.jl),
  [scripts/benchmark_inference_vs_train.jl](../../scripts/benchmark_inference_vs_train.jl),
  [scripts/diag_overhead.jl](../../scripts/diag_overhead.jl),
  [scripts/profile_rollout.jl](../../scripts/profile_rollout.jl),
  [scripts/typecheck_forward.jl](../../scripts/typecheck_forward.jl). These are the
  **acceptance gate** for §5 — run `benchmark_message_passing` + `benchmark_rollout`
  before/after the refactor. They stay in `scripts/` but must be updated to the new
  module/stage API so they keep compiling.

### 4.4 One-off inspection / archive

- [scripts/inspect_scaling.jl](../../scripts/inspect_scaling.jl),
  [scripts/network_diameter.jl](../../scripts/network_diameter.jl),
  [scripts/plot_ldd.jl](../../scripts/plot_ldd.jl),
  [scripts/rating_curve_test.jl](../../scripts/rating_curve_test.jl),
  [scripts/regime_analysis.jl](../../scripts/regime_analysis.jl),
  [scripts/smoke_test_sparse.jl](../../scripts/smoke_test_sparse.jl). Mostly
  procedural one-offs. **Exceptions worth salvaging:** `plot_ldd.jl`'s
  network-drawing is already slated for reuse by the timeseries network inset
  (TODO §3b) — factor that out into `src/plot.jl`; and `smoke_test_sparse.jl`
  should migrate into the real test suite (`test/test_gnn.jl`) as it exercises the
  sparse batching paths the perf optimizations depend on. The rest can stay as
  scripts or be archived under `scripts/oneoff/`.

### 4.5 Net effect

`scripts/` shrinks to: `pipeline.jl` (the staged CLI), `tune.jl` driver,
the benchmark/profiler harnesses, and a small `oneoff/` archive. All genuinely
reusable analysis/diagnostic/tuning logic lives in `src/` behind a tested API,
and the post-hoc analysis + layer-free MB residual are shared by scripts and the
stage-4 pipeline alike.

---

## 5. Performance optimizations to preserve (non-negotiable)

The redesign is a **code-structure** change; these hot-path properties must be
byte-for-byte preserved (regression-test them):

- `_topology_mul` + its `rrule` (NoTangent for A) — the single most important
  autodiff-memory optimization; move to `src/sparse_ops.jl`, do not alter.
- `SparseConv` three-path dispatch (single / precomputed block-diagonal / reshape
  fallback) and `precompute_batched` for both `SparseConv` and the MB layer.
- CuSPARSE CSR (`Int32` colptr) device conversions in `gpu`/`cpu` overloads.
- Zero-copy `reshape` of per-node constants when `n_rep == 1` (the rollout path).
- Explicit `Flux.gpu`/`Flux.cpu` overloads (Functors restricted to trainable
  fields) — must follow the layer when it moves files.
- Build block-diagonal on CPU before `Flux.gpu` (ordering invariant).

Keep a `scripts/benchmark_message_passing.jl` / `benchmark_rollout.jl` run
before/after as the acceptance gate.

---

## 6. Migration sequencing (low-risk order)

1. **Extract** `src/sparse_ops.jl` (`_topology_mul` + rrule) and
   `src/massbalance.jl` (move MB code verbatim) — pure code move, no behavior
   change; tests green.
2. **Introduce** `MassBalanceFormulation` hierarchy with `NoMassBalance` +
   `ThetaKinematicWave` wrapping the existing fields; convert the `nothing`
   sentinel and the forward if-else to dispatch; tests green (bit-identical
   outputs).
3. **Serialize** the preprocessing bundle (`prep/preprocessed.jld2`) and split
   `run_wflow_gnn` into `preprocess_run`/`train_run`; compose for equivalence.
4. **Split** `evaluate_and_write` into `infer_run` (rollout+regrid+write) and
   `analyze_run` (metrics/plots); verify artifact-set equality against a golden
   run.
5. **Add** `mb_residual_diagnostics` (post-hoc, layer-free) and route stage-4 MB
   analysis through it; dispatch the layer-internal diagnostics on the
   formulation type.
6. **Consolidate `scripts/`** (§4): promote `resolve_staticmaps`/`load_regrid_nc`,
   the layer-free MB routing/residual, and the LR/batch tuning toolkit into `src/`;
   retarget thin drivers + benchmarks to the new stage/module API; migrate
   `smoke_test_sparse.jl` into the test suite and factor `plot_ldd.jl`'s network
   drawing into `src/plot.jl`.
7. **CLI** `scripts/pipeline.jl <stage>` + `Stage` dispatch; point `hpar_search`
   and checkpoint eval at the new stage functions.

Each step is independently testable and reversible; equivalence is checked by
diffing the artifact set of a small golden run (`test_sava_small_v081`).

---

## 7. Risks & open questions

- **Artifact contract churn.** Serializing `norm_stats`/`postscale`/`static_arr`
  adds a `prep/` schema that must version alongside `wflow_schema`. Mitigate with
  a small `PrepBundle` struct + version tag.
- **Node-array recovery (resolved).** Stage 4 recovers node arrays by gathering
  river cells from the gridded NetCDFs via the **preprocessing-owned node↔cell
  mapping** — inference stays purely gridded/wflow-faithful, and the mapping (not
  a per-split node dump) is the extra artifact, produced once in preprocessing.
  Implementation detail: confirm the gather handles inactive/NaN-fill cells and
  reproduces `ldd_to_graph`'s column-major node order exactly.
- **θ branch granularity.** `θ==1` fast path: keep as an inner scalar branch
  (cheap, not a dispatch concern) vs. a `NetFlux` trait — proposal says keep the
  branch to avoid over-dispatching. Confirm.
- **How much of stage 4 may forward the model?** The standard diagnostics are
  forward-free, but an optional Tier-2 counterfactual (cf. subcatchment note)
  would re-infer — keep that out of the default analyze path.
- **`run_dir` layout change** (`prep/ model/ output/ metrics/ plots/`) touches
  downstream tooling (evaluation/performance agents, plotting scripts). Audit
  consumers before renaming; may keep current flat layout and only *add* `prep/`.
- **Formulation ↔ preprocessing coupling.** Some future formulations may need
  different per-node constants (e.g. diffusive-wave needs slope). The `PrepBundle`
  should be a superset or the factory should pull extra statics on demand.
