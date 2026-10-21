# Subcatchment Error Attribution — Which Subcatchments Inject Downstream Error?

A design note for a **post-hoc** diagnostic that tests the hypothesis:

> *Some subcatchments contribute disproportionately to downstream (outlet) error,
> and that contribution correlates with WFLOW subcatchment attributes (slope,
> stream order, soil, reservoir presence, …).*

Like every other analysis in this repo (cf.
[scripts/spatial_eval.jl](../../scripts/spatial_eval.jl),
[scripts/regime_analysis.jl](../../scripts/regime_analysis.jl)), it runs on an
**already-trained** model from the artifacts `run_wflow_gnn` already writes — no
retraining, no model change required.

---

## 0. Motivation

The persistent spatial-dependence pattern (downstream cells worse; outlet-
dominated rollout error; `amp≈7`/`mb_gain≈22` physical flux accumulation) is
currently only known *in aggregate*. We do not know **where** the error is
*injected* versus merely **advected**. Because discharge accumulates downstream,
a naive per-outlet NSE conflates the two: an outlet inherits every upstream
subcatchment's error. To test "which subcatchments contribute", we need the
**incremental (local) error** each subcatchment adds on top of what it receives.

This also partly unblocks the open TODO §2b "no gauge concept exists" item:
WFLOW's own `wflow_gauges*` / `wflow_subcatch*` rasters give a natural
gauge/subcatchment definition and per-subcatchment outlets.

---

## 1. Everything needed is already on disk (post-hoc inputs)

| input | source | role |
|---|---|---|
| gridded `river_q`/`river_h` pred & truth `(lon,lat,time)` | `output/<split>_pred.nc`, `output/<split>_true.nc` | per-cell timeseries |
| subcatchment ID raster | `staticmaps.nc` → `wflow_subcatch` (and `…_grdc`, `…_calib-locs`, `…_grdc-subset`) | subcatchment membership |
| outlet / gauge cells | `wflow_gauges*`, `idx_out` / `x_out` / `y_out` | subcatchment outlet node |
| drainage network | `wflow_ldd` via [`ldd_to_graph`](../../src/preprocess.jl) | build the subcatchment tree + confluences |
| size / order | `wflow_uparea`, `wflow_streamorder` | "sufficiently large" filter + attribute |
| channel attrs | `RiverSlope`, `wflow_riverwidth`, `wflow_riverlength`, `RiverDepth`, `N_River` | attributes |
| hydraulic nonlinearity | `qbankfull`, `bankfull_volume`, `floodplain_volume` | over-bank regime attribute |
| land/soil attrs | `KsatVer`, `SoilThickness`, `M`, `f`, `thetaS`, `PathFrac`, `WaterFrac`, `Slope`, `wflow_landuse` | attributes |
| specials | `ResTrapEff` (reservoir), `wflow_glacierfrac` (glacier), `calibration_mask` | binary attributes |

Everything is grid-based `(lon,lat)` and aligns cell-for-cell with the pred/truth
grids — identical to how `spatial_eval.jl` already operates (reuse
`resolve_staticmaps`, `load_regrid_nc`).

---

## 2. Method

### 2.1 Subcatchment definition (two routes)

- **(A) WFLOW-native (preferred when gauge set is dense enough).** Use
  `wflow_subcatch` (or a chosen gauge-set variant) directly: each river cell is
  labelled by the subcatchment it drains to; the outlet is the subcatchment's
  gauge cell (`wflow_gauges` / `idx_out`). Attributes then map one-to-one onto
  WFLOW's own partition — the cleanest match to "subcatchment attributes found in
  the WFLOW model itself".
- **(B) Network-derived (when the native gauge set is too sparse / uneven).**
  Delineate **incremental catchments** from the LDD tree: split at confluences
  where *both* merging branches exceed an upstream-area threshold. Gives direct
  control over subcatchment size and a complete tessellation independent of gauge
  placement.

**"Sufficiently large size" filter.** Keep only subcatchments whose incremental
upstream area (or median outlet flow) exceeds a **configurable threshold**. This
(a) makes NSE/KGE meaningful and (b) avoids the intermittent-flow / dry-headwater
regime where the `max(0,·)` depth floor dominates (see
[mass_balance_stability_notes.md](mass_balance_stability_notes.md) and the E11
floor diagnostics) and skill metrics are degenerate.

### 2.2 Per-outlet skill (the inherited-plus-local baseline)

For each retained outlet node `o`, extract pred/true `q` (and `h`) timeseries and
compute standard skill (NSE, KGE + r/α/β, PBIAS, peak ratio, RMSE) — reuse
`river_q_performance_metrics` ([src/postprocess.jl](../../src/postprocess.jl)).
This is the *total* error at `o` (local + advected) and is **not** yet an
attribution.

### 2.3 Incremental (contributed) error — the crux

Build the **subcatchment tree** (which subcatchment drains into which) from the
LDD. For subcatchment `S` with outlet `o` and immediate upstream sub-outlets
`{u_1,…,u_k}`, define the **local injected discharge error**

$$e_{\text{local}}(S,t) \;=\; \big[Q^{\text{pred}}_o(t) - Q^{\text{true}}_o(t)\big]
\;-\; \sum_{i} \big[Q^{\text{pred}}_{u_i}(t) - Q^{\text{true}}_{u_i}(t)\big].$$

This is the error in the **incremental mass balance** between `S`'s inlets and its
outlet — how much error `S` *adds* on top of what it received. Summaries over `t`:

- `rms(e_local)` / `mean|e_local|` — magnitude of local error injection;
- signed `mean(e_local)` — systematic local over/under-contribution;
- **specific injection** `rms(e_local) / Σ_incremental inwater` (or per
  incremental area) — size-normalised so large and small subcatchments are
  comparable.

**Advected-vs-local decomposition.** At each outlet compare
`rms(Σ_i upstream error)` (advected) against `rms(e_local)` (injected). This
directly shows whether an outlet's error is mostly *inherited* or mostly *local*
— the heart of the hypothesis.

> **Tessellation requirement.** The difference form is exact only if the
> subcatchment partition is a proper tessellation: every cell belongs to exactly
> one subcatchment and *all* non-local inflow to `S` passes through tracked
> sub-outlets `u_i` (the only untracked inflow is `S`'s own incremental lateral
> inflow). Verify completeness before trusting `e_local`.

**Tier-2 counterfactual (optional, needs re-inference, still no retraining).**
Re-run the trained model substituting *true* upstream inflow at `S`'s boundary and
measure the outlet error — isolates local skill from inherited error exactly. This
is an inference-time intervention (like a boundary-forced rollout), more invasive
than the difference form but still post-hoc on the fixed weights.

### 2.4 Attribute aggregation (per subcatchment)

Aggregate each WFLOW attribute over `S`'s **incremental** cells (or its river
cells), choosing the reduction that matches the attribute:

- **outlet-cell value:** `wflow_uparea`, `wflow_streamorder`, `RiverSlope`@outlet;
- **incremental mean/median:** `KsatVer`, `SoilThickness`, `M`, `f`, `PathFrac`,
  `WaterFrac`, `Slope`, `N_River`, `RiverSlope`;
- **presence (binary):** reservoir (`ResTrapEff > 0`), glacier
  (`wflow_glacierfrac > 0`);
- **nonlinearity regime:** fraction of time the simulated `Q` exceeds `qbankfull`
  (over-bank), and `bankfull_volume`.

### 2.5 Attribute regression / ranking

Across the retained subcatchments, **Spearman** rank-correlate each attribute
against the local-error metric (`rms(e_local)` and specific injection); render a
scatter per attribute with ρ; rank attributes by |ρ|. Because the analysis is
exploratory and the number of large subcatchments on a small basin is modest,
report ρ with its CI/p and treat multiple comparisons with caution — the test is
**far better-powered on the full `sava_v081` basin** (8235 nodes, 585-hop
diameter, many subcatchments) than on `sava_small`.

---

## 3. Outputs (mirror `spatial_eval.jl` conventions)

- `metrics/<split>_subcatch_error_attribution.csv` — one row per subcatchment:
  id, outlet node, incremental area, n_cells, outlet {NSE, KGE, PBIAS,
  peak_ratio, RMSE}, `rms(e_local)`, signed `mean(e_local)`, specific injection,
  advected RMS, + all aggregated attributes.
- Plots: (a) subcatchment map coloured by local error injection; (b)
  advected-vs-local stacked bars along the main stem (outlet → headwater); (c)
  attribute scatter panel vs local error with Spearman ρ.

---

## 4. Post-hoc guarantee & reuse

Nothing here forwards the model for the **primary** (difference-of-errors)
method: it consumes only the written pred/truth NCs + `staticmaps.nc` + settings,
exactly like `spatial_eval.jl`. Reuse `resolve_staticmaps`, `load_regrid_nc`,
`ldd_to_graph`, and `river_q_performance_metrics`. Only the optional Tier-2
counterfactual re-runs inference (still fixed weights).

---

## 5. Connections & caveats

- **Tests the spatial-dependence pattern directly:** distinguishes "outlet error
  is uniform accumulation from everywhere" from "a few subcatchments inject most
  of it" — which changes the fix (global closure/scaling vs targeted).
- **Links to the floor/`amp` findings:** can check whether high-injection
  subcatchments coincide with floored headwaters, low-slope/backwater reaches, or
  reservoir-affected cells.
- **Unblocks part of TODO §2b:** `wflow_gauges*`/`wflow_subcatch*` is a ready-made
  gauge/subcatchment definition for per-gauge reporting.
- **Caveats:** reservoirs/lakes break simple routing (flag them); low-flow
  subcatchments give unreliable NSE (hence the size filter); multi-sink / clipped
  basins need an outlet-set convention consistent with the volume-budget outlet
  open question; the difference form depends on a verified tessellation.

---

## 6. As implemented (21-10-2026)

- **Outlet set is the only input.** Route A (gauge raster) and Route B (river
  sinks plus branch ends above confluences with upstream area >= threshold) both
  yield outlet cells. Every cell goes to its first downstream outlet, so the
  regions tessellate by construction; `tree.check` still verifies it and compares
  cumulative region area with `wflow_uparea` (median ~1 %, max ~5-8 % on both Sava
  models, the residual being cell-area vs the model's own area).
- **Same-time difference.** `e_local = dQ(o) - sum dQ(u_i)` ignores the routing
  lag between inlets and outlet, so a travelling error can show up as local. Read
  it as an attribution at the outlet's time step, not as an exact injection.
- **Ranking.** Spearman rho with a permutation p-value (5000 shuffles), Bonferroni
  over the attributes tested, `low_power` for n < 10, attributes with relative
  spread <= 1e-3 skipped (`SoilThickness` in `sava_small` varies only in the 4th
  digit and gave a spurious rho of 0.5).
- **First results (`sava_small_v081_e13b_depth_floored_hps01_s01`, Route B, 16
  retained regions).** Headwater regions are fully local (rms_adv = 0); the
  largest absolute local errors sit in the lower main stem (cumulative area
  > 2000 km2). The strongest rank correlations with el_local are river slope
  and  (rho ~ -0.7), but none survive Bonferroni at n = 16, so treat the
  rankings as hypotheses.
- **Not yet validated statistically on `sava_v081`**: the local full-Sava runs have
  diverged predictions, and the run now warns when max |Q_pred| exceeds 1e3 times
  max |Q_true|.
