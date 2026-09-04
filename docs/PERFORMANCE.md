This document serves as an archive of benchmarks, listing the computational performance of the mission critical aspects of the project.

Full methodology and tables in
[notes/timing_findings.md](notes/timing_findings.md),
[notes/message_passing_notes.md](notes/message_passing_notes.md),
[notes/training_tuning_notes.md](notes/training_tuning_notes.md).
Model unless noted: `experiments/test_sava_v081`, 8235 nodes, hidden 64,
3 SparseConv layers.

> **Measurement caveat:** GPU kernels launch asynchronously — always
> `CUDA.synchronize()` before reading the clock, warm up 1–3 calls, report the
> **median**. An inference step = one forward pass; a training step =
> forward + backward + optimiser. Never compare training per-batch time to a
> forward-only step. Verify from the printed table, not the shell exit code
> (benign CUDA 13.1-vs-13.2 stderr warning makes exit non-zero on success).

---

# STATUS AS OF 02-09-2026

## SUMMARY

**When / code state.** Benchmarks run 02-09-2026 against `HEAD = 0ad6586`
(*"uppdate agent prompts"*, 2026-09-01) **with uncommitted working-tree edits**
to `src/{run,training,strategy,postprocess,WflowRoutingGNN}.jl` (working tree
dirty; `docs/CHANGELOG.md` was reset to a one-line header, so there is no
per-commit changelog to attribute these edits to). Numbers reflect that dirty
tree, not a clean tagged commit.

**Device / specs.**

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12800HX, 31.5 GB RAM |
| GPU | NVIDIA RTX A1000 Laptop, **4 GB VRAM** (small) |
| Julia | 1.12.6 · CUDA functional = true |

**Model benchmarked.** `experiments/sava_small_v081_mb_prior_increment` —
**323 nodes**, hidden 64, **8** `SparseConv` layers, `mlp_layers=2`, hard mass
balance (`mb_augment_decoder=true`, `θ=1`). This run was chosen because it is
the smallest/most-recent trained model **compatible with the current struct**:
`test_sava_small` (the nominal smallest run) fails to load — its 4-field
`WflowGNN` state predates the `augment_mb` field added 2026-08-14 (`5b75540`),
so `loadmodel!` errors on a structure mismatch. The full `sava_v081` (8235
nodes) was not benchmarked here: it does not fit the 4 GB laptop GPU. **These
numbers are therefore not directly comparable to the archived 8235-node /
3-layer `test_sava_v081` figures below** (fewer/​smaller nodes, but more and
wider layers).

Persisted artefacts:
[experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml](../experiments/sava_small_v081_mb_prior_increment/metrics/performance.toml)
(`rollout_benchmark`, `inference_vs_train_benchmark`,
`training_diagnostic_overhead` tables).

### Inference / rollout (the surrogate-decision number)

Per-timestep, per-member wall time (median), 200-step window, B=16 ensemble:

| Device | Single rollout | Ensemble (B=16), per member | Per-member speedup |
|---|---|---|---|
| GPU | **1.70 ms/step** | **0.25 ms/step/member** | **6.78×** |
| CPU | 3.94 ms/step | 2.75 ms/step/member | 1.43× |

- A rollout step ≈ a bare forward pass on both devices (rollout/forward = 0.95×
  GPU, 0.98× CPU) — rollout adds no measurable overhead over the model call.
- GPU ensembling delivers a **6.78× per-member throughput gain** (vs. the ~2.5×
  archived on the 8235-node model): the single-member GPU step is launch-latency
  bound and batching 16 members fills the device far more effectively at
  323 nodes. GPU ensemble variance is also far tighter (std 0.20 vs 4.11 ms).
- CPU ensembling is a modest 1.43× (compute-bound, little launch latency to
  amortise).

### Training step

Median per-step (horizon = current_steps = 1, reps = 50):

| Device | forward | fwd+bwd | full train step | bwd/fwd | step/fwd |
|---|---|---|---|---|---|
| GPU | 1.71 ms | 11.99 ms | 13.32 ms | 6.99× | 7.77× |
| CPU | 4.37 ms | 14.56 ms | 15.17 ms | 3.33× | 3.47× |

- GPU fwd:bwd ratio (~7×) is higher than the textbook ~2.5–3× and than CPU
  (~3.3×). At 323 nodes the *forward* is almost pure launch latency (1.7 ms),
  while the backward pass launches many more (and larger) kernels through the
  Zygote tape, so the ratio inflates. Not a regression — an artefact of a tiny
  forward on GPU.
- Full-train-step distributions are heavy-tailed (GPU max 119 ms vs median
  13 ms; CPU max 538 ms): optimiser update, `isfinite`/grad-norm host syncs, and
  GC pauses. Report medians only.

## IMPROVEMENTS

- **GPU ensemble rollout throughput: 6.78× per member** (evidence:
  `rollout_benchmark.gpu_ensemble` 0.251 ms vs `gpu_single` 1.700 ms median).
  The block-diagonal single-SpMM ensemble path is doing its job and then some at
  this graph size.
- **Rollout ≈ forward (no rollout-loop overhead)** on both devices
  (`*_ratios.rollout_over_forward` ≈ 0.95–0.98).

## DEGRADED

- No prior structured status log with matching model/hardware exists, so no
  strict regression can be asserted. Relative to the archived 8235-node figures,
  absolute per-step GPU rollout is faster (1.70 vs ~2.0 ms) but on a
  much smaller graph — **not** a like-for-like comparison; treat as a new
  baseline, not a delta.

## HYPOTHESES  (evidence vs speculation)

1. **Type instability in the hot forward path — CONFIRMED (evidence).**
   `fieldtype(SparseConv, :A)` and `fieldtype(MassBalanceLayer, :A_routing)` are
   both `AbstractMatrix{Float32}` (abstract) even though the concrete stored
   value is `SparseMatrixCSC{Float32,Int64}`. `@code_warntype` on `SparseConv`,
   `MassBalanceLayer`, and the top-level `WflowGNN` forward shows `Body::ANY`
   with red `getproperty(l, :A)::ABSTRACTMATRIX{FLOAT32}` propagating `::ANY`
   through every downstream value (`neigh`, `net_flux`, `q_new`, `h_new`, `Δ`).
   Measured cost of one `SparseConv` forward (323-node CPU graph, BenchmarkTools):
   **18 allocations / 331 KB / ~289 µs** — a non-trivial, size-independent
   dispatch/boxing tax paid on *every* layer, *every* step, in both rollout and
   training. *Speculation:* parameterising the adjacency fields to their concrete
   type should remove the `::ANY` propagation and cut these allocations; needs an
   A/B to quantify the wall-clock gain (the SpMM itself may still dominate).

2. **Per-batch training diagnostics cost as much as the train step — CONFIRMED
   (evidence).** `training_diagnostic_overhead`: the logging-only forward passes
   run every batch (`one_step_loss` unconditionally, plus `loss_components` +
   `mb_amplification` when a mass balance is present) sum to **62.4 ms CPU /
   25.6 ms GPU**, versus a real train step of **61.9 ms CPU / 15.5 ms GPU** —
   i.e. **50.2% (CPU) / 62.2% (GPU)** of each batch is spent on epoch-granularity
   metrics. *Caveat / speculation:* measured at horizon = 1. At deeper curriculum
   horizons the real train step grows (BPTT over up to 10 steps) while these
   diagnostics stay ~1-step, so the fraction shrinks with horizon — but for the
   many horizon-1 epochs it is a ~2× training slowdown. Moving them to
   once-per-epoch on a single fixed batch (as `peak_epoch_diagnostics` already
   documents for itself) is the obvious fix.

3. **GPU is launch-latency bound at this scale (evidence + interpretation).**
   Single-member GPU forward 1.71 ms for a 323-node / 8-layer model, dropping to
   0.25 ms/member under B=16 batching, is the classic signature of kernel-launch
   overhead dominating arithmetic below a few thousand nodes (consistent with
   `docs/DECISIONS.md` message-passing notes). The many small per-step broadcasts
   in `MassBalanceLayer` (separate kernels for `q_phys_new`, `upstream_q`,
   `net_flux`, `h_phys_new`, re-normalise) compound this. *Speculation:* fusing
   those broadcasts would reduce launch count; unquantified.

## RECOMMENDATIONS

1. **Parameterise the adjacency fields** (`SparseConv{...,SA<:AbstractMatrix}`,
   same for `MassBalanceLayer.A_routing`/`A_routing_batched`) to restore type
   stability in the innermost SpMM. A/B with `@code_warntype` (red→blue) and
   BenchmarkTools allocations before/after. Cheapest, highest-confidence win.
2. **Gate the per-batch diagnostics** behind an epoch-level toggle (compute
   `one_step_loss`/`loss_components`/`mb_amplification` once per epoch on one
   fixed batch, not every batch). Expected ~1.3–2× training speedup at shallow
   curriculum horizons.
3. **Re-baseline on the production model.** These numbers are a 323-node /
   8-layer proxy on a 4 GB laptop GPU. Run `benchmark_rollout.jl` +
   `benchmark_inference_vs_train.jl` on `sava_v081` (8235 nodes) on a ≥16 GB GPU
   to get the real surrogate-vs-Wflow inference number, and re-train/refresh
   `test_sava_small` so a lightweight current-struct smoke model exists.
4. **(Lower priority) Fuse the `MassBalanceLayer` broadcasts** to cut GPU
   kernel-launch count; quantify with `CUDA.@profile` before committing.

---

# OLD STATUS AS OF 27-08-2026

*Everything below this header predates the introduction of these structured
status logs; it back-fills benchmarks recorded before 27-08-2026.*

---

## Inference / rollout (the number for the surrogate decision)

| Device | Single rollout | Ensemble (B=16), per member |
|---|---|---|
| GPU | **~2 ms/step** | **~1.2 ms/step/member** |
| CPU | ~25 ms/step | ~36 ms/step/member |

- **GPU ensembling ~2.53× per-member throughput**, far tighter variance
  (std 0.09 vs 10.7 ms): single-member GPU step is launch-latency bound; batching
  fills the GPU. CPU ensembling is a slight loss (0.69×).
- A rollout step ≈ one bare forward pass on both devices.

## Training step

| Device | forward | fwd+bwd | full train step |
|---|---|---|---|
| GPU | 2.57 ms | 6.63 ms | 8.53 ms |
| CPU | 34.9 ms | 99.8 ms | 102.8 ms |

- forward:(fwd+bwd) ratio ≈ 2.5–3× (textbook).
- Full train step adds side-forwards, grad-norm/`isfinite` host syncs, optimiser
  update, per-batch host→GPU transfer → ~18–27 ms/batch. Not inference.
- No unexplained speedup ever occurred — an earlier "6×" was a training-iteration
  time compared against a forward step.

## Message passing (single layer, N=743)

| Device | Fastest | Note |
|---|---|---|
| GPU | AdjMat **sparse** ≈ dense (~0.26–0.28 ms) | ~1.6× over GraphConv sparse; scatter (COO) ~0.43 ms |
| CPU | GraphConv sparse ≈ AdjMat sparse (~1.85–1.95 ms) | AdjMat dense ~2× slower |

CPU→GPU speedups ~4–13×; below a few thousand nodes launch/overhead dominates.

## GPU memory

- **OOM root cause & fix:** dense `∂A` (~17 GB at `(B·N)² = 65880²`) from
  Zygote's generic `*` rrule → custom `_topology_mul` rrule never materialises it.
- **Resident dataset ≈ 1.1 GB** for full `sava_v081` train+val (uploaded once).
  Windowing is free (overlapping windows share graph objects, deduped by
  `objectid`).
- Peak model: `peak ≈ (D_resident + M_model) + c₁·B + c₂·B·N·S`
  (`S` = unrolled steps / BPTT tape depth).