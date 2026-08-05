# Training Performance, Memory & LR-Tuning Notes

A consolidated reference of the design discussions and decisions around GPU memory,
efficient message passing / backprop, batch-size selection, and the curriculum
learning-rate schedule for `WflowRoutingGNN`. Written to be consulted later; it
records both what we *implemented* and the alternatives we *considered but did not
pursue*.

Relevant code:
[src/gnn.jl](../src/gnn.jl), [src/strategy.jl](../src/strategy.jl),
[src/training.jl](../src/training.jl), [src/rollout.jl](../src/rollout.jl),
[src/run.jl](../src/run.jl), [src/preprocess.jl](../src/preprocess.jl),
[scripts/lr_range_test.jl](../scripts/lr_range_test.jl),
[scripts/autotune_train.jl](../scripts/autotune_train.jl).

---

## 1. The OOM that started it all

Symptom: with a modest batch size (8) and a short horizon (2 steps), the
`sava_v081` run hit `Out of GPU memory trying to allocate 16.168 GiB` even on an
80 GiB GPU.

**Root cause — dense gradient of a sparse topology matrix.** When batching,
the graph adjacency becomes a block-diagonal sparse matrix `A_batched` of shape
`(B·N) × (B·N)`. For `B=8`, `N=8235` that is `65880 × 65880`. Even though
`A_batched` is *not* trainable, Zygote's generic `rrule` for `*` computed
`∂L/∂A = ΔY · hᵀ` as a **dense** `65880² × 4 bytes ≈ 17.4 GB` outer product in the
backward pass.

**Fix implemented — a custom `rrule` that never materializes `∂A`.** We route
every sparse topology multiply through a thin helper `_topology_mul(A, B) = A * B`
with a `ChainRulesCore.rrule` that returns `NoTangent()` for `A` and computes only
the node-feature gradient `∂B = Aᵀ · ΔY`. All 6 SpMM sites (3 in `SparseConv`,
3 in `MassBalanceLayer`) go through it. `ChainRulesCore` was added to `[deps]`.

### Alternatives considered and rejected
- **`Flux.no_gradient(() -> A * h')`** — zeros *all* gradients in the block,
  including `∂h` for the trainable node features. Loses learning signal. ✗
- **`ChainRulesCore.ignore_derivatives(A) * h'`** — looks right, but the `rrule`
  for `*` still computes the 17 GB `ΔA` *before* `ignore_derivatives` discards it.
  The allocation still happens. ✗
- A custom `rrule` on `_topology_mul` is the *only* approach that skips computing
  `ΔA` entirely. ✓ (8 lines, localized to `gnn.jl`.)

Also relevant: `Functors.@functor` field restriction on `MassBalanceLayer` /
`SparseConv` keeps the sparse matrices off the Flux/Optimisers traversal path.

---

## 2. Where the GPU memory actually goes

Split peak GPU memory into two parts:

$$
\text{peak} \approx \underbrace{D_\text{resident} + M_\text{model}}_{\text{fixed}}
\;+\; \underbrace{c_1 B}_{A_\text{batched}}
\;+\; \underbrace{c_2\, B\, N\, S}_{\text{activations + BPTT tape}}
$$

where `B` = batch size, `N` = nodes per graph, `S` = `strategy.current_steps`
(number of unrolled rollout steps).

### Resident dataset (uploaded once, up front)
`dev_fn(train_loader)` in [src/training.jl](../src/training.jl) eagerly moves the
**whole** train+val set to GPU and keeps it resident. Measured for `sava_v081`:

| Data | Total |
|---|---|
| state (2×N Float32, per timestep) | ≈ 744 MB |
| forcing (1×N Float32, per timestep) | ≈ 372 MB |
| static (5×N Float32) | **shared → one copy ≈ 165 KB** |
| edges (topology) | deduped to ≈ 128 KB |

Total resident ≈ **1.1 GB** for the full train+val set — far below the 12 GB
`output.nc` file. (File size is *not* a valid bound: NetCDF is zlib-compressed on
disk, but on the other hand only active river nodes + the `state`/`forcing`
variables + the train/val split are moved.) The real figure is
`n_times × n_nodes × (n_state + n_forcing) × 4 B × (train_frac + val_frac)`.

**Windowing is free.** `windows = [graphs[t:t+nhorizon-1] …]` allocates only
`nwindows × nhorizon` *pointers*; every overlapping window references the same
per-timestep `GNNGraph` objects. Functors' `CachedWalk` (IdDict by `objectid`)
uploads each timestep's arrays exactly once regardless of overlap. So `nhorizon`
overlap costs **zero** extra GPU memory.

### The `static` refactor (correctness, not memory)
We removed `static` from every `GNNGraph.ndata` and now pass it as a single
N-wide matrix threaded through `loss_function`, `one_step_loss`,
`loss_components`, `rollout`, and `train_model!`. Net GPU memory change ≈ **zero**
(the old code already stored the *same* `static` object in every graph, so
CachedWalk deduplicated to one copy). The benefits are explicit data flow, a
cleaner Zygote path, and robustness against accidental N-fold copies.

> ⚠️ **Latent bug this exposed:** the model forward did `vcat(state, forcing,
> static)`, but after the refactor `static` is N-wide while a batched `state` is
> `B·N`-wide. Batched training (`batch > 1`) was silently broken at the `vcat` —
> all prior real runs happened to be `batch=1`. **Fixed** by tiling `static` to
> the node count inside the forward pass in [src/gnn.jl](../src/gnn.jl).

### Scaling summary
- **Linear in `B`** (activations, block-diagonal `A_batched` nnz, collated arrays).
- **Linear in `S`** (BPTT tape retains every unrolled step's activations),
  *not* in `nhorizon` directly — `nhorizon` only bounds the max `S`.
- `A_batched` is reused across steps → does **not** scale with `S`.

---

## 3. Reducing the activation / BPTT-tape term

Ranked levers for the `c₂·B·N·S` term:

### ✅ Implemented — gradient checkpointing (biggest win)
In [src/strategy.jl](../src/strategy.jl) the S-step rollout loop wraps the model
call in `Flux.Zygote.checkpointed(...)` when `nsteps > 1`. This tells Zygote to
discard each step's internal activations and **recompute** the forward during
backprop, collapsing tape memory:

$$
O(S \cdot B \cdot N \cdot \text{hidden}) \;\longrightarrow\;
O(B \cdot N \cdot \text{hidden}) + O(S \cdot B \cdot N)
$$

Only *one* step's internals are live at any instant. Cost ≈ +30–50 % wall-time
per step (one extra forward). Gradients are **exact**.

**Critical caveat honored:** the noise draw (`noise_scale > 0` → `randn`) is kept
*outside* the `checkpointed` boundary, so recomputation is deterministic and
gradients stay correct. Single-step phases (`nsteps == 1`) skip checkpointing —
no memory benefit, avoids the recompute cost.

*Reference:* Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016),
arXiv:1604.06174.

### ❌ Considered, not pursued — static-encoder factoring
Idea: since the encoder's first layer is affine and `static` is identical across
all `S` steps, precompute `s = W_static · static` once inside the differentiable
region and add it per step:
`h_t = W_sf · vcat(state_t, forcing_t) .+ s .+ b`.

- This is a valid *algebraic* refactor (it does **not** freeze `W_static`;
  gradients accumulate `∂L/∂W_static = Σ_t ∂L/∂h_t · staticᵀ`, identical to the
  original). An early framing that suggested detaching it under
  `ignore_derivatives` would have been *wrong* and was corrected.
- **Why we skipped it:** the dominant tape cost is the per-step *hidden*
  activations (`h_t` and the `nlayers` processor outputs), which genuinely change
  every step and are **not** removed by this factoring. It's mostly a *compute*
  win with only a minor memory trim — worthwhile only if `static` rows dominate
  the encoder input. Checkpointing is the real memory lever.

### ❌ Considered, not pursued — other levers
- **Gradient accumulation** (micro-batches → accumulate → one `update!`): reduces
  peak `B` without changing effective batch size. Safe fallback if still OOM.
- **Reshape path instead of `A_batched`** (reuse single `N×N` matrix, reshape
  `q` to `(N, B)`): shrinks the `c₁·B` term. It was added originally for ~2×
  speed, so this is a speed↔memory trade — measure only if `A_batched` is a
  meaningful fraction of the footprint.
- **Lazy per-batch GPU transfer** (keep loader on CPU, move each batch in the
  loop): trades PCIe transfer per epoch for a much smaller steady-state footprint.
  Not needed while ~4 GiB fits.
- **Mixed precision (Float16 activations):** halves both scaling terms but is
  risky — the `MassBalanceLayer` does physical-unit arithmetic where Float16
  dynamic range is dangerous. Last resort; would keep the MB layer in Float32.

---

## 4. Batch size ↔ learning rate coupling

Two independent effects — a **memory** reason (affects `B`) and an
**optimization** reason (affects LR):

- **Larger `B` → fewer, lower-variance gradient steps.** Per-epoch wall time
  drops (fixed per-step overhead amortizes, GPU occupancy improves on larger
  `B·N` matmuls) — but only until **throughput saturation**, after which doubling
  `B` just doubles per-step time. Empirically here, step time flattened past
  `B ≈ 4–16`.
- **Larger `B` usually needs a higher LR** to converge in the same number of
  *epochs* (fewer `update!` calls, lower gradient noise). Rough rules: **linear**
  scaling (Goyal et al., *Accurate, Large Minibatch SGD*, 2017, arXiv:1706.02677)
  or **sqrt** scaling — and sqrt is the better fit for **Adam**. Always validate
  on **time-to-target-loss**, not time-per-epoch.
- **Keep `precompute_batched` in sync** with the chosen `B` — `A_batched` is built
  for a fixed `B`.

**Does the `(B, LR)` optimum survive a `current_steps` (`S`) change?** No:
- *Memory side:* even with checkpointing, the step-boundary states grow slowly as
  `c₃·B·N·S`. A `B` tuned at `S=1` can OOM at `S=10`. → **Size `B` for
  `max(strategy.steps)`** to be safe throughout.
- *Optimization side:* increasing `S` changes the loss *function*, not just
  gradient variance. Longer BPTT unrolls amplify/ill-condition gradients, so the
  stable LR tends to **fall** as `S` grows. This is exactly why curriculum-on-
  rollout-length is itself an implicit stability schedule.

---

## 5. The curriculum learning-rate schedule

### The reasoning we validated
User's intuition: "a stable long-rollout solution lives near the single-step
optimum, so LR should decay as the curriculum advances." This is **directionally
right** (the single-step loss is the local linearization of the rollout loss; the
long-horizon basin is a *subset* of the one-step-good region) **but wrong if
applied as a strictly monotone, epoch-keyed decay** — because each time
`current_steps` jumps, a *harder, higher-curvature* objective switches on, and a
globally-decaying LR starves it of step size at the exact moment it needs a small
bump to cross the (small) displacement to the new optimum.

The principled answer is **warm restarts**: decay *within* a phase, mild reset
*at* each phase boundary.

*Reference:* Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm
Restarts* (2017), arXiv:1608.03983.

### ✅ Implemented — `curriculum_lr(ts, epoch)` in [src/training.jl](../src/training.jl)
Four properties:
1. **Cosine decay** within each phase (peak → `lr_final`).
2. **Linear warmup** of `lr_warmup_epochs` at each phase restart
   (`lr_final` → phase peak).
3. **Geometrically shrinking peaks**: `peak(p) = max(lr_start · lr_peak_decay^(p-1),
   lr_final)`.
4. **Phase boundaries driven by `strategy.durations`** (synced with
   `update_steps!`), *not* the epoch clock. Epochs past the last phase hold at
   `lr_final`.

Old fields `lr_steps` / the epoch-keyed `Step` schedule are retained only for TOML
backward-compat (ignored by `curriculum_lr`). New `TrainSettings` fields:
`lr_warmup_epochs` (default 1), `lr_peak_decay` (default 0.7).

### ✅ Implemented — per-epoch gradient-norm logging
`_grad_l2norm` traverses the structural gradient tree; `train_model!` accumulates
a per-epoch mean, shows it in the progress bar, and returns it as `grad_norm`.
**A rising `grad_norm` at phase restarts is the early-warning that a phase peak LR
is too hot** (i.e. `lr_peak_decay` too close to 1).

---

## 6. Tuning the schedule *without* a full training run

### The LR range test (Leslie Smith)
Ramp LR exponentially from ~1e-7 → ~1 over ~100–300 mini-batches, doing a real
optimizer step each batch; record loss + grad-norm. Max usable LR ≈ the point of
steepest loss descent, ~1 order of magnitude below the divergence knee.
Implemented as `--mode range` in
[scripts/lr_range_test.jl](../scripts/lr_range_test.jl).

*Reference:* Smith, *Cyclical Learning Rates for Training Neural Networks* (2017),
arXiv:1506.01186; popularized as `lr_find` in fast.ai.

**Key finding — run the range test at horizon 1.**
- A horizon-10 test *from scratch* is **meaningless**: an untrained model rolled
  out 10 autoregressive steps diverges exponentially (loss ~10¹⁸, `grad_norm =
  Inf`), with 13 orders of magnitude run-to-run variance. The LR never matters.
- The **horizon-1** curve is clean (loss 10.5 → 2.8, sane grad norms 10²–10⁴,
  diverging only at lr = 1.0). On `sava_small_v081` this gave **`lr_start ≈ 1e-2`**
  (steep zone, safely below the ~4e-1 knee; conservative upper bound
  `min_over_10 ≈ 7e-2`).
- To calibrate the *deep-horizon* peak you must range-test from a horizon-1/2
  **checkpoint**, not from scratch.

> ⚠️ **Known imperfection:** the auto-picked "steepest descent" scalar is unreliable
> because the range test's EMA smoothing (`beta = 0.98`, ~50-step memory) makes the
> early curve the EMA converging toward the mean batch loss, which the detector
> mistakes for the steepest drop. **Trust the human-readable curve and
> `min_over_10`, not the steepest scalar.** A pending improvement is to lower
> `beta` to ~0.9 and compute the slope on log-loss.

### Batch-size estimator
`--mode batch` in the same script probes increasing batch sizes, runs one real
fwd+bwd each, reports peak GPU memory / fraction / step-time, stops at OOM or a
`--mem-fraction` ceiling, and recommends the largest that fits (÷2 for headroom on
GPU). On `sava_small_v081` at horizon 10, **B = 1→64 all fit** (even B=64 used only
~2.5 GiB / 64 %); step time flattened past B≈4, so **B = 16–32** is the sweet spot.

### Estimating `lr_peak_decay` and `lr_final` without a pretrained model
Three routes (cheapest → most faithful):

- **Route 1 — theory default (chosen, "option 1").** Because we use **Adam**, not
  SGD, updates are divided by `√v̂` and are *largely invariant to gradient
  magnitude*. The naive "stable LR ~ 1/S" (SGD) rule massively overstates the
  needed decay. With Adam the residual horizon effect is gradient *noise*/surface
  sharpness — much milder. Plus the `MassBalanceLayer` is contractive and damps
  gradient growth. Defensible defaults with **zero measurement**:
  - `lr_peak_decay ≈ 0.7–0.85` (we use **0.8**).
  - `lr_final = lr_start / 50` to `lr_start / 100` (we use **`lr_start / 100`**) —
    a *design floor*, not something to measure.
  - `lr_start ≈ 1e-2` from the horizon-1 range test.
- **Route 2 — gradient-growth probe (not implemented).** One fwd+bwd per horizon
  `S ∈ {1,2,3,4,…}` at init (only while the forward is finite), record
  `‖∇‖_S`, fit the growth factor `g(S) = ‖∇‖_S / ‖∇‖_1`, extrapolate, set
  `lr_peak_decay` to the geometric fit of `1/g(S)`. Was proposed as a possible
  `--mode growth`; **not built**. Caveat: init-time Jacobian ratio is only a
  first-order proxy for a trained model.
- **Route 3 — self-calibrating via the curriculum (not implemented).** The
  curriculum never trains deep from scratch anyway. Run a short curriculum at a
  conservative fixed LR and read the already-logged `grad_norm` at each phase's
  first epoch; `g_p = grad_norm(phase p)/grad_norm(phase 1)` is the growth factor
  for an appropriately-trained model. Best cost/accuracy trade since the logging
  already exists, but **not built**.

---

## 7. The end-to-end auto-tune script

[scripts/autotune_train.jl](../scripts/autotune_train.jl) ties it together for a
given config TOML:
1. Estimate batch size at the **deepest** curriculum horizon (or `--batch-size`).
2. LR range test at **horizon 1** → `lr_start` (uses `min_over_10`; or
   `--lr-start`).
3. Apply **option-1** heuristics: `lr_peak_decay = 0.8`,
   `lr_final = lr_start / 100`.
4. Write the tuned config (comment-preserving text edit; `--in-place` keeps a
   `.bak`, else `<config>_autotuned.toml`) and launch `run_wflow_gnn_from_toml`.
   `--dry-run` tunes without training.

It reuses `load_graphs`, `make_model_loader`, `lr_range_test`, `recommend_lr`,
`estimate_batch_size` (via `include` of `lr_range_test.jl`, whose `main()` is now
guarded by `if abspath(PROGRAM_FILE) == @__FILE__`) plus `parse_run_config` /
`build_gnn_model` / `run_wflow_gnn_from_toml` from `run.jl`.

Templates [experiments/template.toml](../experiments/template.toml) and
[experiments/template_hparsearch.toml](../experiments/template_hparsearch.toml)
now document `lr_warmup_epochs`, `lr_peak_decay`, and the new meaning of
`lr_start` (first-phase peak) / `lr_final` (schedule floor); `lr_steps` is marked
deprecated.

---

## 8. Quick reference — recommended starting point

| Knob | Value | Source |
|---|---|---|
| `lr_start` | ~1e-2 | horizon-1 LR range test (per basin) |
| `lr_final` | `lr_start / 100` (~1e-4) | option-1 design floor |
| `lr_peak_decay` | 0.8 | option-1 (Adam-invariance argument) |
| `lr_warmup_epochs` | 1–2 | per-phase on-ramp after each `S` jump |
| `batch_size` | 16–32 | batch estimator (sized for `max(S)`) |
| checkpointing | on for `nsteps > 1` | memory lever for BPTT tape |

**Golden rules:**
1. Validate on **time-to-target-loss**, not time-per-epoch.
2. Range-test at **horizon 1**; trust the curve + `min_over_10`, not the steepest
   scalar.
3. Size `batch_size` for the **deepest** horizon to avoid mid-curriculum OOM.
4. Watch `grad_norm` at phase restarts — rising = peak LR too hot.

---

## References
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174 (2016).
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv:1608.03983 (2017).
- Smith, *Cyclical Learning Rates for Training Neural Networks*, arXiv:1506.01186 (2017). (fast.ai `lr_find`.)
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, arXiv:1706.02677 (2017). (Linear LR scaling.)
- Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (muP/µTransfer), arXiv:2203.03466 (2022). (Mentioned as the rigorous width-transfer option; not pursued.)
