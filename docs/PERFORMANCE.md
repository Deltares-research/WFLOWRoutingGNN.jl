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