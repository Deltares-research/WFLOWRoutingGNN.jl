# Inference vs. Training Timing — Findings & Measurement Guide

Model: `experiments/test_sava_v081` (river domain, schema v0.8.1), 8235 nodes,
hidden 64, 3 SparseConv layers. GPU: NVIDIA RTX A1000 Laptop (4 GiB).

## 1. The confusion we cleared up

A "~20 ms/step on GPU" figure from late July was being compared against a
"~3 ms/step on GPU" figure from the new rollout benchmark, suggesting a 6×
speedup. **There was no speedup.** The two numbers measure different things:

| Number | What it actually is |
|--------|---------------------|
| ~2–3 ms/step (GPU) | one **forward pass** = a real rollout/inference step |
| ~18–27 ms/batch (GPU) | one **training-loop iteration** (fwd+bwd + extra forwards + host syncs + device copy + optimiser update) |

The model forward compute never changed. The only code change to `rollout` this
week (commit `eb24719`) added a warm-up step, a per-step `CUDA.synchronize()`,
and median reporting — it did **not** touch the compute path.

## 2. Direct measurement (same model, graph, device)

Measured with `scripts/benchmark_inference_vs_train.jl` (40 reps, warm-up 3,
`CUDA.synchronize()` after every timed call), per-step in ms:

| device | operation | min | median | mean | std | max |
|--------|-----------|----:|-------:|-----:|----:|----:|
| cpu | forward (inference) | 24 | 25   | 26.4  | 7.0   | 65  |
| cpu | fwd+bwd (gradient)  | 62 | 70   | 112.9 | 149.4 | 846 |
| cpu | full train step     | 63 | 68.5 | 101.8 | 114.9 | 586 |
| gpu | forward (inference) | 1  | 2    | 4.6   | 10.0  | 55  |
| gpu | fwd+bwd (gradient)  | 4  | 5    | 7.7   | 9.4   | 60  |
| gpu | full train step     | 6  | 6    | 12.0  | 21.7  | 124 |

Ratios: GPU gradient/forward = 2.5×, full-step/forward = 3.0×; CPU ≈ 2.8×.
This is the textbook forward:(forward+backward) ratio — nothing anomalous.

### Why the training loop is even slower than a single gradient step

Per batch, the training loop (`src/training.jl`) does substantially more than
one gradient step:

- `Flux.withgradient` (forward + backward),
- a separate `one_step_loss` forward,
- a separate `loss_components` forward,
- gradient-L2-norm reduction and `isfinite` checks (host↔device syncs),
- `Flux.update!` (ClipNorm + Adam),
- per-batch `dev_fn.(batch)` host→GPU transfer,
- **no** per-step `CUDA.synchronize()`, so async launch timing is noisy.

Stacking these on top of a ~6 ms gradient step lands in the observed
18–27 ms/batch range. None of this is part of an inference rollout.

## 3. The number that matters for the surrogate decision

For "is a surrogate worth building vs. running wflow", use the **forward-only
inference** cost:

- **GPU single rollout: ~2 ms/step** (≈25 ms/step on CPU).
- **GPU ensemble (B=16): ~1.2 ms/step/member** — batching amortises kernel-launch
  latency across members (same total FLOPs, more work per launch).

Do **not** use training-loop per-batch times as the inference cost.

### Rollout benchmark: single vs. ensemble (`scripts/benchmark_rollout.jl`)

Measured on `experiments/test_sava_v081`, 30-step window, 16 ensemble members
(same initial state and forcing for every member). Per-timestep, per-member ms:

| device | mode | members | total (s) | min | median | mean | std | max |
|--------|------|--------:|----------:|----:|-------:|-----:|----:|----:|
| cpu | single   | 1  | 0.836  | 23.0   | 25.0   | 27.9 | 11.7 | 82.0 |
| cpu | ensemble | 16 | 20.814 | 31.6   | 36.0   | 43.4 | 14.7 | 71.1 |
| gpu | single   | 1  | 0.233  | 2.0    | 3.0    | 7.8  | 10.7 | 34.0 |
| gpu | ensemble | 16 | 0.581  | 1.13   | 1.19   | 1.21 | 0.09 | 1.63 |

Per-member median speedup (ensemble vs. single): **CPU 0.69×, GPU 2.53×**.

- On **CPU**, ensembling is a slight loss — no kernel-launch latency to
  amortise, so `B` members is just `B×` the serial work plus batching overhead.
- On **GPU**, ensembling gives a **2.53× per-member throughput gain** and much
  tighter variance (std 0.09 ms vs. 10.7 ms for single): the single-member GPU
  step is latency-bound on kernel launches, and batching fills the GPU to
  amortise that latency. This is exactly the win the ensemble path was built for.

## 4. How to measure these timings properly

### General
- **Warm up first.** Discard the first 1–3 calls so Julia JIT compilation and
  CUDA kernel autotuning are excluded from the reported statistics.
- **Report median, not mean.** Wall-clock timing has a heavy right tail (GC,
  OS scheduling, allocator, CUDA autotune). Median is the stable central value;
  keep min/mean/std/max for context.
- **Measure the right unit.** An inference/rollout step = one forward pass.
  A training step = forward + backward (+ optimiser update). Never compare a
  training-loop per-batch time to a forward-only step.
- **Isolate the operation.** Time exactly the call of interest, not the
  surrounding bookkeeping (logging, metric side-forwards, host transfers).

### GPU-specific (essential — otherwise the numbers are meaningless)
- **Synchronize before reading the clock.** CUDA kernels launch
  asynchronously: the CPU returns from `model(...)` before the GPU finishes.
  Call `CUDA.synchronize()` after the operation and before `time()`/`time_ns()`,
  otherwise you are timing the launch, not the compute.
- **Sync once at the start too**, after warm-up, so pending async work from the
  warm-up isn't charged to the first timed step.
- **Keep data on the device.** Allocate the output trajectory as a device array
  (`similar(state, …)`) and copy to host **once** after the loop. A per-step
  host copy adds a hidden device→host sync and dominates the measurement.
- **Watch the timer resolution.** On Windows, `time()` quantises to ~1 ms, which
  is coarse relative to a ~2 ms GPU step (values look like integers). For
  sub-millisecond fidelity use `time_ns()` (nanosecond counter) or, best,
  **CUDA events** (`CUDA.@elapsed` / `CUDA.CuEvent`) which time on the GPU
  timeline and need no host sync.
- **Account for the launch floor.** At small problem sizes GPU per-step time is
  dominated by kernel-launch latency, not arithmetic. A near-constant `min`
  across configs is the launch floor. Batching (ensembles) is the lever that
  amortises it.
- **Report per-member for batched runs.** For an ensemble of `B`, divide the
  batched step time by `B` to get the per-member cost.

### Reproduce
```powershell
# inference vs. gradient vs. full train step, CPU + GPU
julia --project=. scripts/benchmark_inference_vs_train.jl experiments/test_sava_v081 --reps 40 --devices cpu,gpu

# rollout benchmark: single vs. ensemble, CPU + GPU
julia --project=. scripts/benchmark_rollout.jl experiments/test_sava_v081 --members 16 --timesteps 30 --devices cpu,gpu
```

> Note on exit codes: locally, CUDA 13.1 vs. precompiled-13.2 prints a benign
> stderr warning that makes PowerShell report a non-zero exit code even on
> success. Verify results from the printed table/log, not the exit code.
