# Efficient Message Passing — Notes & Benchmarks

A consolidated reference of the discussions on making GNN message passing fast on
GPU for `WflowRoutingGNN`, which culminated in
[scripts/benchmark_message_passing.jl](../scripts/benchmark_message_passing.jl)
and the `SparseConv` / `MassBalanceLayer` implementations in
[src/gnn.jl](../src/gnn.jl). It records the benchmark findings, the final design,
and the alternatives we considered but did **not** implement.

Companion documents:
[docs/architecture_alternatives.md](architecture_alternatives.md) (the original
brainstorm) and [docs/training_tuning_notes.md](training_tuning_notes.md) (memory,
BPTT, LR tuning — includes the `_topology_mul` gradient fix that also lives in the
message-passing code).

---

## 1. The problem: scatter-gather is slow on GPU

Standard GNN message passing (`GraphConv` / `propagate` in
`GraphNeuralNetworks.jl`) aggregates neighbours with **scatter-gather** kernels.
On GPU these are memory-bound, poorly coalesced, and underutilize the SMs — the
opposite of what GPUs are good at. The river-routing graph, however, is **fixed
for the entire run** (static topology), which opens up much faster fixed-topology
formulations.

The core idea: neighbour aggregation
`out[i] = Σ_{j→i} h[:,j]` is exactly a **sparse matrix–dense matrix multiply
(SpMM)** `neigh = (A · hᵀ)ᵀ`, where `A[i,j] = 1` iff `j` drains into `i`.
cuSPARSE SpMM is far better optimized than CUDA scatter kernels.

*References:* Gilmer et al., *Neural Message Passing for Quantum Chemistry*,
arXiv:1704.01212 (2017); Kipf & Welling, *Semi-Supervised Classification with
Graph Convolutional Networks*, arXiv:1609.02907 (2016) — both express propagation
as adjacency-matrix products.

---

## 2. What the benchmark compares

[scripts/benchmark_message_passing.jl](../scripts/benchmark_message_passing.jl)
builds a real wflow graph and pits equivalent single-layer models against each
other (matched to within 10 % trainable-parameter count), on CPU and GPU:

- **GraphConv COO** — `GraphConv` on a default (`:coo`) `GNNGraph` (scatter).
- **GraphConv sparse** — same layer on a `graph_type=:sparse` `GNNGraph`.
- **AdjMat dense** — custom layer, aggregation via dense `A * hᵀ`.
- **AdjMat sparse** — custom layer, aggregation via sparse `A * hᵀ` (what
  `SparseConv` uses).

It has three studies:
1. **Single-graph** timing (743-node basin) on CPU + GPU.
2. **Scaling** with graph size (100 → 10 000 nodes) via random spanning-tree
   graphs that mimic a river network (each node drains to one earlier node).
3. **Batched aggregation** — six strategies (below) across batch sizes 1–32 for
   two graph sizes (743 and 5000).

---

## 3. Benchmark findings

### Single graph (N = 743)
| Device | Fastest | Notes |
|---|---|---|
| CPU | GraphConv **sparse** (~1.85 ms) | AdjMat sparse ~1.95 ms, essentially tied; AdjMat **dense** ~3.7 ms (2× slower). |
| GPU | **AdjMat sparse ≈ AdjMat dense** (~0.26–0.28 ms) | ~1.6× faster than GraphConv sparse; scatter (GraphConv COO) slowest at ~0.43 ms. |

CPU→GPU speedups: AdjMat dense ~13×, AdjMat sparse ~7×, GraphConv sparse ~5.6×,
GraphConv COO ~4.4×. Below a few thousand nodes most of the forward pass is
**launch/overhead**, so small-graph timing differences are partly noise.

### Batched aggregation — the decisive study
Six strategies for the batched neighbour-aggregation step (a batch of `B` graphs
gives `B·N` feature columns):

| Strategy | What it does |
|---|---|
| `dense_reshape` | reshape trick with a **dense** `A` (BLAS/cuBLAS) |
| `sparse_reshape` | reshape trick with a **sparse** `A` (`h` → `(H·B, N)`, one SpMM) |
| `blockdiag` | build a block-diagonal `A_blk` (`B·N × B·N`) **every call**, then SpMM (the PyG approach) |
| `blkdiag_precomp` | build `A_blk` **once**, then only time the SpMM |
| `propagate_coo` | `GNNGraphs.batch` + `propagate` scatter |
| `propagate_sp` | same, but `graph_type=:sparse` |

**Key results (n = 5000, B = 32):**

| Strategy | GPU | CPU |
|---|---|---|
| **`blkdiag_precomp`** | **1.55 ms** | 109 ms |
| `sparse_reshape` | 2.97 ms | **45 ms** |
| `blockdiag` (build+run) | 4.75 ms | 108 ms |
| `propagate_coo` | 6.42 ms | 41 ms |
| `dense_reshape` | 31 ms | 363 ms |
| `propagate_sp` | 90 ms | 102 ms |

**Conclusions:**
- **GPU: precomputed block-diagonal wins (~2× vs sparse reshape) at every batch
  size and both graph sizes.** It presents a single `(B·N)×(B·N)` SpMM to
  cuSPARSE, which parallelizes across all batch elements at once. The reshape
  trick forces the batch dimension onto the *dense* side (`(H·B)×N`) and loses
  that parallelism.
- **CPU: sparse reshape wins** — the larger block-diagonal matrix causes more
  cache pressure, and `blockdiag` construction cost dominates.
- **Building the block-diagonal per call is wasteful** — with a static graph and
  fixed batch size you pay that cost **once**, which is exactly why
  `blkdiag_precomp` is the right training strategy.
- **`propagate_sp` is disqualified** — a constant ~10–90 ms CPU↔GPU transfer
  overhead per call regardless of size.
- **Dense adjacency explodes** in both time and memory (`O(N²)` storage).

---

## 4. The implemented design

`SparseConv` in [src/gnn.jl](../src/gnn.jl) stores three topology-related fields:

```julia
A          :: AbstractMatrix{Float32}                  # single-graph (N×N)
A_batched  :: Union{Nothing, AbstractMatrix{Float32}}  # block-diagonal (B·N × B·N)
batch_size :: Int                                       # B for A_batched; 0 = none
```

Forward pass dispatches on `size(h,2)` (three paths, priority order):
1. **Single graph** (`N_total == N_per`): direct `(A · hᵀ)ᵀ` SpMM — used by
   rollout/inference.
2. **Batched + precomputed** (`A_batched !== nothing`, matches `batch_size`):
   single `(B·N × B·N)` SpMM — the fast GPU training path.
3. **Batched fallback** (reshape trick): reshape `h` to `(H·B, N)`, multiply by
   `Aᵀ`, reshape back — no block-diagonal materialization, fastest on CPU and a
   safe path when the runtime batch size differs from `batch_size`.

The layer also adds a **residual** (`… .+ h`) and applies `W_self·h + W_neigh·neigh
+ bias`, i.e. a GraphConv-style self + neighbour transform.

`precompute_batched(model, B)` builds `A_batched = blockdiag(A, …, A)` (B copies,
`Int32` indices for cuSPARSE) — **call it before `Flux.gpu`**. One call handles
both `SparseConv` and `MassBalanceLayer`.

### Device movement — the subtle part
`A` / `A_batched` are **not** trainable and must be kept off the
Optimisers/Functors path, but still converted between `SparseMatrixCSC` (CPU) and
`CuSparseMatrixCSR` (GPU). Two things make this work:

- **`Functors.@functor SparseConv (W_self, W_neigh, bias)`** restricts traversal
  to the trainable fields. This is *different* from
  `Flux.@layer … trainable=(…)`: `trainable=` only restricts which fields get
  gradient **updates**, but `Optimisers.setup`/`Flux.gpu` still **traverse** the
  excluded fields via Functors — and hit the non-in-place-writable
  `CuSparseMatrixCSR`, which errors. `@functor` restricts the traversal itself.
  (We keep `@layer` too, for the pretty `show`.)
- **Explicit `Flux.gpu(::SparseConv)` / `Flux.cpu(::SparseConv)` overloads** that
  convert `A` and `A_batched` to the device-appropriate sparse type. Because
  Functors won't call these when walking the parent `WflowGNN`, there are also
  explicit `Flux.gpu(::WflowGNN)` / `Flux.cpu(::WflowGNN)` overloads that walk the
  `SparseConv` layers.

### MassBalanceLayer mirrors SparseConv
The physics layer's upstream-Q sum is the same aggregation, so
`MassBalanceLayer` got the identical treatment: `A_routing`, `A_routing_batched`,
`batch_size`, matching `gpu`/`cpu` overloads, and the same three dispatch paths.
Batched training was **broken** before this (the `N×N` `A_routing` couldn't
multiply a `(1, B·N)` `q`) — fixed by the block-diagonal path plus a reshape
fallback.

### Gradient caveat (cross-reference)
The batched block-diagonal `A_batched` is `(B·N)×(B·N)`; Zygote's generic `*`
rule would materialize a **dense** `∂A` (~17 GB at B=8, N=8235). All SpMM sites
route through `_topology_mul` with a custom `rrule` returning `NoTangent()` for
`A`. Details in [docs/training_tuning_notes.md](training_tuning_notes.md#1-the-oom-that-started-it-all).

---

## 5. Alternatives considered but NOT implemented

### `NNlib.batched_mul` for batching ✗
Requires **dense** 3D arrays — does not dispatch to cuSPARSE. Using it would force
`Matrix(A)` densification every forward pass, defeating the point. The reshape
trick performs the same arithmetic `batched_mul` would, but keeps `A` sparse.

### `graph_type=:csr` in `GNNGraph` ✗ (not supported)
`GraphNeuralNetworks.jl` (v1.1.0) supports only `:coo`, `:sparse` (CSC), and
`:dense` — there is no `:csr`. This is why the fast path uses a `CuSparseMatrixCSR`
**outside** the `GNNGraph`, in the custom layer.

### `graph_type=:sparse` + `propagate` ✗
Comparable to sparse-adjacency on CPU, but on GPU it carries a constant ~10–90 ms
CPU↔GPU transfer overhead per call (`propagate_sp` in the benchmark). Disqualified.

### Dense adjacency ✗
`O(N²)` storage and time; unusable past a few thousand nodes (363 ms at n=5000,
B=32 on CPU).

### LDD-masked 2D shifts (option 2 in the brainstorm) — not pursued
The LDD is a raster where each cell drains in one of 8 directions, so the upstream
sum is exactly **8 masked array shifts**
(`upstream_q += shift(q, −dr, −dc) .* (ldd .== d)`) — no scatter, fully coalesced.
Would only accelerate the `MassBalanceLayer` step and requires unpack→shift→repack
between the compressed node list and the full raster. Left as a future targeted
optimization for the physics step.

### Full 2D CNN on the raster (option 3) — not pursued
Replace the whole GNN with a U-Net / dilated-CNN on the `(C, H, W)` raster: dense
3×3 convs (heavily cuDNN-optimized) capture the 8-direction LDD neighbourhood;
dilated convs expand the receptive field. Fastest primitive available, but loses
explicit topology (directionality must be learned, e.g. by injecting LDD as input
channels) and needs a full rewrite of `gnn.jl` / `preprocess.jl` / the pipeline.
Reserved as a long-term option if GNN throughput becomes the bottleneck at scale.

*Reference for the CNN direction:* Shi et al., *Convolutional LSTM Network*,
arXiv:1506.04214 (2015); U-Net, Ronneberger et al., arXiv:1505.04597 (2015).

### Rollout input buffer (`x_buf`) pre-allocation — tried, reverted
Pre-allocating the `vcat(state, forcing, static)` buffer gave **no improvement**:
CUDA's memory pool already reuses those allocations, and below a few thousand
nodes the rollout is dominated by per-step kernel-launch overhead, not allocation.
The measured rollout variance meant apparent differences were largely noise.

---

## 6. Practical guidance

- **Training on GPU with a fixed batch size and static graph:** call
  `precompute_batched(model, batch_size)` **before** `Flux.gpu(model)` to get the
  ~2× block-diagonal SpMM path. This is wired into `run_wflow_gnn`.
- **Rollout / inference:** the single-graph `(N×N)` SpMM path is used
  automatically — no precomputation needed.
- **CPU:** the reshape-trick fallback is fastest; the block-diagonal offers no CPU
  benefit.
- **Don't** rely on `Flux.@layer trainable=` alone to keep sparse matrices off the
  optimizer path — you also need `Functors.@functor` field restriction + explicit
  `gpu`/`cpu` overloads.
- **Small graphs (≲ few k nodes):** message passing is overhead-bound; expect
  noisy speed comparisons and limited absolute gains.

### Model expressiveness (aside)
When single-step h-loss stayed high, we added **residual connections** in
`SparseConv` and a configurable number of encoder/decoder MLP layers
(`mlp_layers`) rather than only widening `hidden_dim` — deeper message passing +
residuals improve long-range routing without destabilizing training. Multiple
**edge features** are not naturally compatible with the single scalar adjacency
matrix; that would require either per-feature adjacency matrices or reverting to
`propagate` for those terms.

---

## 7. Quick reference

| Question | Answer |
|---|---|
| Fastest single-graph GPU aggregation | sparse (or dense) `A · hᵀ`, ~1.6× over scatter |
| Fastest **batched** GPU aggregation | **precomputed block-diagonal SpMM** (~2× over reshape) |
| Fastest batched **CPU** aggregation | sparse reshape trick |
| `batched_mul`? | No — densifies, no cuSPARSE dispatch |
| `graph_type=:csr`? | Not supported by GNN.jl |
| Keep `A` off optimizer path | `Functors.@functor` restriction + `gpu`/`cpu` overloads |
| Avoid dense `∂A` (~17 GB) | `_topology_mul` custom `rrule` |

---

## References
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, arXiv:1704.01212 (2017).
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, arXiv:1609.02907 (2016).
- Fey & Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*, arXiv:1903.02428 (2019). (Block-diagonal batching of graphs.)
- Shi et al., *Convolutional LSTM Network*, arXiv:1506.04214 (2015). (2D-CNN / ConvLSTM emulator direction.)
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, arXiv:1505.04597 (2015).
- `GraphNeuralNetworks.jl` docs — `GNNGraph` `graph_type` (`:coo` / `:sparse` / `:dense`), `propagate`.
- NVIDIA cuSPARSE SpMM; `NNlib.batched_mul` (dense-only).
