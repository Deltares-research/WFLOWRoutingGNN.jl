# Architecture alternatives for faster GPU routing

GNNs tend to be slow on GPU due to scatter-gather operations underlying message passing.
Three alternatives are discussed below, ordered by implementation effort.

---

## 1. Sparse adjacency matrix multiplication (drop-in)

**Effort**: low
**Scope**: replaces scatter-gather in `GraphConv` and `propagate` calls

The river network adjacency matrix A is fixed for the entire run. cuSPARSE SpMM
(sparse-dense matrix multiplication) is significantly better optimised on GPU than
CUDA scatter kernels.

- Precompute A once as a `CUDA.CUSPARSE.CuSparseMatrixCSR`.
- Replace `propagate` calls with `A * q` (standard SpMM).
- `GraphNeuralNetworks.jl` has a `graph_type=:csr` option in the `GNNGraph`
  constructor that may route through SpMM internally. Try switching this in
  `preprocess.jl` first — if it hits cuSPARSE you get the speedup for free.

**Limitations**: still limited by sparse-matrix bandwidth of the GPU.
Does not change the fundamental scatter pattern for the `GraphConv` processor layers.

---

## 2. LDD-masked 2D shifts (targeted, likely fastest for the physics part)

**Effort**: medium
**Scope**: replaces the `propagate` call in `MassBalanceLayer` only

The LDD is a 2D raster where each cell drains in one of 8 directions.
"Sum Q of all upstream neighbours" is exactly 8 masked array shifts:

```
for each direction d with raster offset (dr, dc):
    upstream_q += shift(Q, −dr, −dc) .* (ldd .== d)
```

`shift` is a `circshift`-style padded roll. On GPU this is 8 element-wise
multiplies and adds — no scatter, fully coalesced memory access.

**Implementation notes**:
- The GNN currently operates on a compressed node list (river cells only).
  For this approach, unpack to the full 2D raster, do shifts, repack.
  Unpacking is cheap; the full grid for a typical wflow domain is small.
- Only replaces `propagate` in `MassBalanceLayer`; the GNN processor still
  uses scatter unless option 3 is also adopted.

**Limitations**: only accelerates the physics constraint step, not the GNN itself.

---

## 3. Full 2D CNN on the raster (architectural replacement)

**Effort**: high
**Scope**: replaces the entire GNN encoder-processor-decoder

Replace `WflowGNN` with a U-Net-style or dilated CNN operating on the 2D raster:

- State, forcing, and static variables become channels of a 2D feature map `(C, H, W)`.
- Standard `Conv((3,3), C=>hidden)` layers aggregate 8-direction neighbourhoods,
  which is precisely the LDD connectivity pattern plus self.
- Dilated convolutions `(3,3, dilation=2^k)` cheaply expand the receptive field
  to capture long-range routing without extra parameters.
- The river mask is applied as an input channel and/or as output masking.

**Advantages**:
- Dense 3x3 convolutions are the most heavily optimised GPU primitive (cuDNN fused kernels).
- No scatter, no graph data structures, no `GNNGraph` batching overhead.
- Architecturally similar to ConvLSTM / U-Net weather emulators that scale well.

**Disadvantages**:
- Loses explicit graph topology; directionality must be learned from data
  (mitigated by injecting the LDD direction as additional input channels).
- Output padding and masking for non-river cells adds complexity.
- Requires a significant rewrite of `gnn.jl`, `preprocess.jl`, and the training pipeline.

---

## Recommendation

1. **Quick win**: try `graph_type=:csr` in the `GNNGraph` constructor in `preprocess.jl`.
   If GNN.jl routes through cuSPARSE SpMM, option 1 is essentially free.
2. **Targeted physics fix**: replace the `propagate` in `MassBalanceLayer` with
   8 masked shifts (option 2), eliminating scatter from the physics-critical step.
3. **Long-term**: option 3 (2D CNN) if GNN throughput remains the bottleneck at scale.
