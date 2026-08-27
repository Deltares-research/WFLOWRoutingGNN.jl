# Recommended Reading: Efficient Edge-Feature Message Passing in GNNs

This reading list focuses on graph neural network architectures and systems techniques that make edge-dependent message passing more computationally efficient. The papers are ordered from architectural decompositions to low-level kernel fusion.

## Suggested reading order

1. **R-GCN** for basis-decomposed edge-specific transformations.
2. **MoNet** for learnable edge-coordinate weighting kernels.
3. **SplineCNN** for continuous kernels represented with locally supported basis functions.
4. **FusedMM** for avoiding materialised edge-message tensors.
5. **fuseGNN** and **DF-GNN** for GPU kernel fusion and systems-level optimisation.

---

## 1. Modelling Relational Data with Graph Convolutional Networks

**Citation**  
Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). *Modelling Relational Data with Graph Convolutional Networks*. ESWC 2018.

- [arXiv abstract and paper](https://arxiv.org/abs/1703.06103)
- DOI: [10.48550/arXiv.1703.06103](https://doi.org/10.48550/arXiv.1703.06103)

### Why it is relevant

R-GCN introduces a basis decomposition for relation-specific transformation matrices:

$$
W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}.
$$

The relation type, which is a discrete edge feature, determines coefficients over a small set of shared basis matrices. This constrains the family of edge-conditioned operators and prevents the number of independent parameters from growing linearly with the number of relation types.

### Connection to efficient message passing

A continuous-edge analogue would be:

$$
W(e_{ij}) = \sum_{b=1}^{B} a_b(e_{ij})V_b,
$$

where a compact function of the edge attributes predicts the basis coefficients. This is best described as a **low-dimensional operator basis** rather than necessarily a low-rank matrix decomposition. The basis matrices may still be full rank.

---

## 2. Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs

**Citation**  
Monti, F., Boscaini, D., Masci, J., Rodolà, E., Svoboda, J., & Bronstein, M. M. (2017). *Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs*. CVPR 2017.

- [arXiv abstract and paper](https://arxiv.org/abs/1611.08402)
- DOI: [10.48550/arXiv.1611.08402](https://doi.org/10.48550/arXiv.1611.08402)

### Why it is relevant

MoNet uses local pseudo-coordinates on graph or manifold neighbourhoods and a family of learnable weighting functions, including Gaussian kernels. It is an important example of decomposing an edge-dependent convolution into a finite collection of kernels.

A useful abstraction is:

$$
m_{ij} = \sum_{k=1}^{K} \phi_k(e_{ij})\,\psi_k(h_j),
$$

where $\phi_k$ evaluates an edge feature or pseudo-coordinate and $\psi_k$ transforms the neighbouring node feature.

### Connection to efficient message passing

If $\psi_k(h_j)=B_kh_j$, define a weighted sparse adjacency matrix for each kernel:

$$
[A_k]_{ij} = \phi_k(e_{ij}).
$$

The node update can then be written as:

$$
H' = \sum_{k=1}^{K} A_k H B_k.
$$

This converts an edge-conditioned operation into a small number of weighted sparse-dense matrix multiplications. This reformulation follows algebraically from the separable message form; whether it is efficient depends on keeping $K$ small.

---

## 3. SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels

**Citation**  
Fey, M., Lenssen, J. E., Weichert, F., & Müller, H. (2018). *SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels*. CVPR 2018, 869–877.

- [CVPR open-access paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Fey_SplineCNN_Fast_Geometric_CVPR_2018_paper.html)
- [arXiv abstract and paper](https://arxiv.org/abs/1711.08920)
- DOI: [10.48550/arXiv.1711.08920](https://doi.org/10.48550/arXiv.1711.08920)

### Why it is relevant

SplineCNN parameterises a continuous edge-conditioned convolution kernel using B-spline basis functions. B-splines have local support, so only a subset of the basis functions is active for a particular pseudo-coordinate.

### Connection to efficient message passing

This paper is particularly useful when edge features encode geometric quantities such as relative position, direction or distance. It demonstrates how a structured basis can retain continuous edge dependence without requiring an unconstrained dense operator to be generated independently for every edge.

---

## 4. FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks

**Citation**  
Rahman, M. K., Sujon, M. H., & Azad, A. (2021). *FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks*. IPDPS 2021.

- [arXiv abstract and paper](https://arxiv.org/abs/2011.06391)
- DOI: [10.48550/arXiv.2011.06391](https://doi.org/10.48550/arXiv.2011.06391)

### Why it is relevant

Edge-message generation can often be viewed as a sampled dense-dense matrix multiplication, or SDDMM-like operation, followed by sparse-dense matrix multiplication for destination-node aggregation. Executing these operations separately can require a high-dimensional intermediate message for every edge.

FusedMM combines the two phases so that messages can be generated and aggregated without explicitly storing the complete edge-message tensor.

### Connection to efficient message passing

This is the most direct systems reference for the pattern:

1. gather source and destination features;
2. compute an edge-dependent message;
3. reduce the message into the destination node;
4. avoid writing the intermediate edge message to global memory.

It is therefore central reading when memory traffic, rather than arithmetic, is the dominant bottleneck.

---

## 5. fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU

**Citation**  
Chen, Z., Yan, M., Li, G., & Li, S. (2020). *fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU*. ICCAD 2020.

- [View the paper](https://mingyuyan-ict.github.io/MingyuYan-ICT/files/fuseGCN.pdf)

### Why it is relevant

fuseGNN develops dedicated GPU kernels for graph processing and aggregation and applies kernel fusion to reduce kernel-launch overhead, latency, data movement and storage footprint.

### Connection to efficient message passing

This provides broader systems context for why separate gather, edge computation and scatter operations may underperform even when their arithmetic complexity is linear in the number of edges. The memory hierarchy, data reuse and kernel boundaries are often as important as the asymptotic operation count.

---

## 6. DF-GNN: Dynamic Fusion Framework for Attention Graph Neural Networks on GPUs

**Citation**  
Liu, J., Cai, Z., Chen, Z., & Wang, M. (2024). *DF-GNN: Dynamic Fusion Framework for Attention Graph Neural Networks on GPUs*.

- [arXiv abstract and paper](https://arxiv.org/abs/2411.16127)
- DOI: [10.48550/arXiv.2411.16127](https://doi.org/10.48550/arXiv.2411.16127)

### Why it is relevant

DF-GNN focuses on attention-based GNNs, where edge coefficients are computed dynamically before aggregation. It addresses data movement and kernel-launch overhead through dynamic kernel fusion and workload-aware thread scheduling.

### Connection to efficient message passing

Attention is an important special case of edge-feature message passing:

$$
m_{ij} = \alpha_{ij} W h_j.
$$

Once the scalar edge coefficient $\alpha_{ij}$ is available, aggregation is a weighted sparse operation. DF-GNN is useful for understanding how the coefficient computation and aggregation can be fused or scheduled without sacrificing the flexibility of attention-style models.

---

## Synthesis: a promising design pattern

A practical compromise between expressiveness and efficiency is:

$$
m_{ij}
= \sum_{k=1}^{K} a_k(e_{ij}) B_k h_j,
\qquad K \ll d.
$$

This design:

- retains more edge expressivity than scalar gating;
- avoids generating a full $d_{\text{out}}\times d_{\text{in}}$ matrix for every edge;
- admits a weighted-SpMM interpretation when $K$ is small;
- can instead be implemented as a fused edge-centric kernel when repeated SpMMs are inefficient;
- allows further factorisation of $B_k$ if matrix rank also needs to be constrained.

The key implementation question is whether the intermediate edge quantities $a_k(e_{ij})$ and messages should be materialised. The architectural papers motivate compact edge-conditioned bases, while the systems papers motivate fusing their evaluation and aggregation to minimise memory traffic.
