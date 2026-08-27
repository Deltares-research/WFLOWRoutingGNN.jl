# Efficient Message Passing with Edge Features

## Overview

Sparse adjacency matrix multiplication is highly effective for graph neural networks when every edge applies the same linear message rule, for example

$$
H' = AHW.
$$

The limitation is that a general edge-aware message

$$
m_{ij}=m(h_i,h_j,e_{ij})
$$

cannot usually be represented by one scalar-valued sparse adjacency matrix. A fully general implementation therefore gathers node features onto edges, computes an edge message, and scatters or reduces those messages back to destination nodes. This is expressive, but can materialise large edge-sized tensors and create substantial memory traffic.

The central design question is therefore:

> Can the dependence on the edge feature be constrained or factorised so that edge-aware message passing retains sparse-matrix-like efficiency?

Our discussion identified three main formulations:

1. **scalar edge-basis channels**;
2. **implicit low-rank edge blocks**;
3. **full edge-conditioned block-sparse adjacency**.

The first two avoid storing a full feature-mixing matrix per edge. The third represents that matrix explicitly and is mainly attractive for small, fixed, dense blocks.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of nodes |
| $E$ | number of directed edges |
| $F_{in}$ | input node-feature width |
| $F_{out}$ | output node-feature width |
| $D_e$ | raw edge-feature width |
| $K$ | number of scalar edge bases or channels |
| $r$ | latent rank |
| $X\in\mathbb R^{N\times F_{in}}$ | node-feature matrix |
| $Q\in\mathbb R^{E\times D_e}$ | edge-feature matrix |
| $s_e,t_e$ | source and destination of edge $e$ |

The estimates below describe a forward pass, omit biases, activations and normalisation, and assume sum aggregation. They are algorithmic scaling estimates, not measured runtimes.

---

## 1. Conventional sparse aggregation and its limitation

With no multidimensional edge dependence, aggregation can be written as

$$
Z=AX,
$$

where $A\in\mathbb R^{N\times N}$ is sparse. If all edges share a transformation $W$, linearity gives

$$
A(XW)=(AX)W.
$$

This allows the dense transform to be placed before or after sparse aggregation according to feature widths and implementation efficiency.

Once every edge has a rich feature vector, a general message such as

$$
m_e=\operatorname{MLP}([x_{s_e},x_{t_e},q_e])
$$

is no longer a conventional sparse matrix multiplication. The common gather-compute-scatter route is then

1. gather $x_{s_e}$ and, if needed, $x_{t_e}$;
2. compute $m_e$ for every edge;
3. reduce $m_e$ by $t_e$.

Its basic message computation scales with $E$, but the intermediate edge tensors can scale as $O(EF)$ and may dominate memory use. The FusedMM work explicitly treats message generation and aggregation as operations that can be fused, avoiding an otherwise separate edge-level intermediate.

---

## 2. Batching edge channels through stacked sparse operators

Ordinary graph batching places graph adjacencies on the diagonal of one block-diagonal matrix. A similar idea can separate edge-feature channels, but because all channels usually act on the same node set, **vertical stacking** is leaner than a true block diagonal.

For $K$ edge channels, construct

$$
A_1,\ldots,A_K\in\mathbb R^{N\times N}
$$

and stack them:

$$
A_{stack}=
\begin{bmatrix}
A_1\\
\vdots\\
A_K
\end{bmatrix}
\in\mathbb R^{KN\times N}.
$$

One sparse multiplication produces

$$
A_{stack}X=
\begin{bmatrix}
A_1X\\
\vdots\\
A_KX
\end{bmatrix}.
$$

The result can be reshaped to $K\times N\times F_{in}$ and mixed over the channel dimension. A true block diagonal,

$$
\operatorname{diag}(A_1,\ldots,A_K),
$$

requires $K$ copies of the node matrix and is mainly useful when every channel also has a distinct node representation.

A useful interpretation is a lifted bipartite graph. Each destination node $i$ becomes $K$ virtual destination states $(i,k)$, while the original source nodes are retained. Folding the virtual states over $k$ gives the updated nodes.

---

# Three implementation variants

## Variant A: scalar edge-basis channels

### Construction

The node features remain

$$
X\in\mathbb R^{N\times F_{in}}.
$$

An edge encoder maps every raw edge feature to $K$ coefficients:

$$
\phi_e=f_\phi(q_e)\in\mathbb R^K,
\qquad
\Phi\in\mathbb R^{E\times K}.
$$

For each basis $k$, construct a scalar sparse matrix

$$
(A_k)_{ij}=
\sum_{e:s_e=j,t_e=i}c_e\Phi_{e,k},
$$

where $c_e$ may contain degree normalisation or another scalar edge weight.

Each basis has a shared node transformation

$$
W_k\in\mathbb R^{F_{in}\times F_{out}}.
$$

### Full message pass

The per-edge message is

$$
m_e=
\sum_{k=1}^{K}
 c_e\Phi_{e,k}\,x_{s_e}W_k.
$$

The complete layer is

$$
Y=\sum_{k=1}^{K}A_kXW_k.
$$

Because each $W_k$ is shared within a basis, two equivalent schedules are available:

**Transform first**

$$
T_k=XW_k,
\qquad
Y=\sum_kA_kT_k.
$$

**Aggregate first**

$$
Z_k=A_kX,
\qquad
Y=\sum_kZ_kW_k.
$$

### Estimated scaling

For a linear edge encoder $\Phi=QW_\phi+b$:

- edge encoding: $O(ED_eK)$;
- shared node transforms: $O(KNF_{in}F_{out})$;
- sparse aggregation: $O(KEF)$, where $F$ is $F_{out}$ for transform-first and $F_{in}$ for aggregate-first;
- edge coefficients: $O(EK)$ memory;
- message parameters: $O(KF_{in}F_{out})$;
- largest straightforward channel intermediate: $O(KNF_{out})$ or $O(KNF_{in})$.

If all channels share the same edge index arrays, topology storage can remain $O(E+N)$ with $O(EK)$ values. Independently storing $K$ CSR matrices may repeat indices and approach $O(KE+KN)$ index storage.

### Strengths and limits

This formulation works well when continuous edge features can be represented by a small set of scalar basis functions, or when edge types are discrete. It remains close to standard SpMM. Its cost grows with $K$, and a dense activation of all $K$ bases for every edge creates $EK$ values and potentially a large $K\times N\times F$ intermediate. Local bases or sparse top-$q$ activation can reduce this.

R-GCN provides a clear precedent for relation-specific transformations and basis decomposition. MoNet and SplineCNN provide relevant examples of edge or geometric coordinates controlling a finite collection of continuous weighting kernels.

---

## Variant B: implicit low-rank edge blocks

### Construction

Project each node into an $r$-dimensional message space:

$$
P=XV,
\qquad
V\in\mathbb R^{F_{in}\times r},
\qquad
P\in\mathbb R^{N\times r}.
$$

Map each edge feature to an $r$-dimensional gate:

$$
g_e=f_g(q_e)\in\mathbb R^r,
\qquad
G\in\mathbb R^{E\times r}.
$$

Use a shared output projection

$$
U\in\mathbb R^{r\times F_{out}}.
$$

The implicit edge-specific block is

$$
B_e=U^\top\operatorname{diag}(g_e)V^\top
$$

under a column-vector convention. It is never necessary to materialise $B_e$.

### Full message pass

1. Project nodes: $P=XV$.
2. Encode edge gates: $G=f_g(Q)$.
3. Form latent edge messages:

$$
\ell_e=c_e\,g_e\odot P_{s_e}.
$$

4. Reduce them:

$$
S_i=\sum_{e:t_e=i}\ell_e.
$$

5. Project to output features:

$$
Y=SU.
$$

Equivalently,

$$
Y_i=
\left[
\sum_{e:t_e=i}
 c_e f_g(q_e)\odot(x_{s_e}V)
\right]U.
$$

### Estimated scaling

For a linear edge encoder:

- edge encoding: $O(ED_er)$;
- input node projection: $O(NF_{in}r)$;
- edge gating and aggregation: $O(Er)$;
- output projection: $O(NrF_{out})$;
- edge-dependent values: $O(Er)$;
- shared node intermediates: $O(Nr)$;
- message parameters, including a linear edge encoder: $O(r(F_{in}+F_{out}+D_e))$.

An unfused implementation may construct an $E\times r$ latent message tensor. A fused gate-and-reduce kernel can avoid materialising that complete tensor.

### Strengths and limits

This is the strongest general efficiency candidate when $r\ll F_{in},F_{out}$. It traverses each edge once, stores only $r$ dynamic coefficients per edge, and places cross-feature mixing in shared node-level projections. Its limitation is expressiveness: every edge operator belongs to a shared diagonal-gated rank-$r$ family.

This variant can also be viewed as a structured block adjacency. It achieves local, edge-conditioned feature modulation without the $F_{in}F_{out}$ storage cost of a free block per edge.

---

## Variant C: full edge-conditioned block-sparse adjacency

### Construction

The node matrix is

$$
X\in\mathbb R^{N\times F_{in}}.
$$

An edge network generates a full block:

$$
B_e=f_B(q_e)
\in\mathbb R^{F_{out}\times F_{in}}.
$$

Across all edges, the block tensor has shape

$$
E\times F_{out}\times F_{in}.
$$

These blocks form a global block-sparse operator

$$
A_B\in\mathbb R^{NF_{out}\times NF_{in}},
$$

with block $(i,j)$ equal to the sum of blocks for edges from $j$ to $i$.

Block Sparse Row, or BSR, storage is analogous to CSR but stores one fixed-size dense block at every non-zero block position. This saves repeated graph indices and permits regular arithmetic inside each block, but all entries of a stored block occupy memory, including explicit zeros.

### Full message pass

For edge $e$:

$$
m_e=c_eB_ex_{s_e}.
$$

Then

$$
y_i=\sum_{e:t_e=i}m_e.
$$

Globally:

$$
\operatorname{vec}(Y)=A_B\operatorname{vec}(X).
$$

A fused block multiply-and-reduce kernel can avoid storing every $m_e$, but it cannot avoid reading or producing the block coefficients unless they are generated on the fly.

### Estimated scaling

For already available blocks:

- block application: $O(EF_{in}F_{out})$;
- dynamic block storage: $O(EF_{in}F_{out})$;
- block-level topology indices: $O(E+N)$;
- unfused edge-message intermediate: $O(EF_{out})$.

If a linear edge encoder generates the blocks directly:

$$
\operatorname{vec}(B_e)=q_eW_B+b,
$$

then:

- block generation: $O(ED_eF_{in}F_{out})$;
- generator parameters: $O(D_eF_{in}F_{out})$.

### Strengths and limits

This is the most expressive variant because every edge can have its own dense feature-mixing transformation. It is likely to be practical only when blocks are small, fixed-size, genuinely dense and supported by an efficient BSR kernel. It is especially plausible for compact physical state vectors such as $2\times2$ or $4\times4$ local couplings.

For ordinary hidden widths, the quadratic edge storage is severe. At $F_{in}=F_{out}=64$, a full edge block contains 4,096 values. If the block is internally sparse, BSR still stores its complete fixed-size block, making a diagonal, low-rank, basis-mixture or Kronecker structure preferable.

---

## Direct comparison

| Property | Scalar bases | Low-rank blocks | Full BSR blocks |
|---|---:|---:|---:|
| Node features | $N\times F_{in}$ | $N\times F_{in}$ | $N\times F_{in}$ |
| Dynamic edge representation | $E\times K$ | $E\times r$ | $E\times F_{out}\times F_{in}$ |
| Message | $\sum_k\phi_{e,k}x_jW_k$ | $[g_e\odot(x_jV)]U$ | $B_ex_j$ |
| Conceptual topology traversals | $K$ | 1 | 1 |
| Edge aggregation compute | $O(KEF)$ | $O(Er)$ | $O(EF_{in}F_{out})$ |
| Shared feature mixing | $K$ full matrices | two low-rank projections | generated per edge |
| Full edge block stored | no | no | yes |
| Best use case | small basis or discrete types | continuous features and large graphs | very small dense state blocks |

For equal hidden width $F$ and modest $K,r$, the dynamic values per edge are:

- scalar basis: $K$;
- low-rank block: $r$;
- full block: $F^2$.

This comparison explains why a low-rank or basis representation usually remains practical at widths where full blocks do not.

---

## Weighted adjacency as the simplest edge-aware case

If an edge feature only determines a scalar coefficient,

$$
m_e=\alpha(q_e)Wh_{s_e},
$$

then define the sparse adjacency values by

$$
A_{t_e,s_e}=\alpha(q_e).
$$

The layer reduces to weighted SpMM:

$$
Y=A(XW).
$$

This is the most efficient edge-aware regime but only modulates message strength. It cannot provide edge-specific cross-feature mixing.

---

## Kernel decomposition and multiple SpMMs

A more expressive separable message is

$$
m_{ij}=\sum_{k=1}^{K}\phi_k(e_{ij})\psi_k(h_j).
$$

For linear $\psi_k(h_j)=W_kh_j$, this becomes

$$
Y=\sum_kA_kXW_k.
$$

This algebraic decomposition is the bridge between continuous edge features and sparse aggregation. Its practical value depends on keeping $K$ small. Otherwise repeated sparse work and the $K$-channel intermediate may cost more than a fused edge-centric kernel.

Locally supported bases are useful because an edge can activate only a few basis functions. SplineCNN is directly relevant here: it uses continuous B-spline kernels and attributes its kernel-size-independent computation to the local support of the B-spline basis.

---

## Fusion and memory traffic

Asymptotic arithmetic alone does not determine performance. Graph operations are commonly limited by irregular memory access, intermediate writes, atomics, launch overhead and degree imbalance.

A generic edge-aware implementation may materialise:

- gathered source features, $E\times F$;
- gathered destination features, $E\times F$;
- encoded edge coefficients;
- edge messages, $E\times F_{out}$.

A fused kernel can instead load the relevant node and edge values, compute the message, and accumulate directly into destination output. FusedMM is directly relevant because it combines the sampled dense-dense and sparse-dense stages used in message generation and aggregation. Its reported performance results apply to its tested models, hardware and implementation, so they should not be treated as universal speedups.

For the three variants:

- **scalar bases:** fusion can avoid writing a $K\times N\times F$ intermediate, although basis-specific transforms complicate the kernel;
- **low-rank blocks:** fusion is particularly natural because the edge loop only gates $r$ latent channels before reduction;
- **full BSR blocks:** fusion avoids edge-message storage, but block coefficient traffic remains large.

---

## Recommended experimental progression

### Baseline 1: weighted scalar SpMM

Use

$$
m_e=\alpha(q_e)Wh_{s_e}.
$$

This establishes the fastest restricted model and measures how much accuracy richer edge mixing actually adds.

### Baseline 2: generic gather-compute-scatter

Use the intended unrestricted message function. This establishes the expressivity and memory-cost reference.

### Candidate A: scalar basis channels

Sweep $K$, compare aggregate-first with transform-first, and test dense versus sparse basis activation.

### Candidate B: low-rank blocks

Sweep $r$, test fused and unfused implementations, and compare the same parameter budget against Variant A.

### Candidate C: full blocks

Restrict this to small block sizes and test only where a suitable BSR or custom block-edge kernel exists.

Useful measurements include:

- forward and backward wall-clock time;
- peak allocated memory;
- memory bandwidth and arithmetic intensity;
- bytes stored per edge;
- time spent encoding edges versus aggregating messages;
- sensitivity to degree distribution;
- scaling with $E$, $F$, $K$ and $r$;
- task quality at matched parameter count and matched runtime.

---

## Main conclusions

1. **General edge MLPs usually require edge-centric processing.** A single ordinary scalar adjacency cannot represent arbitrary nonlinear interactions between source nodes, destination nodes and multidimensional edge features.
2. **Edge dependence can recover SpMM structure when it is separable.** Representing messages as a sum of scalar edge bases times shared node transforms gives $\sum_kA_kXW_k$.
3. **Vertical stacking is usually better than block diagonal stacking for edge channels.** All channels share the same source-node set, so a rectangular $KN\times N$ sparse operator avoids replicating the node matrix.
4. **Full adjacency blocks are exact but expensive.** They reduce repeated index traversal but require $O(EF_{in}F_{out})$ values and compute.
5. **Implicit low-rank blocks provide the best general compromise.** They traverse each edge once and scale with $Er$, while retaining edge-dependent feature modulation.
6. **Fusion is a separate optimisation axis.** Even a good algebraic factorisation can underperform if it materialises large edge or channel intermediates.
7. **The best formulation is hardware- and graph-dependent.** Sparse operator support, block size, feature width, degree distribution and datatype should be benchmarked rather than inferred from FLOP counts alone.

A strong default candidate is therefore

$$
\boxed{
Y_i=
\left[
\sum_{e:t_e=i}
f_g(q_e)\odot(x_{s_e}V)
\right]U
}
$$

with a small rank $r$ and a fused edge gate-and-reduce implementation. It includes both node and edge features in every message without materialising a full matrix per edge.

---

## References

1. Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). *Modeling Relational Data with Graph Convolutional Networks*. ESWC. [arXiv record](https://arxiv.org/abs/1703.06103). Introduces R-GCN for multi-relational graphs and includes basis decomposition of relation-specific transformations.

2. Monti, F., Boscaini, D., Masci, J., Rodolà, E., Svoboda, J., & Bronstein, M. M. (2017). *Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs*. CVPR, 5115–5124. [CVPR open-access paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Monti_Geometric_Deep_Learning_CVPR_2017_paper.html). Relevant to learnable kernels over graph or manifold pseudo-coordinates.

3. Fey, M., Lenssen, J. E., Weichert, F., & Müller, H. (2018). *SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels*. CVPR, 869–877. [CVPR open-access paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Fey_SplineCNN_Fast_Geometric_CVPR_2018_paper.html). Uses continuous B-spline kernels with local support for spatial graph aggregation.

4. Simonovsky, M., & Komodakis, N. (2017). *Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs*. CVPR. [arXiv paper](https://arxiv.org/abs/1704.02901). Direct precedent for generating graph convolution filters conditioned on edge labels.

5. Rahman, M. K., Sujon, M. H., & Azad, A. (2021). *FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks*. IPDPS. [arXiv record](https://arxiv.org/abs/2011.06391). Relevant to fusing edge-level computation and sparse aggregation to reduce intermediates and memory traffic.

6. PyTorch Geometric. *Memory-Efficient Aggregations*. [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html). Contrasts general gather-scatter aggregation with SpMM-compatible message functions and discusses the memory implications of materialising gathered edge features.

7. SciPy. *Block Sparse Row matrix documentation*. [SciPy BSR reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html). Defines BSR storage, including fixed-size blocks and the counting of explicitly stored zeros.
