"""
benchmark_message_passing.jl

Compare two implementations of a single-layer GNN forward pass:
  1. GraphConvGNN      – uses GraphConv from GraphNeuralNetworks.jl (scatter-gather)
  2. AdjMatGNN (dense)  – message passing via dense adjacency matrix multiplication
  3. AdjMatGNN (sparse) – same but with a sparse adjacency matrix (SparseMatrixCSC / CuSparseMatrixCSR)

All three are benchmarked on CPU and (if available) GPU.
CUDA.@sync is used in the GPU benchmarks to ensure the device has completed
all work before the timer stops.

Usage:
    julia --project=. scripts/benchmark_message_passing.jl [staticmaps.nc output.nc]
"""

# ── Dependencies ─────────────────────────────────────────────────────────────
import Pkg
if !haskey(Pkg.project().dependencies, "BenchmarkTools")
    @info "Adding BenchmarkTools to the active environment"
    Pkg.add("BenchmarkTools")
end

using WflowRoutingGNN
using GraphNeuralNetworks
using Flux
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using SparseArrays
using Statistics
using BenchmarkTools
using MLUtils: batch
using Printf

# ── CLI args ──────────────────────────────────────────────────────────────────
const STATICMAPS = get(ARGS, 1,
    joinpath(@__DIR__, "..", "models", "sava_small", "staticmaps.nc"))
const OUTPUT_NC  = get(ARGS, 2,
    joinpath(@__DIR__, "..", "models", "sava_small", "run_default", "output.nc"))

# ── 1. Build graph with existing functionality ────────────────────────────────
@info "Building wflow graph …"
graphs, _, _, _ = build_wflow_graph(STATICMAPS, OUTPUT_NC, "river")

g0_raw  = graphs[1]
# Add self-loops so each node also aggregates from itself during message passing.
g0      = add_self_loops(g0_raw)          # COO  (default)
g0_sp   = GNNGraph(edge_index(g0)...,     # same topology, :sparse storage
                   num_nodes  = g0.num_nodes,
                   graph_type = :sparse)
n_nodes = g0.num_nodes
n_edges = g0.num_edges
@info "Graph: $n_nodes nodes, $n_edges edges (incl. $(n_nodes) self-loops)"

# Concatenated input features: (in_dim, n_nodes)
X_cpu  = vcat(g0.ndata.state, g0.ndata.forcing, g0.ndata.static)
in_dim = size(X_cpu, 1)

# ── 2. Build adjacency matrix ─────────────────────────────────────────────────
# A[tgt, src] = 1  →  aggregated neighbours of node i = (A * h')[:,i]
# i.e. (A * h')' gives (hidden, n_nodes) where column i is Σ_{j→i} h[:,j]
src_nodes, tgt_nodes = edge_index(g0_raw)
A_cpu = zeros(Float32, n_nodes, n_nodes)
for (s, t) in zip(src_nodes, tgt_nodes)
    A_cpu[t, s] = 1f0
end
# Self-loops: diagonal = 1
for i in 1:n_nodes
    A_cpu[i, i] = 1f0
end
A_sparse_cpu = sparse(tgt_nodes, src_nodes, ones(Float32, length(src_nodes)), n_nodes, n_nodes)
# Add self-loop entries to the sparse matrix
A_sparse_cpu += sparse(1:n_nodes, 1:n_nodes, ones(Float32, n_nodes), n_nodes, n_nodes)
@info "Adjacency matrix: $(n_nodes)×$(n_nodes), $(count(!iszero, A_cpu)) non-zeros (dense / sparse: $(nnz(A_sparse_cpu)))"

# ── 3. Model definitions ──────────────────────────────────────────────────────
const HIDDEN = 64

# ── 3a. GraphConv model ───────────────────────────────────────────────────────
struct GraphConvGNN
    encoder :: Dense
    conv    :: GraphConv
    decoder :: Dense
end
Flux.@layer GraphConvGNN

function (m::GraphConvGNN)(g::GNNGraph, x::AbstractMatrix)
    h = m.encoder(x)   # (hidden, n)
    h = m.conv(g, h)   # (hidden, n)  — scatter-based aggregation
    m.decoder(h)       # (1,      n)
end

# ── 3b. AdjMat model ──────────────────────────────────────────────────────────
# Replicates GraphConv aggregation:
#   out[i] = σ( W_self · h[:,i]  +  W_neigh · Σ_{j→i} h[:,j]  +  b )
# In matrix form (h is (hidden, n)):
#   neigh = (A * h')'              →  (hidden, n)
#   out   = σ( W_self*h + W_neigh*neigh + b )
struct AdjMatGNN
    encoder :: Dense
    W_self  :: AbstractMatrix{Float32}   # (hidden, hidden)
    W_neigh :: AbstractMatrix{Float32}   # (hidden, hidden)
    bias    :: AbstractVector{Float32}   # (hidden,)
    decoder :: Dense
    A       :: AbstractMatrix{Float32}   # (n_nodes, n_nodes)
end
Flux.@layer AdjMatGNN trainable=(encoder, W_self, W_neigh, bias, decoder)

function (m::AdjMatGNN)(x::AbstractMatrix)
    h     = m.encoder(x)                                        # (hidden, n)
    neigh = (m.A * h')'                                         # (hidden, n)
    out   = swish.(m.W_self * h .+ m.W_neigh * neigh .+ m.bias) # (hidden, n)
    m.decoder(out)                                              # (1, n)
end

# ── Construct instances ───────────────────────────────────────────────────────
gc_model = GraphConvGNN(
    Dense(in_dim => HIDDEN, swish),
    GraphConv(HIDDEN => HIDDEN, swish),
    Dense(HIDDEN => 1),
)

am_model = AdjMatGNN(
    Dense(in_dim => HIDDEN, swish),
    randn(Float32, HIDDEN, HIDDEN) .* Float32(sqrt(2 / HIDDEN)),
    randn(Float32, HIDDEN, HIDDEN) .* Float32(sqrt(2 / HIDDEN)),
    zeros(Float32, HIDDEN),
    Dense(HIDDEN => 1),
    A_cpu,
)

# Sparse variant — reuses AdjMatGNN; only A differs (SparseMatrixCSC)
am_sparse_model = AdjMatGNN(
    am_model.encoder,
    am_model.W_self,
    am_model.W_neigh,
    am_model.bias,
    am_model.decoder,
    A_sparse_cpu,
)

# ── 4. Sanity-check shapes and parameter counts ───────────────────────────────
out_gc    = gc_model(g0,    X_cpu)
out_gc_sp = gc_model(g0_sp, X_cpu)
out_am    = am_model(X_cpu)
out_am_sp = am_sparse_model(X_cpu)
@assert size(out_gc)    == (1, n_nodes) "GraphConv COO  output shape: $(size(out_gc))"
@assert size(out_gc_sp) == (1, n_nodes) "GraphConv SP   output shape: $(size(out_gc_sp))"
@assert size(out_am)    == (1, n_nodes) "AdjMat dense   output shape: $(size(out_am))"
@assert size(out_am_sp) == (1, n_nodes) "AdjMat sparse  output shape: $(size(out_am_sp))"
@info "Output shape: $(size(out_gc)) ✓"

n_params_gc = sum(length, Flux.trainables(gc_model))
n_params_am = sum(length, Flux.trainables(am_model))
@printf "Trainable parameters  GraphConv: %d   AdjMat (dense/sparse): %d\n" n_params_gc n_params_am
rel_diff = abs(n_params_gc - n_params_am) / max(n_params_gc, n_params_am)
if rel_diff > 0.10
    @warn "Parameter counts differ by $(round(100*rel_diff; digits=1)) % — models may not be comparable"
else
    @info "Parameter counts within 10 % ✓"
end

# ── 5. CPU benchmarks ─────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CPU BENCHMARKS")
println("="^60)

b_gc_cpu       = @benchmark $(gc_model)($(g0),    $(X_cpu)) seconds=10
b_gc_sp_cpu    = @benchmark $(gc_model)($(g0_sp), $(X_cpu)) seconds=10
b_am_cpu       = @benchmark $(am_model)($(X_cpu))           seconds=10
b_am_sp_cpu    = @benchmark $(am_sparse_model)($(X_cpu))    seconds=10

println("\nGraphConv COO GNN (CPU):")
show(stdout, MIME"text/plain"(), b_gc_cpu)
println("\n\nGraphConv sparse GNN (CPU):")
show(stdout, MIME"text/plain"(), b_gc_sp_cpu)
println("\n\nAdjMatrix dense GNN (CPU):")
show(stdout, MIME"text/plain"(), b_am_cpu)
println("\n\nAdjMatrix sparse GNN (CPU):")
show(stdout, MIME"text/plain"(), b_am_sp_cpu)

gc_cpu_ms       = median(b_gc_cpu).time    / 1e6
gc_sp_cpu_ms    = median(b_gc_sp_cpu).time / 1e6
am_cpu_ms       = median(b_am_cpu).time    / 1e6
am_sp_cpu_ms    = median(b_am_sp_cpu).time / 1e6
@printf "\n\nMedian (CPU)  GraphConv COO: %.3f ms   GraphConv sparse: %.3f ms   AdjMat dense: %.3f ms   AdjMat sparse: %.3f ms\n" gc_cpu_ms gc_sp_cpu_ms am_cpu_ms am_sp_cpu_ms
best_cpu_ms  = min(gc_cpu_ms, gc_sp_cpu_ms, am_cpu_ms, am_sp_cpu_ms)
best_cpu_lbl = ["GraphConv COO","GraphConv sparse","AdjMat dense","AdjMat sparse"][[gc_cpu_ms,gc_sp_cpu_ms,am_cpu_ms,am_sp_cpu_ms] .== best_cpu_ms][1]
@printf "Fastest on CPU: %s\n" best_cpu_lbl

# ── 6. GPU benchmarks ─────────────────────────────────────────────────────────
if CUDA.functional()
    println("\n" * "="^60)
    println("GPU BENCHMARKS")
    println("="^60)

    g0_gpu    = g0    |> Flux.gpu
    g0_sp_gpu = g0_sp |> Flux.gpu
    X_gpu     = X_cpu |> Flux.gpu
    gc_gpu    = gc_model |> Flux.gpu
    # Flux.@layer registers AdjMatGNN with Functors, so Flux.gpu recursively
    # moves every AbstractArray field (W_self, W_neigh, bias, A, Dense layers)
    # to GPU — no manual CuArray(...) calls needed.
    am_gpu = am_model |> Flux.gpu
    # Sparse GPU model: construct explicitly with CuSparseMatrixCSR since
    # Flux.gpu/Functors may not convert SparseMatrixCSC to a CUSPARSE type.
    am_sparse_gpu = AdjMatGNN(
        am_sparse_model.encoder |> Flux.gpu,
        am_sparse_model.W_self  |> cu,
        am_sparse_model.W_neigh |> cu,
        am_sparse_model.bias    |> cu,
        am_sparse_model.decoder |> Flux.gpu,
        CuSparseMatrixCSR(A_sparse_cpu),
    )

    # Warm-up — avoids measuring first-run JIT / kernel compilation
    CUDA.@sync gc_gpu(g0_gpu,    X_gpu)
    CUDA.@sync gc_gpu(g0_sp_gpu, X_gpu)
    CUDA.@sync am_gpu(X_gpu)
    CUDA.@sync am_sparse_gpu(X_gpu)

    # CUDA.@sync inside the benchmark expression ensures the GPU has finished
    # all queued work before BenchmarkTools records the elapsed time.
    b_gc_gpu       = @benchmark CUDA.@sync($(gc_gpu)($(g0_gpu),    $(X_gpu))) seconds=10
    b_gc_sp_gpu    = @benchmark CUDA.@sync($(gc_gpu)($(g0_sp_gpu), $(X_gpu))) seconds=10
    b_am_gpu       = @benchmark CUDA.@sync($(am_gpu)($(X_gpu)))               seconds=10
    b_am_sp_gpu    = @benchmark CUDA.@sync($(am_sparse_gpu)($(X_gpu)))        seconds=10

    println("\nGraphConv COO GNN (GPU):")
    show(stdout, MIME"text/plain"(), b_gc_gpu)
    println("\n\nGraphConv sparse GNN (GPU):")
    show(stdout, MIME"text/plain"(), b_gc_sp_gpu)
    println("\n\nAdjMatrix dense GNN (GPU):")
    show(stdout, MIME"text/plain"(), b_am_gpu)
    println("\n\nAdjMatrix sparse GNN (GPU):")
    show(stdout, MIME"text/plain"(), b_am_sp_gpu)

    gc_gpu_ms       = median(b_gc_gpu).time    / 1e6
    gc_sp_gpu_ms    = median(b_gc_sp_gpu).time / 1e6
    am_gpu_ms       = median(b_am_gpu).time    / 1e6
    am_sp_gpu_ms    = median(b_am_sp_gpu).time / 1e6
    @printf "\n\nMedian (GPU)  GraphConv COO: %.3f ms   GraphConv sparse: %.3f ms   AdjMat dense: %.3f ms   AdjMat sparse: %.3f ms\n" gc_gpu_ms gc_sp_gpu_ms am_gpu_ms am_sp_gpu_ms
    best_gpu_ms  = min(gc_gpu_ms, gc_sp_gpu_ms, am_gpu_ms, am_sp_gpu_ms)
    best_gpu_lbl = ["GraphConv COO","GraphConv sparse","AdjMat dense","AdjMat sparse"][[gc_gpu_ms,gc_sp_gpu_ms,am_gpu_ms,am_sp_gpu_ms] .== best_gpu_ms][1]
    @printf "Fastest on GPU: %s\n" best_gpu_lbl
    @printf "\nCPU→GPU speedup   GraphConv COO: %.1f×   GraphConv sparse: %.1f×   AdjMat dense: %.1f×   AdjMat sparse: %.1f×\n" (gc_cpu_ms/gc_gpu_ms) (gc_sp_cpu_ms/gc_sp_gpu_ms) (am_cpu_ms/am_gpu_ms) (am_sp_cpu_ms/am_sp_gpu_ms)
else
    @warn "CUDA not functional — skipping GPU benchmarks"
end

# ── 7. Summary plots ──────────────────────────────────────────────────────────
using CairoMakie

const LABELS  = ["GraphConv\nCOO", "GraphConv\nsparse", "AdjMat\ndense", "AdjMat\nsparse"]
const PALETTE = [:steelblue, :dodgerblue, :orangered, :forestgreen]

# Collect timing data.  gpu_* vectors stay Nothing when CUDA was unavailable.
cpu_medians_ms = [gc_cpu_ms, gc_sp_cpu_ms, am_cpu_ms, am_sp_cpu_ms]

have_gpu = @isdefined(gc_gpu_ms)
if have_gpu
    gpu_medians_ms = [gc_gpu_ms, gc_sp_gpu_ms, am_gpu_ms, am_sp_gpu_ms]
    cpu2gpu        = cpu_medians_ms ./ gpu_medians_ms
end

ncols = have_gpu ? 3 : 1
fig   = Figure(size = (420 * ncols, 440))

# Helper: grouped bar chart with min/max error bars, y-axis clipped to p99
function bar_panel!(ax, trials, vals_ms, ylabel, title)
    xs = 1:length(LABELS)
    barplot!(ax, xs, vals_ms; color = PALETTE)
    # error bars: 5th–95th percentile to suppress GC outliers
    for (i, (tr, med)) in enumerate(zip(trials, vals_ms))
        sorted = sort(tr.times) ./ 1e6
        lo = quantile(sorted, 0.05)
        hi = quantile(sorted, 0.95)
        rangebars!(ax, [i], [lo], [hi]; color = :black, whiskerwidth = 8, linewidth = 1.5)
    end
    p99_ms = maximum(quantile(sort(tr.times) ./ 1e6, 0.99) for tr in trials)
    ax.xticks         = (xs, LABELS)
    ax.ylabel         = ylabel
    ax.title          = title
    ax.xticklabelsize = 11
    ylims!(ax, 0, p99_ms * 1.15)
end

# CPU panel
ax_cpu = Axis(fig[1, 1])
cpu_trials = [b_gc_cpu, b_gc_sp_cpu, b_am_cpu, b_am_sp_cpu]
bar_panel!(ax_cpu, cpu_trials, cpu_medians_ms, "Time [ms]", "CPU  (median + p5/p95)")

if have_gpu
    # GPU panel
    ax_gpu = Axis(fig[1, 2])
    gpu_trials = [b_gc_gpu, b_gc_sp_gpu, b_am_gpu, b_am_sp_gpu]
    bar_panel!(ax_gpu, gpu_trials, gpu_medians_ms, "Time [ms]", "GPU  (median + p5/p95)")

    # CPU→GPU speedup panel
    ax_sp = Axis(fig[1, 3];
                 title          = "CPU→GPU speedup",
                 ylabel         = "speedup [×]",
                 xticks         = (1:4, LABELS),
                 xticklabelsize = 11)
    barplot!(ax_sp, 1:4, cpu2gpu; color = PALETTE)
    hlines!(ax_sp, [1f0]; color = :black, linestyle = :dash, linewidth = 1)
    for (i, v) in enumerate(cpu2gpu)
        text!(ax_sp, i, v + 0.05; text = @sprintf("%.1f×", v),
              align = (:center, :bottom), fontsize = 10)
    end
    ylims!(ax_sp, 0, nothing)
end

# Subtitle with graph info
Label(fig[2, 1:ncols],
      @sprintf("Graph: %d nodes, %d edges | hidden_dim = %d | %d trainable params",
               n_nodes, n_edges, HIDDEN, sum(length, Flux.trainables(gc_model)));
      fontsize = 11, tellwidth = false)

out_path = joinpath(@__DIR__, "benchmark_message_passing.png")
save(out_path, fig)
@info "Plot saved to $out_path"

# ── 8. Scaling study ──────────────────────────────────────────────────────────
# Generate random spanning-tree graphs of increasing size (same structure as a
# river network: each node drains to exactly one already-placed upstream node).
@info "Running scaling study …"

const SCALE_SIZES = [100, 300, 743, 2_000, 5_000, 10_000]
const SCALE_SECS  = 3   # seconds per benchmark per size point

function random_river_graph(n::Int)
    # Prüfer-like random tree: node i links to a random parent in 1:(i-1).
    src = collect(2:n)
    tgt = [rand(1:(i - 1)) for i in 2:n]
    add_self_loops(GNNGraph(src, tgt; num_nodes = n))
end

# Accumulate results row by row
sc_ns         = Int[]
sc_gc_cpu     = Float64[];  sc_gc_sp_cpu  = Float64[]
sc_am_d_cpu   = Float64[];  sc_am_s_cpu   = Float64[]
sc_gc_gpu     = Float64[];  sc_gc_sp_gpu  = Float64[]
sc_am_d_gpu   = Float64[];  sc_am_s_gpu   = Float64[]

for n in SCALE_SIZES
    @info "  size = $n"
    g_n    = random_river_graph(n)
    g_n_sp = GNNGraph(edge_index(g_n)...; num_nodes = n, graph_type = :sparse)
    x_n    = randn(Float32, in_dim, n)

    # Build adjacency matrices
    s_n, t_n = edge_index(g_n)
    A_n_d = zeros(Float32, n, n)
    for (si, ti) in zip(s_n, t_n); A_n_d[ti, si] = 1f0; end
    A_n_s = sparse(t_n, s_n, ones(Float32, length(s_n)), n, n)

    # Fresh model instances for this size (weights are random but consistent)
    gc_n   = GraphConvGNN(Dense(in_dim => HIDDEN, swish),
                          GraphConv(HIDDEN => HIDDEN, swish),
                          Dense(HIDDEN => 1))
    am_n_d = AdjMatGNN(gc_n.encoder,
                       randn(Float32, HIDDEN, HIDDEN) .* Float32(sqrt(2/HIDDEN)),
                       randn(Float32, HIDDEN, HIDDEN) .* Float32(sqrt(2/HIDDEN)),
                       zeros(Float32, HIDDEN),
                       gc_n.decoder, A_n_d)
    am_n_s = AdjMatGNN(am_n_d.encoder, am_n_d.W_self, am_n_d.W_neigh,
                       am_n_d.bias, am_n_d.decoder, A_n_s)

    # CPU
    push!(sc_ns, n)
    push!(sc_gc_cpu,    median(@benchmark $(gc_n)($(g_n),    $(x_n)) seconds=SCALE_SECS).time / 1e6)
    push!(sc_gc_sp_cpu, median(@benchmark $(gc_n)($(g_n_sp), $(x_n)) seconds=SCALE_SECS).time / 1e6)
    push!(sc_am_d_cpu,  median(@benchmark $(am_n_d)($(x_n))          seconds=SCALE_SECS).time / 1e6)
    push!(sc_am_s_cpu,  median(@benchmark $(am_n_s)($(x_n))          seconds=SCALE_SECS).time / 1e6)

    if CUDA.functional()
        g_n_gpu    = g_n    |> Flux.gpu
        g_n_sp_gpu = g_n_sp |> Flux.gpu
        x_n_gpu    = x_n    |> Flux.gpu
        gc_n_gpu   = gc_n   |> Flux.gpu
        # Dense AdjMat: Flux.gpu moves all array fields (incl. A_n_d)
        am_n_d_gpu = am_n_d |> Flux.gpu
        # Sparse AdjMat: construct explicitly — Functors can't convert
        # SparseMatrixCSC to a CUSPARSE type automatically
        am_n_s_gpu = AdjMatGNN(am_n_s.encoder |> Flux.gpu,
                               am_n_s.W_self   |> cu,
                               am_n_s.W_neigh  |> cu,
                               am_n_s.bias     |> cu,
                               am_n_s.decoder  |> Flux.gpu,
                               CuSparseMatrixCSR(A_n_s))

        CUDA.@sync gc_n_gpu(g_n_gpu,    x_n_gpu)
        CUDA.@sync gc_n_gpu(g_n_sp_gpu, x_n_gpu)
        CUDA.@sync am_n_d_gpu(x_n_gpu)
        CUDA.@sync am_n_s_gpu(x_n_gpu)

        push!(sc_gc_gpu,    median(@benchmark CUDA.@sync($(gc_n_gpu)($(g_n_gpu),    $(x_n_gpu))) seconds=SCALE_SECS).time / 1e6)
        push!(sc_gc_sp_gpu, median(@benchmark CUDA.@sync($(gc_n_gpu)($(g_n_sp_gpu), $(x_n_gpu))) seconds=SCALE_SECS).time / 1e6)
        push!(sc_am_d_gpu,  median(@benchmark CUDA.@sync($(am_n_d_gpu)($(x_n_gpu)))              seconds=SCALE_SECS).time / 1e6)
        push!(sc_am_s_gpu,  median(@benchmark CUDA.@sync($(am_n_s_gpu)($(x_n_gpu)))              seconds=SCALE_SECS).time / 1e6)
    end
end

# ── Scaling plots ─────────────────────────────────────────────────────────────
have_gpu_sc = !isempty(sc_gc_gpu)
fig2 = Figure(size = (620, 480))

SCALE_SERIES = [
    (sc_gc_cpu,    sc_gc_gpu,    "GraphConv COO",    :steelblue,   :solid),
    (sc_gc_sp_cpu, sc_gc_sp_gpu, "GraphConv sparse", :dodgerblue,  :dash),
    (sc_am_d_cpu,  sc_am_d_gpu,  "AdjMat dense",     :orangered,   :dot),
    (sc_am_s_cpu,  sc_am_s_gpu,  "AdjMat sparse",    :forestgreen, :dashdot),
]

# Compute y-axis limits across all data
all_cpu_vals = vcat(sc_gc_cpu, sc_gc_sp_cpu, sc_am_d_cpu, sc_am_s_cpu)
all_vals     = have_gpu_sc ? vcat(all_cpu_vals, sc_gc_gpu, sc_gc_sp_gpu, sc_am_d_gpu, sc_am_s_gpu) : all_cpu_vals
y_lo = 10.0 ^ floor(log10(minimum(all_vals)))
y_hi = 10.0 ^ ceil( log10(maximum(all_vals)))

# Nice rounded tick labels (avoid scientific notation)
function _nice_label(v::Float64)
    v >= 1   && return string(round(Int, v))
    v >= 0.1 && return string(round(v; digits=1))
    v >= 0.01 && return string(round(v; digits=2))
    return string(v)
end
_ytick_exps   = Int(log10(y_lo)):Int(log10(y_hi))
_yticks_vals  = [10.0^k for k in _ytick_exps]
_yticks_lbls  = [_nice_label(v) for v in _yticks_vals]
shared_yticks = (_yticks_vals, _yticks_lbls)

ax_sc = Axis(fig2[1, 1];
             title   = "Scaling: CPU (solid) vs GPU (dashed)",
             xlabel  = "Nodes",
             ylabel  = "Median time [ms]",
             xscale  = log10,
             yscale  = log10,
             limits  = (nothing, (y_lo, y_hi)),
             yticks  = shared_yticks)

for (cpu_vals, gpu_vals, lbl, col, _) in SCALE_SERIES
    # CPU: solid lines + filled markers
    lines!(  ax_sc, sc_ns, cpu_vals; color=col, linestyle=:solid, linewidth=2, label="$lbl (CPU)")
    scatter!(ax_sc, sc_ns, cpu_vals; color=col, markersize=7)
    # GPU: dashed lines + open markers
    if have_gpu_sc
        lines!(  ax_sc, sc_ns, gpu_vals; color=col, linestyle=:dash, linewidth=2, label="$lbl (GPU)")
        scatter!(ax_sc, sc_ns, gpu_vals; color=:white, marker=:circle, markersize=7,
                 strokecolor=col, strokewidth=1.5)
    end
end

axislegend(ax_sc; position=:lt, labelsize=9, nbanks=2)

Label(fig2[2, 1],
      @sprintf("Random spanning-tree graphs | hidden_dim = %d | each point = %d s benchmark",
               HIDDEN, SCALE_SECS);
      fontsize = 11, tellwidth = false)

scale_path = joinpath(@__DIR__, "benchmark_scaling.png")
save(scale_path, fig2)
@info "Scaling plot saved to $scale_path"

# ── 9. Batched aggregation benchmark ─────────────────────────────────────────
# Compare three strategies for the neighbour-aggregation step when the GNN is
# called with a batched GNNGraph (B graphs stacked, giving B·N feature columns):
#
#   a) Dense reshape      – treat A as dense (N×N), reshape h to (N, H·B), matmul,
#                           reshape back.  Uses BLAS / cuBLAS.
#   b) Sparse reshape     – same reshape trick but A stays sparse (SparseMatrixCSC /
#                           CuSparseMatrixCSR).  Uses SparseArrays / cuSPARSE.
#                           This is what SparseConv uses today.
#   c) Propagate COO      – GNNGraph.propagate scatter over batched COO topology.
#   d) Propagate sparse   – same but GNNGraph built with graph_type=:sparse
#                           (CSC internally, triggers a different propagate path).
#
# All strategies are wrapped in identical encoder/decoder Dense layers so the
# total work (excl. aggregation) is equal.
# ─────────────────────────────────────────────────────────────────────────────
@info "Running batched aggregation benchmark …"

const BATCH_SIZES  = [1, 2, 4, 8, 16, 32]
const BATCH_SECS   = 5

# Helper: run the four aggregation kernels and return median times [ms]
# g_single_sp_cpu must always be a CPU graph: GNNGraphs.batch() calls blockdiag
# which requires SparseMatrixCSC and is not supported on GPU.  COO batching
# works on any device (just index arithmetic), so g_single_coo may be on GPU.
function bench_batched_agg(A_d, A_s, A_s_cpu, g_single_coo, g_single_sp_cpu, x_single,
                            batch_sizes, device_label; on_gpu=false)
    results = Dict{String, Vector{Float64}}(
        "dense_reshape"     => Float64[],
        "sparse_reshape"    => Float64[],
        "blockdiag_precomp" => Float64[],
        "blockdiag_sparse"  => Float64[],
        "propagate_coo"     => Float64[],
        "propagate_sp"      => Float64[],
    )

    for B in batch_sizes
        X_batch = repeat(x_single, 1, B)   # (H, N·B)  — inherits device of x_single
        H, N_total = size(X_batch)
        N_per = size(A_d, 1)

        # Closures — assigned to variables so @benchmark can interpolate them
        _agg_dense = let Ad=A_d, Hv=H, Nv=N_per, Bv=B, Nt=N_total
            X -> begin
                h2     = reshape(permutedims(reshape(X, Hv, Nv, Bv), (1, 3, 2)), Hv * Bv, Nv)
                neigh2 = h2 * Ad'
                reshape(permutedims(reshape(neigh2, Hv, Bv, Nv), (1, 3, 2)), Hv, Nt)
            end
        end
        _agg_sparse = let As=A_s, Hv=H, Nv=N_per, Bv=B, Nt=N_total
            X -> begin
                h2     = reshape(permutedims(reshape(X, Hv, Nv, Bv), (1, 3, 2)), Hv * Bv, Nv)
                neigh2 = h2 * As'
                reshape(permutedims(reshape(neigh2, Hv, Bv, Nv), (1, 3, 2)), Hv, Nt)
            end
        end
        # GNN.jl built-in batching included in the timing:
        # COO: batch() works on any device — edge indices are plain arrays.
        _agg_prop_coo = let g=g_single_coo, Bv=B
            X -> propagate((xi, xj, e) -> xj, batch(fill(g, Bv)), +; xj = X)
        end
        # Sparse: blockdiag requires CPU; for GPU we add device transfer inside.
        if on_gpu
            _agg_prop_sp = let g=g_single_sp_cpu, Bv=B
                X -> propagate((xi, xj, e) -> xj, batch(fill(g, Bv)) |> Flux.gpu, +; xj = X)
            end
            # Block-diagonal (build every call) — measures construction + SpMM.
            # This is the PyG batching approach.  blockdiag needs CPU SparseMatrixCSC;
            # for GPU we then convert the result to CuSparseMatrixCSR.
            _agg_blockdiag = let As=A_s_cpu, Bv=B
                X -> begin
                    A_blk = CuSparseMatrixCSR(blockdiag(fill(As, Bv)...))
                    (A_blk * X')'
                end
            end
            # Block-diagonal (pre-computed) — A_blk built once here, measures SpMM only.
            # This is what you'd do in practice with a static graph and fixed batch size.
            A_blk_precomp = CuSparseMatrixCSR(blockdiag(fill(A_s_cpu, B)...))
            _agg_blockdiag_precomp = let A_blk=A_blk_precomp
                X -> (A_blk * X')'
            end
        else
            _agg_prop_sp = let g=g_single_sp_cpu, Bv=B
                X -> propagate((xi, xj, e) -> xj, batch(fill(g, Bv)), +; xj = X)
            end
            _agg_blockdiag = let As=A_s_cpu, Bv=B
                X -> begin
                    A_blk = blockdiag(fill(As, Bv)...)
                    (A_blk * X')'
                end
            end
            A_blk_precomp = blockdiag(fill(A_s_cpu, B)...)
            _agg_blockdiag_precomp = let A_blk=A_blk_precomp
                X -> (A_blk * X')'
            end
        end

        if on_gpu
            CUDA.@sync _agg_dense(X_batch)
            CUDA.@sync _agg_sparse(X_batch)
            CUDA.@sync _agg_blockdiag_precomp(X_batch)
            CUDA.@sync _agg_blockdiag(X_batch)
            CUDA.@sync _agg_prop_coo(X_batch)
            CUDA.@sync _agg_prop_sp(X_batch)
            bd   = median(@benchmark CUDA.@sync($(_agg_dense)($(X_batch)))              seconds=BATCH_SECS).time / 1e6
            bs   = median(@benchmark CUDA.@sync($(_agg_sparse)($(X_batch)))             seconds=BATCH_SECS).time / 1e6
            bbdp = median(@benchmark CUDA.@sync($(_agg_blockdiag_precomp)($(X_batch)))  seconds=BATCH_SECS).time / 1e6
            bbd  = median(@benchmark CUDA.@sync($(_agg_blockdiag)($(X_batch)))          seconds=BATCH_SECS).time / 1e6
            bpc  = median(@benchmark CUDA.@sync($(_agg_prop_coo)($(X_batch)))           seconds=BATCH_SECS).time / 1e6
            bps  = median(@benchmark CUDA.@sync($(_agg_prop_sp)($(X_batch)))            seconds=BATCH_SECS).time / 1e6
        else
            _agg_dense(X_batch); _agg_sparse(X_batch)
            _agg_blockdiag_precomp(X_batch); _agg_blockdiag(X_batch)
            _agg_prop_coo(X_batch); _agg_prop_sp(X_batch)  # warm-up
            bd   = median(@benchmark $(_agg_dense)($(X_batch))             seconds=BATCH_SECS).time / 1e6
            bs   = median(@benchmark $(_agg_sparse)($(X_batch))            seconds=BATCH_SECS).time / 1e6
            bbdp = median(@benchmark $(_agg_blockdiag_precomp)($(X_batch)) seconds=BATCH_SECS).time / 1e6
            bbd  = median(@benchmark $(_agg_blockdiag)($(X_batch))         seconds=BATCH_SECS).time / 1e6
            bpc  = median(@benchmark $(_agg_prop_coo)($(X_batch))          seconds=BATCH_SECS).time / 1e6
            bps  = median(@benchmark $(_agg_prop_sp)($(X_batch))           seconds=BATCH_SECS).time / 1e6
        end
        push!(results["dense_reshape"],     bd)
        push!(results["sparse_reshape"],    bs)
        push!(results["blockdiag_precomp"], bbdp)
        push!(results["blockdiag_sparse"],  bbd)
        push!(results["propagate_coo"],     bpc)
        push!(results["propagate_sp"],      bps)
        @info @sprintf("  %s B=%2d  dense=%.3f ms  sparse=%.3f ms  blkdiag_pre=%.3f ms  blkdiag=%.3f ms  prop_coo=%.3f ms  prop_sp=%.3f ms",
                        device_label, B, bd, bs, bbdp, bbd, bpc, bps)
    end
    results
end

# Encoder hidden representation used as surrogate input for the aggregation kernel
X_hidden_cpu = randn(Float32, HIDDEN, n_nodes)
A_dense_cpu  = Matrix(A_sparse_cpu)   # dense copy
# g0_sp already built at top of script (graph_type=:sparse)

# ── 5000-node random river graph (same generator as scaling study) ────────────
const N_LARGE = 5_000
@info "Building 5000-node random graph for batched benchmark …"
g_large_coo = random_river_graph(N_LARGE)
g_large_sp  = GNNGraph(edge_index(g_large_coo)...; num_nodes=N_LARGE, graph_type=:sparse)
src_lg, tgt_lg = edge_index(g_large_coo)
A_sparse_large_cpu = sparse(tgt_lg, src_lg, ones(Float32, length(src_lg)), N_LARGE, N_LARGE)
A_dense_large_cpu  = Matrix(A_sparse_large_cpu)
X_hidden_large_cpu = randn(Float32, HIDDEN, N_LARGE)

# ── Run both graph sizes ──────────────────────────────────────────────────────
GRAPH_CONFIGS = [
    (A_dense_cpu,       A_sparse_cpu,       g0,       g0_sp,      X_hidden_cpu,       n_nodes,  "n=$(n_nodes)"),
    (A_dense_large_cpu, A_sparse_large_cpu, g_large_coo, g_large_sp, X_hidden_large_cpu, N_LARGE, "n=$(N_LARGE)"),
]

all_cpu_results = Dict{String,Any}[]
all_gpu_results = Union{Dict{String,Any},Nothing}[]

for (Ad, As, gcoo, gsp, Xh, nn, lbl) in GRAPH_CONFIGS
    @info "Batched aggregation benchmark — $lbl …"
    push!(all_cpu_results, bench_batched_agg(Ad, As, As, gcoo, gsp, Xh, BATCH_SIZES, "CPU $lbl"))
    if CUDA.functional()
        Ad_gpu   = Ad  |> Flux.gpu
        As_gpu   = CuSparseMatrixCSR(As)
        gcoo_gpu = gcoo |> Flux.gpu
        # gsp and As stay as CPU — blockdiag/batch() need SparseMatrixCSC (CPU only)
        Xh_gpu   = Xh  |> Flux.gpu
        push!(all_gpu_results,
              bench_batched_agg(Ad_gpu, As_gpu, As, gcoo_gpu, gsp, Xh_gpu, BATCH_SIZES, "GPU $lbl"; on_gpu=true))
    else
        push!(all_gpu_results, nothing)
    end
end

# ── Plot ──────────────────────────────────────────────────────────────────────
have_gpu_batch = any(!isnothing, all_gpu_results)
nrows3 = have_gpu_batch ? 2 : 1
ncols3 = length(GRAPH_CONFIGS)
fig3 = Figure(size = (700 * ncols3, 350 * nrows3))

BATCH_SERIES = [
    ("dense_reshape",     "Dense reshape",              :orangered,   :solid),
    ("sparse_reshape",    "Sparse reshape",             :forestgreen, :dash),
    ("blockdiag_precomp", "Block-diag precomp (PyG★)",  :sienna,      :solid),
    ("blockdiag_sparse",  "Block-diag build+run (PyG)", :sienna,      :dash),
    ("propagate_coo",     "Propagate COO",              :steelblue,   :dot),
    ("propagate_sp",      "Propagate sparse",           :purple,      :dashdot),
]

# Shared log y-axis limits across all panels
_all_batch_vals = vcat(
    [vcat(values(r)...) for r in all_cpu_results]...,
    [vcat(values(r)...) for r in all_gpu_results if !isnothing(r)]...,
)
_b_y_lo = 10.0 ^ floor(log10(minimum(_all_batch_vals)))
_b_y_hi = 10.0 ^ ceil( log10(maximum(_all_batch_vals)))
_b_ytick_vals = [10.0^k for k in Int(log10(_b_y_lo)):Int(log10(_b_y_hi))]
_b_ytick_lbls = [_nice_label(v) for v in _b_ytick_vals]
_b_yticks = (_b_ytick_vals, _b_ytick_lbls)

for (ci, (_, _, _, _, _, nn, lbl)) in enumerate(GRAPH_CONFIGS)
    cpu_res = all_cpu_results[ci]
    gpu_res = all_gpu_results[ci]

    for (ri, (res, dev_lbl)) in enumerate(
            filter(x -> !isnothing(x[1]),
                   [(cpu_res, "CPU"), (gpu_res, "GPU")]))
        ax = Axis(fig3[ri, ci];
                  title   = "$dev_lbl — $lbl",
                  xlabel  = "Batch size B",
                  ylabel  = "Median time [ms]",
                  yscale  = log10,
                  limits  = (nothing, (_b_y_lo, _b_y_hi)),
                  yticks  = _b_yticks,
                  xticks  = (1:length(BATCH_SIZES), string.(BATCH_SIZES)))
        for (key, series_lbl, col, ls) in BATCH_SERIES
            vals = res[key]
            lines!(  ax, 1:length(BATCH_SIZES), vals; color=col, linestyle=ls, label=series_lbl, linewidth=2)
            scatter!(ax, 1:length(BATCH_SIZES), vals; color=col, markersize=7)
        end
        axislegend(ax; position=:lt, labelsize=10)
    end
end

Label(fig3[nrows3+1, 1:ncols3],
      @sprintf("hidden_dim = %d | propagate+blockdiag include build cost; reshape strategies: kernel only | each point = %d s",
               HIDDEN, BATCH_SECS);
      fontsize=11, tellwidth=false)

batch_path = joinpath(@__DIR__, "benchmark_batched_agg.png")
save(batch_path, fig3)
@info "Batched aggregation plot saved to $batch_path"

@info "Done."
