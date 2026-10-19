using SparseArrays
import ChainRulesCore

# ---------------------------------------------------------------------------
# Topology-aware sparse multiply — prevents Zygote from allocating a dense
# (N×N) or (B·N×B·N) gradient matrix for the fixed-structure adjacency.
#
# The forward pass is identical to `A * B`.  The custom rrule returns
# `NoTangent()` for `A` so Zygote never tries to materialise ∂A, while the
# gradient ∂B = Aᵀ ⋅ ∂Y flows back correctly to the learnable node features.
# ---------------------------------------------------------------------------
_topology_mul(A, B) = A * B

function ChainRulesCore.rrule(::typeof(_topology_mul), A, B)
    Y = _topology_mul(A, B)
    function _topology_mul_pullback(dY)
        Δ = ChainRulesCore.unthunk(dY)
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), transpose(A) * Δ)
    end
    return Y, _topology_mul_pullback
end

# Helper: convert any sparse matrix to CuSparseMatrixCSR (idempotent on GPU).
# blockdiag returns Int64 column pointers; CUSPARSE requires Int32 — convert.
_to_cusparse(A::SparseMatrixCSC) =
    CUDA.CUSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{Float32, Int32}(A))
_to_cusparse(A) =
    CUDA.CUSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{Float32, Int32}(SparseMatrixCSC(A)))

# Helper: convert any sparse-ish matrix to a CPU SparseMatrixCSC.
_to_cpu_sparse(A::SparseMatrixCSC) = A
_to_cpu_sparse(A)                  = SparseMatrixCSC(A)
