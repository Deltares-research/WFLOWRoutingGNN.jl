using Flux
using Functors
using GraphNeuralNetworks
using SparseArrays
import TOML

# `_topology_mul` (+ its rrule) and the `_to_cusparse`/`_to_cpu_sparse` helpers
# live in sparse_ops.jl; the MassBalanceLayer lives in massbalance.jl (both are
# included before this file).

# Registry of serialisable activation function names.
const ACTIVATIONS = Dict(
    "swish"    => swish,
    "relu"     => relu,
    "tanh"     => tanh,
    "sigmoid"  => sigmoid,
    "identity" => identity,
    "selu"     => selu,
    "elu"      => elu,
    "gelu"     => gelu,
    "softplus" => softplus,
)

# Return the registry name for an activation function, or throw.
function _activation_name(f)
    for (name, fn) in ACTIVATIONS
        fn === f && return name
    end
    throw(ArgumentError("activation $f is not in the ACTIVATIONS registry; register it first"))
end

"""
    ModelSettings

Hyperparameters for a `WflowGNN` model.

- `domain`          : routing domain; must be a key of `DOMAIN_VARS`.
- `hidden_dim`      : width of the hidden representation (default `64`).
- `nlayers`         : number of message-passing layers in the processor (default `3`).
- `mlp_layers`      : number of `Dense` layers in the encoder and decoder MLPs (default `1`).
- `enc_activation`  : activation for encoder `Dense` layers (default `swish`).
- `proc_activation` : activation for each processor layer (default `swish`).
- `enforce_mass_balance` : for the `"river"` domain, enforce the kinematic-wave
                   mass balance as a hard constraint (decoder predicts only `Δq`;
                   `river_h` is derived analytically). When `false`, `river_q`
                   and `river_h` are predicted independently by the decoder
                   (default `true`). Ignored for non-river domains.
- `mb_theta`        : implicitness weight `θ ∈ [0, 1]` of the mass-balance time
                   discretisation (default `1.0` = fully implicit / backward
                   Euler). `0.5` = Crank–Nicolson, `0.0` = fully explicit.
                   Lowering θ damps the stiff `q→h` gradient (`∂h_norm/∂q_norm =
                   −θ·dt·σ_q/σ_h`) that drives rollout h-overshoots. Only used
                   when `enforce_mass_balance` is `true`.
- `mb_augment_decoder` : feed two physics-derived features (the summed upstream
                   discharge at `t` and a fully-explicit forward-Euler estimate
                   of the next-step depth) as extra decoder inputs. Both are
                   known at `t` (non-circular) and derived from the mass-balance
                   layer's routing/constants; the hard mass-balance output is
                   unchanged (exact). Only used when the mass balance is active
                   (`river` domain with `enforce_mass_balance = true`); ignored
                   otherwise (default `false`).
- `include_log_upstream_area` : append one optional synthetic static feature,
                   `log1p(upstream_area)`, to the model input (river domain).
                   Useful for better downstream conditioning. Must match the
                   preprocessing setting used when building graph features
                   (default `false`).
- `mb_smooth_h_floor` : when `true`, apply a smooth non-negative floor
                   (softplus-style) to `h_phys_new` in the mass-balance update
                   instead of a hard `max(0, ·)` clamp. Keeps `h >= 0` while
                   preserving a gradient on near-dry branches (default `false`).
- `mb_h_floor_softness` : softness (in physical meters) of the smooth h-floor;
                   larger values are smoother/less hard-clamped. Only used when
                   `mb_smooth_h_floor = true` (default `0.05`).
- `mb_sigma_h_floor` : lower bound for per-node `σ_h` during preprocessing
                   (train-split fit), used to prevent tiny-depth-variance nodes
                   from dominating the h-loss (default `0.0` = disabled).
- `mb_sigma_q_floor` : lower bound for per-node `σ_q` during preprocessing
                   (default `0.0` = disabled).
- `mb_sigma_inwater_floor` : lower bound for per-node `σ_inwater` during
                   preprocessing (default `0.0` = disabled).
"""
Base.@kwdef struct ModelSettings
    domain               :: String
    hidden_dim           :: Int = 64
    nlayers              :: Int = 3
    mlp_layers           :: Int = 1
    enc_activation              = swish
    proc_activation             = swish
    enforce_mass_balance :: Bool = true
    mb_theta             :: Float32 = 1.0f0
    mb_augment_decoder   :: Bool = false
    include_log_upstream_area :: Bool = false
    mb_smooth_h_floor    :: Bool = false
    mb_h_floor_softness  :: Float32 = 0.05f0
    mb_sigma_h_floor     :: Float32 = 0.0f0
    mb_sigma_q_floor     :: Float32 = 0.0f0
    mb_sigma_inwater_floor :: Float32 = 0.0f0
end

function Base.show(io::IO, s::ModelSettings)
    println(io, "ModelSettings:")
    println(io, "  domain          : ", s.domain)
    println(io, "  hidden_dim      : ", s.hidden_dim)
    println(io, "  nlayers         : ", s.nlayers)
    println(io, "  mlp_layers      : ", s.mlp_layers)
    println(io, "  enc_activation  : ", _activation_name(s.enc_activation))
    println(io, "  proc_activation : ", _activation_name(s.proc_activation))
    println(io, "  enforce_mass_balance : ", s.enforce_mass_balance)
    println(io, "  mb_theta        : ", s.mb_theta)
    println(io, "  mb_augment_decoder : ", s.mb_augment_decoder)
    println(io, "  include_log_upstream_area : ", s.include_log_upstream_area)
    println(io, "  mb_smooth_h_floor : ", s.mb_smooth_h_floor)
    println(io, "  mb_h_floor_softness : ", s.mb_h_floor_softness)
    println(io, "  mb_sigma_h_floor : ", s.mb_sigma_h_floor)
    println(io, "  mb_sigma_q_floor : ", s.mb_sigma_q_floor)
    print(  io, "  mb_sigma_inwater_floor : ", s.mb_sigma_inwater_floor)
end

"""
    save_model_settings(path, settings)

Write `settings` to a TOML file at `path`.
Activation functions are stored by their registered name (see `ACTIVATIONS`).
"""
function save_model_settings(path::String, s::ModelSettings)
    dict = Dict(
        "domain"          => s.domain,
        "hidden_dim"      => s.hidden_dim,
        "nlayers"         => s.nlayers,
        "mlp_layers"      => s.mlp_layers,
        "enc_activation"  => _activation_name(s.enc_activation),
        "proc_activation" => _activation_name(s.proc_activation),
        "enforce_mass_balance" => s.enforce_mass_balance,
        "mb_theta"        => s.mb_theta,
        "mb_augment_decoder" => s.mb_augment_decoder,
        "include_log_upstream_area" => s.include_log_upstream_area,
        "mb_smooth_h_floor" => s.mb_smooth_h_floor,
        "mb_h_floor_softness" => Float32(s.mb_h_floor_softness),
        "mb_sigma_h_floor" => Float32(s.mb_sigma_h_floor),
        "mb_sigma_q_floor" => Float32(s.mb_sigma_q_floor),
        "mb_sigma_inwater_floor" => Float32(s.mb_sigma_inwater_floor),
    )
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    load_model_settings(path) -> ModelSettings

Read a `ModelSettings` from the TOML file at `path`.
"""
function load_model_settings(path::String)
    d = TOML.parsefile(path)
    enc_name  = get(d, "enc_activation",  "swish")
    proc_name = get(d, "proc_activation", "swish")
    enc_name  in keys(ACTIVATIONS) || throw(ArgumentError("unknown enc_activation \"$enc_name\"; choose from $(sort(collect(keys(ACTIVATIONS))))"))
    proc_name in keys(ACTIVATIONS) || throw(ArgumentError("unknown proc_activation \"$proc_name\"; choose from $(sort(collect(keys(ACTIVATIONS))))"))
    return ModelSettings(
        domain          = d["domain"],
        hidden_dim      = get(d, "hidden_dim", 64),
        nlayers         = get(d, "nlayers",    3),
        mlp_layers      = get(d, "mlp_layers", 1),
        enc_activation  = ACTIVATIONS[enc_name],
        proc_activation = ACTIVATIONS[proc_name],
        enforce_mass_balance = get(d, "enforce_mass_balance", true),
        mb_theta        = Float32(get(d, "mb_theta", 1.0)),
        mb_augment_decoder = get(d, "mb_augment_decoder", false),
        include_log_upstream_area = get(d, "include_log_upstream_area", false),
        mb_smooth_h_floor = get(d, "mb_smooth_h_floor", false),
        mb_h_floor_softness = Float32(get(d, "mb_h_floor_softness", 0.05)),
        mb_sigma_h_floor = Float32(get(d, "mb_sigma_h_floor", 0.0)),
        mb_sigma_q_floor = Float32(get(d, "mb_sigma_q_floor", 0.0)),
        mb_sigma_inwater_floor = Float32(get(d, "mb_sigma_inwater_floor", 0.0)),
    )
end

"""
    SparseConv(in_dim => out_dim, σ = identity; A)

A graph convolution layer that uses a pre-stored (sparse) adjacency matrix `A`
for neighbour aggregation instead of scatter operations over a `GNNGraph`.

Equivalent to `GraphConv` with mean aggregation, but the topology is fixed at
construction time, which allows BLAS/CUSPARSE matrix multiplications and avoids
the overhead of graph scatter-gather at every forward pass.

The layer is compatible with `GNNChain`: it accepts `(g::GNNGraph, h)` and
returns `h_new`, ignoring `g` at runtime.

`A` must be a `(n_nodes, n_nodes)` (sparse or dense) `Float32` matrix where
`A[i, j] = 1` means node `j` contributes to the aggregated neighbourhood of
node `i`.  Self-loops should be included explicitly if desired.

`A` is **not** a trainable parameter. It is stored as a CPU `SparseMatrixCSC`
and is converted to the appropriate device format by `Flux.gpu` / `Flux.cpu`.

### Batching strategies

**Single graph** (`size(h,2) == n_nodes`): direct `(N×N)` SpMM.

**Batched graph** (`size(h,2) == B * n_nodes`): two paths depending on whether
a pre-computed block-diagonal has been stored (via `precompute_batched`):

- *With precomputed block-diagonal* (`A_batched !== nothing`, `batch_size == B`):
  a single `(B·N × B·N)` SpMM on the pre-stored block-diagonal matrix.
  ~2× faster than the reshape trick on GPU (benchmarked).

- *Without precomputed block-diagonal* (fallback, good for CPU):
  reshape trick — `h` is reshaped to `(H·B, N)`, multiplied by `A'` (the
  `N×N` single-graph matrix), then reshaped back.  Avoids materialising a
  larger matrix and is the fastest CPU strategy.

Use `precompute_batched(model, B)` to enable the fast GPU batching path.
"""
struct SparseConv{M <: AbstractMatrix{Float32}, V <: AbstractVector{Float32}, F} <: GNNLayer
    W_self     :: M
    W_neigh    :: M
    bias       :: V
    σ          :: F
    A          :: AbstractMatrix{Float32}                  # single-graph (N×N); device-appropriate
    A_batched  :: Union{Nothing, AbstractMatrix{Float32}}  # block-diagonal (B·N × B·N), or nothing
    batch_size :: Int                                       # B for A_batched; 0 = no precomputation
end

Flux.@layer SparseConv trainable=(W_self, W_neigh, bias)
# Restrict Functors traversal to trainable fields only so Optimisers.setup
# never sees A / A_batched (not writable in-place on GPU).  Device transfer is
# handled by the explicit Flux.gpu / Flux.cpu overloads below.
Functors.@functor SparseConv (W_self, W_neigh, bias)

# Move trainable weights to GPU and convert A / A_batched to CuSparseMatrixCSR
# for fast CUSPARSE SpMM.  Called by Flux.gpu(model).
function Flux.gpu(l::SparseConv)
    SparseConv(
        Flux.gpu(l.W_self),
        Flux.gpu(l.W_neigh),
        Flux.gpu(l.bias),
        l.σ,
        _to_cusparse(l.A),
        isnothing(l.A_batched) ? nothing : _to_cusparse(l.A_batched),
        l.batch_size,
    )
end

# Move trainable weights to CPU and convert A / A_batched back to SparseMatrixCSC.
# Called by Flux.cpu(model).
function Flux.cpu(l::SparseConv)
    SparseConv(
        Flux.cpu(l.W_self),
        Flux.cpu(l.W_neigh),
        Flux.cpu(l.bias),
        l.σ,
        _to_cpu_sparse(l.A),
        isnothing(l.A_batched) ? nothing : _to_cpu_sparse(l.A_batched),
        l.batch_size,
    )
end

"""
    SparseConv(ch::Pair{Int,Int}, σ = identity; A)

Construct a `SparseConv` layer mapping `ch.first`-dimensional inputs to
`ch.second`-dimensional outputs.  `W_self` and `W_neigh` are initialised with
Glorot uniform; `bias` is zero-initialised.  `A_batched` is `nothing`; call
`precompute_batched` to enable the fast block-diagonal GPU batching path.
"""
function SparseConv(ch::Pair{Int,Int}, σ = identity; A::AbstractMatrix{Float32})
    in_d, out_d = ch
    glorot(a, b) = Float32.(Flux.glorot_uniform(a, b))
    SparseConv(
        glorot(out_d, in_d),
        glorot(out_d, in_d),
        zeros(Float32, out_d),
        σ,
        A,
        nothing,  # A_batched — set via precompute_batched
        0,        # batch_size
    )
end

"""
    (l::SparseConv)(g::GNNGraph, h::AbstractMatrix{Float32}) -> AbstractMatrix

Forward pass via sparse matrix multiply.  `g` is accepted for `GNNChain`
compatibility but is not used; topology comes from `l.A` / `l.A_batched`.

Three dispatch paths (in priority order):

1. `size(h,2) == size(l.A,1)` — **single graph**: `neigh = (A * h')'`
2. `size(h,2) == l.batch_size * size(l.A,1)` and `l.A_batched !== nothing` —
   **batched, precomputed block-diagonal**: `neigh = (A_batched * h')'`.
   Fastest on GPU (~2× vs reshape trick).
3. Otherwise — **batched, reshape fallback** (fastest on CPU):
   reshapes `h` to `(H·B, N)`, multiplies by `A'`, reshapes back.
"""
function (l::SparseConv)(::GNNGraph, h::AbstractMatrix{Float32})
    H, N_total = size(h)
    N_per = size(l.A, 1)
    if N_total == N_per
        # Single graph: direct (N×N) SpMM.
        neigh = (_topology_mul(l.A, h'))'
    elseif !isnothing(l.A_batched) && N_total == l.batch_size * N_per
        # Batched: single (B·N × B·N) SpMM on the pre-stored block-diagonal.
        neigh = (_topology_mul(l.A_batched, h'))'
    else
        # Batched fallback: reshape trick — no block-diagonal materialisation.
        #   h (H, N·B) → (H·B, N) → * A' → (H·B, N) → (H, N·B)
        B      = N_total ÷ N_per
        h2     = reshape(permutedims(reshape(h, H, N_per, B), (1, 3, 2)), H * B, N_per)
        neigh2 = (_topology_mul(l.A, h2'))'
        neigh  = reshape(permutedims(reshape(neigh2, H, B, N_per), (1, 3, 2)), H, N_total)
    end
    l.σ.(l.W_self * h .+ l.W_neigh * neigh .+ l.bias) .+ h
end

"""
    precompute_batched(layer::SparseConv, batch_size::Int) -> SparseConv

Return a new `SparseConv` with a pre-computed block-diagonal adjacency matrix
`A_batched = blockdiag(A, A, ..., A)` (`batch_size` copies).

`A_batched` is stored as a CPU `SparseMatrixCSC`; call `Flux.gpu` afterwards to
convert it to `CuSparseMatrixCSR` for GPU training.  Always call this **before**
`Flux.gpu` — building the block-diagonal requires a CPU sparse matrix.

The block-diagonal path is selected automatically in the forward pass when
`size(h, 2) == batch_size * n_nodes`.
"""
function precompute_batched(l::SparseConv, B::Int)
    A_cpu = _to_cpu_sparse(l.A)
    # blockdiag may return Int64 column pointers; normalise to Int32 so
    # CuSparseMatrixCSR conversion is always valid without further copies.
    A_blk = SparseMatrixCSC{Float32, Int32}(blockdiag(fill(A_cpu, B)...))
    SparseConv(l.W_self, l.W_neigh, l.bias, l.σ, A_cpu, A_blk, B)
end

"""
    WflowGNN

Encode-process-decode GNN for wflow routing emulation.

- `encoder`      : `Dense` or `Chain` of `Dense` layers mapping `in_dim -> hidden_dim`.
- `processor`    : `GNNChain` of `GraphConv` or `SparseConv` layers operating at `hidden_dim`.
- `decoder`      : `Dense` or `Chain` of `Dense` layers mapping `hidden_dim -> out_dim`.
- `mass_balance` : optional `MassBalanceLayer` that hard-constrains `river_h` via the
                   kinematic-wave mass balance. When present the decoder outputs only
                   `Δq` (`out_dim = 1`) and `river_h` is computed analytically.
- `augment_mb`   : when `true` (and `mass_balance` is present), two physics-derived
                   features (`mb_decoder_features`) are appended to the decoder
                   input, widening its first layer by `MB_DECODER_FEATURES`.
"""
struct WflowGNN
    encoder      :: Union{Dense, Chain}
    processor    :: GNNChain
    decoder      :: Union{Dense, Chain}
    mass_balance :: Union{Nothing, MassBalanceLayer}
    augment_mb   :: Bool
end

Flux.@layer WflowGNN trainable=(encoder, processor, decoder, mass_balance)

# Explicit device overloads for WflowGNN so that SparseConv.A and
# SparseConv.A_batched are converted correctly (Functors traversal only
# reaches the trainable fields declared in @functor SparseConv).
function Flux.gpu(m::WflowGNN)
    WflowGNN(
        Flux.gpu(m.encoder),
        GNNChain(map(Flux.gpu, m.processor.layers)...),
        Flux.gpu(m.decoder),
        isnothing(m.mass_balance) ? nothing : Flux.gpu(m.mass_balance),
        m.augment_mb,
    )
end

function Flux.cpu(m::WflowGNN)
    WflowGNN(
        Flux.cpu(m.encoder),
        GNNChain(map(Flux.cpu, m.processor.layers)...),
        Flux.cpu(m.decoder),
        isnothing(m.mass_balance) ? nothing : Flux.cpu(m.mass_balance),
        m.augment_mb,
    )
end

"""
    WflowGNN(settings::ModelSettings)

Construct a `WflowGNN` from a `ModelSettings`.

The input dimension is the total number of state, forcing, and static variables
for `settings.domain` (from `DOMAIN_VARS`). The output dimension equals the
number of state variables.
"""
function WflowGNN(s::ModelSettings)
    s.domain in keys(DOMAIN_VARS) ||
        throw(ArgumentError("domain must be one of $(join(sort(collect(keys(DOMAIN_VARS))), ", ")), got \"$(s.domain)\""))
    vars    = DOMAIN_VARS[s.domain]
    extra_static = (s.domain == "river" && s.include_log_upstream_area) ? 1 : 0
    in_dim  = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"]) + extra_static
    out_dim = length(vars["state"])
    return WflowGNN(in_dim, s.hidden_dim, out_dim;
                    nlayers         = s.nlayers,
                    mlp_layers      = s.mlp_layers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation)
end

"""
    WflowGNN(settings::ModelSettings, A::AbstractMatrix{Float32})

Construct a `WflowGNN` that uses `SparseConv` layers instead of `GraphConv`.
The processor performs neighbour aggregation via the pre-stored (sparse) adjacency
matrix `A` rather than GNNGraph scatter operations.  All other behaviour is
identical to `WflowGNN(settings)`.

`A` should be a `(n_nodes, n_nodes)` `Float32` matrix (dense or `SparseMatrixCSC`)
with `A[i, j] = 1` indicating that node `j` is an upstream neighbour of node `i`.
Self-loops must be included explicitly if desired.

`A` is not trainable; it is moved to GPU automatically via `Flux.gpu`.
"""
function WflowGNN(s::ModelSettings, A::AbstractMatrix{Float32})
    s.domain in keys(DOMAIN_VARS) ||
        throw(ArgumentError("domain must be one of $(join(sort(collect(keys(DOMAIN_VARS))), ", ")), got \"$(s.domain)\""))
    vars    = DOMAIN_VARS[s.domain]
    extra_static = (s.domain == "river" && s.include_log_upstream_area) ? 1 : 0
    in_dim  = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"]) + extra_static
    out_dim = length(vars["state"])
    return WflowGNN(in_dim, s.hidden_dim, out_dim;
                    nlayers         = s.nlayers,
                    mlp_layers      = s.mlp_layers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation,
                    adj_matrix      = A)
end

"""
    WflowGNN(settings, mass_balance) -> WflowGNN

Construct a `WflowGNN` with a hard mass-balance constraint for the river domain.
The decoder outputs only `Δq` (`out_dim = 1`); `river_h` is derived analytically
in the forward pass via `mass_balance`.
"""
function WflowGNN(s::ModelSettings, mb::MassBalanceLayer)
    s.domain in keys(DOMAIN_VARS) ||
        throw(ArgumentError("domain must be one of $(join(sort(collect(keys(DOMAIN_VARS))), ", ")), got \"$(s.domain)\""))
    vars   = DOMAIN_VARS[s.domain]
    extra_static = (s.domain == "river" && s.include_log_upstream_area) ? 1 : 0
    in_dim = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"]) + extra_static
    return WflowGNN(in_dim, s.hidden_dim, 1;
                    nlayers         = s.nlayers,
                    mlp_layers      = s.mlp_layers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation,
                    mass_balance    = mb,
                    augment_mb      = s.mb_augment_decoder)
end

"""
    WflowGNN(settings, mass_balance, A) -> WflowGNN

Construct a `WflowGNN` with both a hard mass-balance constraint and `SparseConv`
layers (see `WflowGNN(settings, A)` and `WflowGNN(settings, mass_balance)`).
"""
function WflowGNN(s::ModelSettings, mb::MassBalanceLayer, A::AbstractMatrix{Float32})
    s.domain in keys(DOMAIN_VARS) ||
        throw(ArgumentError("domain must be one of $(join(sort(collect(keys(DOMAIN_VARS))), ", ")), got \"$(s.domain)\""))
    vars   = DOMAIN_VARS[s.domain]
    extra_static = (s.domain == "river" && s.include_log_upstream_area) ? 1 : 0
    in_dim = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"]) + extra_static
    return WflowGNN(in_dim, s.hidden_dim, 1;
                    nlayers         = s.nlayers,
                    mlp_layers      = s.mlp_layers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation,
                    mass_balance    = mb,
                    augment_mb      = s.mb_augment_decoder,
                    adj_matrix      = A)
end

# Internal constructor — shared by all public constructors.
# Pass `adj_matrix` to use SparseConv layers instead of GraphConv.
# `mlp_layers` controls the number of Dense layers in the encoder and decoder.
function WflowGNN(
        in_dim    :: Int,
        hidden_dim:: Int,
        out_dim   :: Int;
        nlayers         :: Int = 3,
        mlp_layers      :: Int = 1,
        enc_activation        = swish,
        proc_activation       = swish,
        mass_balance          = nothing,
        augment_mb      :: Bool = false,
        adj_matrix            = nothing)

    mlp_layers >= 1 || throw(ArgumentError("mlp_layers must be >= 1, got $mlp_layers"))

    # Augmentation only applies when a mass-balance layer is present (the extra
    # features are derived from it). The resolved flag is stored on the struct.
    augment = augment_mb && !isnothing(mass_balance)
    dec_in  = hidden_dim + (augment ? MB_DECODER_FEATURES : 0)

    # Encoder: in_dim → hidden_dim → ... → hidden_dim  (mlp_layers Dense layers)
    if mlp_layers == 1
        encoder = Dense(in_dim => hidden_dim, enc_activation)
    else
        enc_ls = Any[Dense(in_dim => hidden_dim, enc_activation)]
        for _ in 2:mlp_layers
            push!(enc_ls, Dense(hidden_dim => hidden_dim, enc_activation))
        end
        encoder = Chain(enc_ls...)
    end

    if isnothing(adj_matrix)
        processor = GNNChain([GraphConv(hidden_dim => hidden_dim, proc_activation) for _ in 1:nlayers]...)
    else
        A = adj_matrix :: AbstractMatrix{Float32}
        processor = GNNChain([SparseConv(hidden_dim => hidden_dim, proc_activation; A=A) for _ in 1:nlayers]...)
    end

    # Decoder: hidden_dim → ... → hidden_dim → out_dim  (mlp_layers Dense layers)
    # When augmenting, the first decoder layer widens to accept the extra
    # physics-derived features (dec_in = hidden_dim + MB_DECODER_FEATURES).
    if mlp_layers == 1
        decoder = Dense(dec_in => out_dim)
    else
        dec_ls = Any[Dense(dec_in => hidden_dim, enc_activation)]
        for _ in 2:(mlp_layers - 1)
            push!(dec_ls, Dense(hidden_dim => hidden_dim, enc_activation))
        end
        push!(dec_ls, Dense(hidden_dim => out_dim))
        decoder = Chain(dec_ls...)
    end

    return WflowGNN(encoder, processor, decoder, mass_balance, augment)
end

"""
    precompute_batched(model::WflowGNN, batch_size::Int) -> WflowGNN

Return a new `WflowGNN` with all `SparseConv` layers augmented with a
pre-computed block-diagonal adjacency matrix for `batch_size`.

Call this **before** `Flux.gpu(model)`.  After `Flux.gpu`, both `A` and
`A_batched` are `CuSparseMatrixCSR` on the GPU, and batched forward passes
use a single `(B·N × B·N)` SpMM (~2× faster than the reshape trick on GPU).

Has no effect on `GraphConv` layers.
"""
function precompute_batched(model::WflowGNN, B::Int)
    new_layers = map(model.processor.layers) do l
        l isa SparseConv ? precompute_batched(l, B) : l
    end
    new_mb = if !isnothing(model.mass_balance)
        precompute_batched(model.mass_balance, B)
    else
        nothing
    end
    WflowGNN(model.encoder, GNNChain(new_layers...), model.decoder, new_mb, model.augment_mb)
end

"""
    (m::WflowGNN)(g, state, forcing, static)

Array-based forward pass. `g` provides the graph topology; `state`, `forcing`,
and `static` are passed directly as matrices, avoiding GNNGraph ndata in the
differentiation path.
"""
function (m::WflowGNN)(g::GNNGraph,
                       state        ::AbstractMatrix,
                       forcing      ::AbstractMatrix,
                       static       ::AbstractMatrix,
                       forcing_next ::AbstractMatrix = forcing)
    # `static` holds per-node constants for a single graph (N columns). For a
    # batched graph the dynamic inputs span B·N columns, so tile the static
    # block B times to match. The batched node ordering is block-wise
    # [graph1 | graph2 | …], each block the same N nodes in the same order, so a
    # plain column-repeat is the correct alignment. When widths already match
    # (single graph / batch size 1) this is a no-op with no allocation.
    n_dyn  = size(state, 2)
    static = size(static, 2) == n_dyn ? static :
             repeat(static, 1, n_dyn ÷ size(static, 2))
    x = vcat(state, forcing, static)
    h = m.encoder(x)
    h = m.processor(g, h)
    if isnothing(m.mass_balance)
        Δ = m.decoder(h)
        return state .+ Δ
    else
        if m.augment_mb
            # Append physics-derived, known-at-t features to the decoder input.
            uq_norm, h_expl_norm = mb_decoder_features(m.mass_balance, g, state, forcing)
            Δ = m.decoder(vcat(h, uq_norm, h_expl_norm))
        else
            Δ = m.decoder(h)
        end
        q_new = _floor_q_norm_nonnegative(m.mass_balance, g, state[1:1, :] .+ Δ)
        h_new = m.mass_balance(g, state, forcing, forcing_next, q_new)
        return vcat(q_new, h_new)
    end
end

