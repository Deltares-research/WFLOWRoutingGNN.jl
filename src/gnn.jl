using Flux
using GraphNeuralNetworks
import TOML

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
- `nlayers`         : number of `GraphConv` layers in the processor (default `3`).
- `enc_activation`  : activation for the encoder `Dense` layer (default `swish`).
- `proc_activation` : activation for each `GraphConv` layer (default `swish`).
"""
Base.@kwdef struct ModelSettings
    domain          :: String
    hidden_dim      :: Int = 64
    nlayers         :: Int = 3
    enc_activation         = swish
    proc_activation        = swish
end

function Base.show(io::IO, s::ModelSettings)
    println(io, "ModelSettings:")
    println(io, "  domain          : ", s.domain)
    println(io, "  hidden_dim      : ", s.hidden_dim)
    println(io, "  nlayers         : ", s.nlayers)
    println(io, "  enc_activation  : ", _activation_name(s.enc_activation))
    print(  io, "  proc_activation : ", _activation_name(s.proc_activation))
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
        "enc_activation"  => _activation_name(s.enc_activation),
        "proc_activation" => _activation_name(s.proc_activation),
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
        enc_activation  = ACTIVATIONS[enc_name],
        proc_activation = ACTIVATIONS[proc_name],
    )
end

"""
    MassBalanceLayer

A non-trainable layer that enforces the kinematic-wave mass balance as a hard
constraint in the river routing forward pass.

Given the predicted normalised discharge `q_norm_new` it computes the new
water depth `h_norm_new` deterministically:

    h_phys_new = h_phys_old + dt / (w·l) · (ΣQ_upstream + Q_inwater − Q_out)

where `1/(w·l) = postscale_h / postscale_q` and all Q values are in m³/s.

Fields (all per-node constants, not optimised):
- `postscale_q`   : upstream area per node [m²] (inverse of the river_q pre-scaling)
- `postscale_h`   : `a/(w·l)` per node (inverse of the river_h pre-scaling)
- `μ_q`, `σ_q`   : z-score statistics of (scaled) river_q
- `μ_h`, `σ_h`   : z-score statistics of (scaled) river_h
- `μ_inwater`, `σ_inwater` : z-score statistics of river_inwater
- `dt`            : model timestep in seconds
"""
struct MassBalanceLayer{V <: AbstractVector{Float32}}
    postscale_q  :: V
    postscale_h  :: V
    μ_q          :: Float32
    σ_q          :: Float32
    μ_h          :: Float32
    σ_h          :: Float32
    μ_inwater    :: Float32
    σ_inwater    :: Float32
    dt           :: Float32
end

Flux.@layer MassBalanceLayer
Flux.trainable(::MassBalanceLayer) = (;)  # physics constants, not optimised

"""
    (l::MassBalanceLayer)(g, state, forcing, q_norm_new) -> h_norm_new

    Compute normalised water depth at the next timestep by enforcing the
    fully-implicit kinematic-wave mass balance:

        h[t+1] = h[t] + dt/(w·l) · (Σq[t+1] + iw[t+1] − q[t+1])

- `state`         : normalised state matrix `(2, n_nodes)` — rows are river_q, river_h at t.
- `forcing`       : normalised forcing matrix at t; unused (kept for API symmetry).
- `forcing_next`  : normalised forcing matrix at t+1; row 1 is river_inwater[t+1].
- `q_norm_new`    : normalised predicted discharge `(1, n_nodes)` at t+1.

Returns normalised `h_norm_new` of shape `(1, n_nodes)`.

Handles batched `GNNGraph`s automatically: per-node constants are tiled to
match the total node count of the batch.
"""
function (l::MassBalanceLayer)(g            ::GNNGraph,
                               state        ::AbstractMatrix,
                               forcing      ::AbstractMatrix,
                               forcing_next ::AbstractMatrix,
                               q_norm_new   ::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    # Tile per-node constants to cover the whole batch  (1 × n_nodes)
    pq = reshape(repeat(l.postscale_q, n_rep), 1, n)
    ph = reshape(repeat(l.postscale_h, n_rep), 1, n)

    # Physical discharge at current and predicted timesteps  [m³/s]
    # q_phys_new is floored at 0 in physical space (z-scored 0 ≠ physical 0).
    q_phys_new  = max.(0f0, pq .* (q_norm_new .* l.σ_q .+ l.μ_q))

    # Lateral inflow at t+1  [m³/s]  (row 1 = river_inwater)  — fully-implicit
    inwater_phys = forcing_next[1:1, :] .* l.σ_inwater .+ l.μ_inwater

    # Sum upstream Q[t+1] into each node via the river network edges  [m³/s]
    # Fully-implicit: both upstream and outflow use the predicted q at t+1.
    upstream_q = propagate((xi, xj, e) -> xj, g, +; xj = q_phys_new)

    # Physical h at current step  [m]
    # h_phys = postscale_h · (norm_h · σ_h + μ_h)
    h_phys_curr = ph .* (state[2:2, :] .* l.σ_h .+ l.μ_h)

    # Mass balance  [m]:  Δh = dt · (1/(w·l)) · net_flux
    #   1/(w·l) = a/(w·l) / a = postscale_h / postscale_q
    h_phys_new = h_phys_curr .+
                 l.dt .* (ph ./ pq) .* (upstream_q .+ inwater_phys .- q_phys_new)

    # Water depth cannot be negative (dry-channel floor)
    h_phys_new = max.(0f0, h_phys_new)

    # Re-normalise:  scaled_h = h_phys / postscale_h  →  norm_h = (scaled_h - μ_h) / σ_h
    return (h_phys_new ./ ph .- l.μ_h) ./ l.σ_h
end

"""
    mb_diagnostics(l, g, state, forcing, q_norm_new) -> NamedTuple

Compute all intermediate physical quantities of the mass balance for one step.
Non-differentiable; intended for validation/debugging only.

Returns a `NamedTuple` with `Vector{Float32}` per node (physical units):
- `q_phys_curr`  [m³/s]: discharge from input state (denorm + postscale)
- `q_phys_new`   [m³/s]: predicted discharge (floored at 0)
- `upstream_q`   [m³/s]: sum of direct upstream neighbours' discharge
- `inwater_phys` [m³/s]: lateral inflow
- `net_flux`     [m³/s]: upstream_q + inwater - q_phys_new
- `h_phys_curr`  [m]:    current water depth
- `h_phys_raw`   [m]:    next water depth before the ≥0 floor
- `h_phys_new`   [m]:    next water depth after floor
"""
function mb_diagnostics(l            ::MassBalanceLayer,
                        g            ::GNNGraph,
                        state        ::AbstractMatrix,
                        forcing      ::AbstractMatrix,
                        forcing_next ::AbstractMatrix,
                        q_norm_new   ::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    # Force everything to CPU plain arrays for the diagnostic
    pq  = reshape(repeat(Array(l.postscale_q), n_rep), 1, n)
    ph  = reshape(repeat(Array(l.postscale_h), n_rep), 1, n)
    st  = Array(state)
    fn  = Array(forcing_next)
    qn  = Array(q_norm_new)
    g_c = g isa GNNGraph ? Flux.cpu(g) : g

    q_phys_curr  = pq .* (st[1:1, :] .* l.σ_q .+ l.μ_q)   # for diagnostics only
    q_phys_new   = max.(0f0, pq .* (qn .* l.σ_q .+ l.μ_q))
    inwater_phys = fn[1:1, :] .* l.σ_inwater .+ l.μ_inwater
    upstream_q   = propagate((xi, xj, e) -> xj, g_c, +; xj = q_phys_new)
    h_phys_curr  = ph .* (st[2:2, :] .* l.σ_h .+ l.μ_h)
    net_flux     = upstream_q .+ inwater_phys .- q_phys_new
    h_phys_raw   = h_phys_curr .+ l.dt .* (ph ./ pq) .* net_flux
    h_phys_new   = max.(0f0, h_phys_raw)

    return (q_phys_curr  = vec(q_phys_curr),
            q_phys_new   = vec(q_phys_new),
            upstream_q   = vec(upstream_q),
            inwater_phys = vec(inwater_phys),
            net_flux     = vec(net_flux),
            h_phys_curr  = vec(h_phys_curr),
            h_phys_raw   = vec(h_phys_raw),
            h_phys_new   = vec(h_phys_new))
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

`A` is **not** a trainable parameter; it is moved to GPU automatically by
`Flux.gpu` / `fmap`.
"""
struct SparseConv{M <: AbstractMatrix{Float32}, V <: AbstractVector{Float32}, F}
    W_self  :: M
    W_neigh :: M
    bias    :: V
    σ       :: F
    A       :: AbstractMatrix{Float32}  # topology — not trained, but gpu-movable
end

Flux.@layer SparseConv trainable=(W_self, W_neigh, bias)

"""
    SparseConv(ch::Pair{Int,Int}, σ = identity; A)

Construct a `SparseConv` layer mapping `ch.first`-dimensional inputs to
`ch.second`-dimensional outputs.  `W_self` and `W_neigh` are initialised with
Glorot uniform; `bias` is zero-initialised.
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
    )
end

"""
    (l::SparseConv)(g::GNNGraph, h::AbstractMatrix) -> AbstractMatrix

Forward pass.  `g` is accepted for `GNNChain` compatibility but is ignored;
neighbour aggregation is performed via the stored matrix `l.A`.
"""
function (l::SparseConv)(::GNNGraph, h::AbstractMatrix{Float32})
    neigh = (l.A * h')'
    l.σ.(l.W_self * h .+ l.W_neigh * neigh .+ l.bias)
end
# Single-argument method so that GNNChain._applylayer (which calls l(x) for
# non-GNNLayer types) works correctly when SparseConv is used inside a GNNChain.
(l::SparseConv)(h::AbstractMatrix{Float32}) = l(nothing, h)

# Accept any first argument so that both `l(g, x)` and `l(nothing, x)` dispatch
# to the same implementation.
function (l::SparseConv)(::Any, h::AbstractMatrix{Float32})
    neigh = (l.A * h')'
    l.σ.(l.W_self * h .+ l.W_neigh * neigh .+ l.bias)
end
"""
    WflowGNN

Encode-process-decode GNN for wflow routing emulation.

- `encoder`      : `Dense` layer mapping `in_dim -> hidden_dim` with a configurable activation.
- `processor`    : `GNNChain` of `GraphConv` or `SparseConv` layers operating at `hidden_dim`.
- `decoder`      : `Dense` layer mapping `hidden_dim -> out_dim` (no activation).
- `mass_balance` : optional `MassBalanceLayer` that hard-constrains `river_h` via the
                   kinematic-wave mass balance. When present the decoder outputs only
                   `Δq` (`out_dim = 1`) and `river_h` is computed analytically.
"""
struct WflowGNN
    encoder      :: Dense
    processor    :: GNNChain
    decoder      :: Dense
    mass_balance :: Union{Nothing, MassBalanceLayer}
end

Flux.@layer WflowGNN

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
    in_dim  = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"])
    out_dim = length(vars["state"])
    return WflowGNN(in_dim, s.hidden_dim, out_dim;
                    nlayers         = s.nlayers,
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
    in_dim  = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"])
    out_dim = length(vars["state"])
    return WflowGNN(in_dim, s.hidden_dim, out_dim;
                    nlayers         = s.nlayers,
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
    in_dim = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"])
    return WflowGNN(in_dim, s.hidden_dim, 1;
                    nlayers         = s.nlayers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation,
                    mass_balance    = mb)
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
    in_dim = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"])
    return WflowGNN(in_dim, s.hidden_dim, 1;
                    nlayers         = s.nlayers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation,
                    mass_balance    = mb,
                    adj_matrix      = A)
end

# Internal constructor — shared by all public constructors.
# Pass `adj_matrix` to use SparseConv layers instead of GraphConv.
function WflowGNN(
        in_dim    :: Int,
        hidden_dim:: Int,
        out_dim   :: Int;
        nlayers         :: Int = 3,
        enc_activation        = swish,
        proc_activation       = swish,
        mass_balance          = nothing,
        adj_matrix            = nothing)

    encoder = Dense(in_dim => hidden_dim, enc_activation)
    if isnothing(adj_matrix)
        processor = GNNChain([GraphConv(hidden_dim => hidden_dim, proc_activation) for _ in 1:nlayers]...)
    else
        A = adj_matrix :: AbstractMatrix{Float32}
        processor = GNNChain([SparseConv(hidden_dim => hidden_dim, proc_activation; A=A) for _ in 1:nlayers]...)
    end
    decoder = Dense(hidden_dim => out_dim)
    return WflowGNN(encoder, processor, decoder, mass_balance)
end

"""
    (m::WflowGNN)(g::GNNGraph)

Forward pass using node features stored in `g.ndata`.

1. Concatenate `g.ndata.state`, `g.ndata.forcing`, and `g.ndata.static` along the
   feature dimension to form the input `x` of shape `(in_dim, n_nodes)`.
2. Encode -> process -> decode: `Delta = decoder(processor(g, encoder(x)))`.
3. Add the decoded output back to the state features (residual connection):
   returns `g.ndata.state .+ Delta` of shape `(n_state, n_nodes)`.
"""
function (m::WflowGNN)(g::GNNGraph)
    m(g, g.ndata.state, g.ndata.forcing, g.ndata.static)
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
    x = vcat(state, forcing, static)
    h = m.encoder(x)
    h = m.processor(g, h)
    Δ = m.decoder(h)
    if isnothing(m.mass_balance)
        return state .+ Δ
    else
        # Δ is (1, n_nodes): predicted Δq.
        # h_new is derived analytically from the fully-implicit mass balance.
        q_new = state[1:1, :] .+ Δ
        h_new = m.mass_balance(g, state, forcing, forcing_next, q_new)
        return vcat(q_new, h_new)
    end
end
