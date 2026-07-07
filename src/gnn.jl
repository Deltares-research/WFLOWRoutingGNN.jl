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

Compute normalised water depth at the next timestep by enforcing mass balance.

- `state`      : normalised state matrix `(2, n_nodes)` — rows are river_q, river_h.
- `forcing`    : normalised forcing matrix; row 1 is river_inwater.
- `q_norm_new` : normalised predicted discharge `(1, n_nodes)` at t+1.

Returns normalised `h_norm_new` of shape `(1, n_nodes)`.

Handles batched `GNNGraph`s automatically: per-node constants are tiled to
match the total node count of the batch.
"""
function (l::MassBalanceLayer)(g      ::GNNGraph,
                               state  ::AbstractMatrix,
                               forcing::AbstractMatrix,
                               q_norm_new::AbstractMatrix)
    n     = g.num_nodes
    n_per = length(l.postscale_q)
    n_rep = n ÷ n_per

    # Tile per-node constants to cover the whole batch  (1 × n_nodes)
    pq = reshape(repeat(l.postscale_q, n_rep), 1, n)
    ph = reshape(repeat(l.postscale_h, n_rep), 1, n)

    # Physical discharge at current and predicted timesteps  [m³/s]
    q_phys_curr = pq .* (state[1:1, :]  .* l.σ_q .+ l.μ_q)
    q_phys_new  = pq .* (q_norm_new     .* l.σ_q .+ l.μ_q)

    # Lateral inflow  [m³/s]  (row 1 = river_inwater)
    inwater_phys = forcing[1:1, :] .* l.σ_inwater .+ l.μ_inwater

    # Sum upstream Q into each node via the river network edges  [m³/s]
    # propagate with aggr=+ sends xj (source q) to each target and sums.
    upstream_q = propagate((xi, xj, e) -> xj, g, +; xj = q_phys_curr)

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
    WflowGNN

Encode-process-decode GNN for wflow routing emulation.

- `encoder`      : `Dense` layer mapping `in_dim -> hidden_dim` with a configurable activation.
- `processor`    : `GNNChain` of `GraphConv` layers operating at `hidden_dim`.
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

# Internal constructor that also accepts a mass_balance keyword argument.
function WflowGNN(
        in_dim    :: Int,
        hidden_dim:: Int,
        out_dim   :: Int;
        nlayers         :: Int = 3,
        enc_activation        = swish,
        proc_activation       = swish,
        mass_balance          = nothing)

    encoder   = Dense(in_dim => hidden_dim, enc_activation)
    processor = GNNChain([GraphConv(hidden_dim => hidden_dim, proc_activation) for _ in 1:nlayers]...)
    decoder   = Dense(hidden_dim => out_dim)
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
                       state  ::AbstractMatrix,
                       forcing::AbstractMatrix,
                       static ::AbstractMatrix)
    x = vcat(state, forcing, static)
    h = m.encoder(x)
    h = m.processor(g, h)
    Δ = m.decoder(h)
    if isnothing(m.mass_balance)
        return state .+ Δ
    else
        # Δ is (1, n_nodes): predicted Δq.
        # h_new is derived analytically from the mass balance; its gradient is
        # blocked so that the h MSE does not flow back through q and overwhelm
        # the q training signal with a factor of dt (~3600 s).
        q_new = max.(0f0, state[1:1, :] .+ Δ)
        h_new = Flux.ignore_derivatives() do
            m.mass_balance(g, state, forcing, q_new)
        end
        return vcat(q_new, h_new)
    end
end
