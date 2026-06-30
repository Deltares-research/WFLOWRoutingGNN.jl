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
    WflowGNN

Encode-process-decode GNN for wflow routing emulation.

- `encoder`   : `Dense` layer mapping `in_dim -> hidden_dim` with a configurable activation.
- `processor` : `GNNChain` of `GraphConv` layers operating at `hidden_dim`.
- `decoder`   : `Dense` layer mapping `hidden_dim -> out_dim` (no activation).
"""
struct WflowGNN
    encoder   :: Dense
    processor :: GNNChain
    decoder   :: Dense
end

Flux.@layer WflowGNN

"""
    WflowGNN(in_dim, hidden_dim, out_dim; nlayers, enc_activation, proc_activation)

Construct a `WflowGNN`.

- `in_dim`           : number of input node features.
- `hidden_dim`       : width of the hidden representation.
- `out_dim`          : number of output node features.
- `nlayers`          : number of `GraphConv` layers in the processor (default `3`).
- `enc_activation`   : activation for the encoder `Dense` layer (default `swish`).
- `proc_activation`  : activation for each `GraphConv` layer (default `swish`).
"""
function WflowGNN(
        in_dim    :: Int,
        hidden_dim:: Int,
        out_dim   :: Int;
        nlayers         :: Int = 3,
        enc_activation        = swish,
        proc_activation       = swish)

    encoder   = Dense(in_dim => hidden_dim, enc_activation)
    processor = GNNChain([GraphConv(hidden_dim => hidden_dim, proc_activation) for _ in 1:nlayers]...)
    decoder   = Dense(hidden_dim => out_dim)

    return WflowGNN(encoder, processor, decoder)
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
    in_dim  = length(vars["state"]) + length(vars["forcing"]) + length(vars["static"])
    out_dim = length(vars["state"])
    return WflowGNN(in_dim, s.hidden_dim, out_dim;
                    nlayers         = s.nlayers,
                    enc_activation  = s.enc_activation,
                    proc_activation = s.proc_activation)
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
    return state .+ Δ
end
