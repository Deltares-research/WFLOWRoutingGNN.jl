#!/usr/bin/env julia
# scripts/train.jl
#
# Launch a WflowRoutingGNN training run from the command line.
#
# Usage:
#   julia --project=<repo_root> scripts/train.jl <path/to/config.toml>
#
# Example:
#   julia --project=. scripts/train.jl experiments/template.toml

if length(ARGS) != 1
    println(stderr, "Usage: julia --project=<repo_root> scripts/train.jl <config.toml>")
    exit(1)
end

toml_path = ARGS[1]

if !isfile(toml_path)
    println(stderr, "Error: file not found: $toml_path")
    exit(1)
end

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN

run_wflow_gnn_from_toml(toml_path)
