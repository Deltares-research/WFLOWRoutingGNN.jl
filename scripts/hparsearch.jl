#!/usr/bin/env julia
# scripts/hparsearch.jl
#
# Launch a WflowRoutingGNN hyperparameter search from the command line.
#
# Usage:
#   julia --project=<repo_root> scripts/hparsearch.jl <path/to/config.toml>
#
# Example:
#   julia --project=. scripts/hparsearch.jl experiments/template_hparsearch.toml

if length(ARGS) != 1
    println(stderr, "Usage: julia --project=<repo_root> scripts/hparsearch.jl <config.toml>")
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

hpar_search(toml_path)
