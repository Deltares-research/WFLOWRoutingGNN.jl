#!/usr/bin/env julia
# scripts/hparsearch.jl
#
# Launch a WflowRoutingGNN hyperparameter search from the command line.
#
# Usage:
#   julia --project=<repo_root> scripts/hparsearch.jl [--resume] <path/to/config.toml>
#
# Example:
#   julia --project=. scripts/hparsearch.jl experiments/template_hparsearch.toml
#
# --resume  skip the cells that already finished (metrics.toml + output NetCDFs present and
#           identical settings) and rebuild their results-CSV rows from disk; incomplete
#           cells are rerun from scratch. Without the flag every cell is run and overwritten.

const USAGE = "Usage: julia --project=<repo_root> scripts/hparsearch.jl [--resume] <config.toml>"

resume = "--resume" in ARGS
args   = filter(!=("--resume"), ARGS)

if length(args) != 1 || startswith(args[1], "--")
    println(stderr, USAGE)
    exit(1)
end

toml_path = args[1]

if !isfile(toml_path)
    println(stderr, "Error: file not found: $toml_path")
    exit(1)
end

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN

hpar_search(toml_path; resume)
