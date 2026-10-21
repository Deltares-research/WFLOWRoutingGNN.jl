#!/usr/bin/env julia
# scripts/pipeline.jl
#
# Run one stage (or all) of the WflowRoutingGNN pipeline from the command line.
#
# Usage:
#   julia --project=<repo_root> scripts/pipeline.jl <stage> <config.toml>
#
# Stages (each reads the artifacts of the earlier ones from the run directory
# <runs_dir>/<run_name>/ given by the config, so they can be run separately):
#   prep      build the graph + datasets            -> prep/, output/data/
#   train     train the model                       -> model/, metrics/losses*.csv
#   infer     roll out the trained model            -> output/<split>_{pred,true}.nc
#   analyze   metrics and plots from the rollouts   -> metrics/, plots/
#   all       the four stages in one process (same as scripts/train.jl)
#
# Example:
#   julia --project=. scripts/pipeline.jl prep    experiments/template.toml
#   julia --project=. scripts/pipeline.jl train   experiments/template.toml
#   julia --project=. scripts/pipeline.jl infer   experiments/template.toml
#   julia --project=. scripts/pipeline.jl analyze experiments/template.toml

const USAGE = "Usage: julia --project=<repo_root> scripts/pipeline.jl " *
              "<prep|train|infer|analyze|all> <config.toml>"

if length(ARGS) != 2
    println(stderr, USAGE)
    exit(1)
end

stage_name, toml_path = ARGS

if !isfile(toml_path)
    println(stderr, "Error: file not found: $toml_path")
    exit(1)
end

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN

stage = try
    stage_from_name(stage_name)
catch e
    e isa ArgumentError || rethrow()
    println(stderr, "Error: ", e.msg, "\n", USAGE)
    exit(1)
end

run_stage_from_toml(stage, toml_path)
