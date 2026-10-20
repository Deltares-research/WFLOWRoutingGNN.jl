"""
regen_mb_diagnostics.jl

Regenerate the layer-internal mass-balance diagnostics (`plots/mb_diagnostics.png`,
`plots/volume_budget.png` and the `mb_*` / `volume_budget` CSVs) of an
already-trained run WITHOUT retraining. Thin CLI over `regen_mb_diagnostics`
(src/analysis.jl): loads the preprocessed bundle (`prep/`) and the trained model,
then recomputes the diagnostics on the validation split. Use this to refresh the
diagnostics after a fix to `rollout_mb_diagnostics` / `plot_mb_diagnostics`.

Needs `prep/preprocessed.jld2`; runs from before the staged pipeline must be
re-preprocessed first.

Usage:
    julia --project=. scripts/regen_mb_diagnostics.jl <run_dir>
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using WflowRoutingGNN

function main(argv)
    length(argv) >= 1 || error("usage: regen_mb_diagnostics.jl <run_dir>")
    res = regen_mb_diagnostics(argv[1])
    res === nothing && @info "Model has no mass-balance layer: nothing to regenerate"
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
