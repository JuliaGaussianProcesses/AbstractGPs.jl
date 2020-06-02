using Documenter, AbstractGPs, KernelFunctions

using Literate
using Plots # to not capture precompilation output

Literate.markdown(joinpath(@__DIR__, "..", "examples/EllipticalSliceSampling.jl"), joinpath(@__DIR__, "src/generated"); name = "EllipticalSliceSampling")

makedocs(;
    modules=[AbstractGPs],
    format=Documenter.HTML(),
    repo="https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/blob/{commit}{path}#L{line}",
    sitename="AbstractGPs.jl",
    pages = Any[
        "index.md",
        "generated/EllipticalSliceSampling.md"
    ],
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl",
)
