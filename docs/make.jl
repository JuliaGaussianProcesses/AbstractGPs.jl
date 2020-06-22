using Documenter, AbstractGPs

using Literate
using Plots # to not capture precompilation output

function preprocessing(str)
    return replace(str, r"# ---[a-zA-Z0-9 #:,.'-_\n]*# ---\n" => ""; count=1)
end

Literate.markdown(
    joinpath(@__DIR__, "..", 
    "examples/EllipticalSliceSampling.jl"), 
    joinpath(@__DIR__, "src/generated"); 
    name = "EllipticalSliceSampling", 
    preprocess=preprocessing
)

DocMeta.setdocmeta!(
    AbstractGPs,
    :DocTestSetup,
    :(using AbstractGPs, KernelFunctions, LinearAlgebra, Random);
    recursive=true,
)

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
