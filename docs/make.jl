### Process examples
using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl")) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

using AbstractGPs
# If any features of AbstractGPs depend on optional packages (e.g. via @require),
# make sure to load them here in order to generate the full API documentation.

JuliaGPsDocs.generate_examples(AbstractGPs)

### Build documentation
using Documenter

# Doctest setup
DocMeta.setdocmeta!(
    AbstractGPs,
    :DocTestSetup,
    quote
        using AbstractGPs
        using LinearAlgebra
        using Random
    end;  # we have to load all packages used (implicitly) within jldoctest blocks in the API docstrings
    recursive=true,
)

makedocs(;
    sitename="AbstractGPs.jl",
    format=Documenter.HTML(),
    modules=[AbstractGPs],
    pages=[
        "Home" => "index.md",
        "The Main APIs" => "api.md",
        "Concrete Features" => "concrete_features.md",
        "Examples" => map(
            basename.(
                filter!(isdir, readdir(joinpath(@__DIR__, "src", "examples"); join=true)),
            ),
        ) do x
            joinpath("examples", x, "index.md")
        end,
    ],
    #strict=true,
    checkdocs=:exports,
    doctestfilters=[
        r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
        r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
        r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
    ],
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl.git", push_preview=true)
