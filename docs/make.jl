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
    format=Documenter.HTML(;
        size_threshold_ignore=[
            "../examples/0-intro-1d/index.md",
            "../examples/1-mauna-loa/index.md".
            "../examples/2-deep-kernel-learning/index.md"
        ]
    ),
    modules=[AbstractGPs],
    pages=[
        "Home" => "index.md",
        "The Main APIs" => "api.md",
        "Concrete Features" => "concrete_features.md",
        "Examples" => JuliaGPsDocs.find_generated_examples(AbstractGPs),
    ],
    warnonly=true,
    checkdocs=:exports,
    doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl.git", push_preview=true)
