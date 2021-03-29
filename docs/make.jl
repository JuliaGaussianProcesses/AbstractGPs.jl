using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using Literate, AbstractGPs

EXAMPLES = joinpath(@__DIR__, "..", "examples")
OUTPUT = joinpath(@__DIR__, "src", "examples")

ispath(OUTPUT) && rm(OUTPUT; recursive=true)

# add links to binder and nbviewer below the first heading of level 1
function preprocess(content)
    sub = s"""
\0
#
# [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)
"""
    return replace(content, r"^# # .*$"m => sub; count=1)
end

for file in readdir(EXAMPLES; join=true)
    endswith(file, ".jl") || continue
    Literate.markdown(file, OUTPUT; documenter=true, preprocess=preprocess)
    Literate.notebook(file, OUTPUT; documenter=true)
end

DocMeta.setdocmeta!(
    AbstractGPs,
    :DocTestSetup,
    :(using AbstractGPs, LinearAlgebra, Random);
    recursive=true,
)

makedocs(;
    modules=[AbstractGPs],
    format=Documenter.HTML(),
    repo="https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/blob/{commit}{path}#L{line}",
    sitename="AbstractGPs.jl",
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => joinpath.("examples", filter(x -> endswith(x, ".md"), readdir(OUTPUT))),
    ],
    strict=true,
    checkdocs=:exports, 
)

deploydocs(;
    repo = "github.com/JuliaGaussianProcesses/AbstractGPs.jl.git",
    push_preview = true,
)
