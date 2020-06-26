using Documenter, AbstractGPs

using Literate
using Plots # to not capture precompilation output

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

Documenter.post_status(; type="pending", repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl.git")

for filename in readdir(joinpath(@__DIR__, "..", "examples"))
    endswith(filename, ".jl") || continue
	name = splitext(filename)[1]
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", filename),
        joinpath(@__DIR__, "src/generated");
        name = name,
        documenter=true,
    )
end

generated_examples = joinpath.("generated", filter(x -> endswith(x, ".md"), readdir(joinpath(@__DIR__, "src", "generated"))))

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
        generated_examples...
    ],

)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl",
    push_preview=true
)
