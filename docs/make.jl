using Documenter, AbstractGPs

using Literate
using Plots # to not capture precompilation output

example_files = filter(x->occursin(r".jl$",x), readdir(joinpath(@__DIR__, "..", "examples")))
generated_examples = Array{String}(undef, length(example_files))

for (i, example) in enumerate(example_files)
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", example),
        joinpath(@__DIR__, "src/generated");
        name = example[begin:end-3],
    )
    generated_examples[i] = joinpath("generated", string(split(example, ".jl")[1], ".md"))
end




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
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl",
    push_preview=true
)
