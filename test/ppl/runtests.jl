using AbstractGPs

using Distributions: Poisson, LogNormal
using SampleChainsDynamicHMC
using Soss: Soss, For
using Test
using Turing: Turing

@testset "PPLs" begin
    include("turing.jl")
    println(" ")
    @info "Ran Turing tests"

    include("soss.jl")
    println(" ")
    @info "Ran Soss tests"
end
