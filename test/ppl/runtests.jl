using AbstractGPs

using Distributions: Poisson, LogNormal, product_distribution
using Test
using Turing: Turing

@testset "PPLs" begin
    include("turing.jl")
    println(" ")
    @info "Ran Turing tests"
end
