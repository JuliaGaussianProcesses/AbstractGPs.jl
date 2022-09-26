using AbstractGPs

using Distributions: Poisson, LogNormal
using Test
using Turing: Turing

@testset "PPLs" begin
    include("turing.jl")
    println(" ")
    @info "Ran Turing tests"
end
