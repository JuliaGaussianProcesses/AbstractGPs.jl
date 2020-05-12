using AbstractGPs
using Documenter
using KernelFunctions
using LinearAlgebra
using Random
using Test

include("test_util.jl")

@testset "AbstractGPs" begin
    @testset "util" begin
        include(joinpath("util", "common_covmat_ops.jl"))
    end
    @testset "abstract_gp" begin
        include(joinpath("abstract_gp", "abstract_gp.jl"))
        include(joinpath("abstract_gp", "finite_gp.jl"))
    end
    @testset "gp" begin
        include(joinpath("gp", "mean_functions.jl"))
        include(joinpath("gp", "gp.jl"))
    end
    @testset "posterior_gp" begin
        include(joinpath("posterior_gp", "posterior_gp.jl"))
        include(joinpath("posterior_gp", "approx_posterior_gp.jl"))
    end
    @testset "doctests" begin
        DocMeta.setdocmeta!(
            AbstractGPs,
            :DocTestSetup,
            :(using AbstractGPs, Random, Documenter, LinearAlgebra, KernelFunctions);
            recursive=true,
        )
        doctest(AbstractGPs)
    end
end
