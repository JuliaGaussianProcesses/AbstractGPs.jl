using AbstractGPs
using AbstractGPs:
    AbstractGP,
    MeanFunction,
    FiniteGP,
    ConstMean,
    GP,
    ZeroMean,
    ConstMean,
    CustomMean,
    Xt_A_X,
    Xt_A_Y,
    Xt_invA_Y,
    Xt_invA_X,
    diag_At_A,
    diag_At_B,
    diag_Xt_A_X,
    diag_Xt_A_Y,
    diag_Xt_invA_X,
    diag_Xt_invA_Y,
    Xtinv_A_Xinv,
    tr_At_A

using Documenter
using ChainRulesCore
using Distributions: MvNormal, PDMat
using FillArrays
using FiniteDifferences
using FiniteDifferences: j′vp, to_vec
using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using Pkg
using Plots
using Random
using Statistics
using Test
using Zygote

include("test_util.jl")

@testset "AbstractGPs" begin
    # @testset "util" begin
    #     include(joinpath("util", "common_covmat_ops.jl"))
    #     include(joinpath("util", "plotting.jl"))
    # end
    # println(" ")
    # @info "Ran util tests"

    # @testset "abstract_gp" begin
    #     include(joinpath("abstract_gp", "abstract_gp.jl"))
    #     include(joinpath("abstract_gp", "finite_gp.jl"))
    # end
    # println(" ")
    # @info "Ran abstract_gp tests"

    # @testset "gp" begin
    #     include(joinpath("gp", "mean_functions.jl"))
    #     include(joinpath("gp", "gp.jl"))
    # end
    # println(" ")
    # @info "Ran gp tests"

    # @testset "posterior_gp" begin
    #     include(joinpath("posterior_gp", "posterior_gp.jl"))
    #     include(joinpath("posterior_gp", "approx_posterior_gp.jl"))
    # end
    # println(" ")
    # @info "Ran posterior_gp tests"

    # include(joinpath("latent_gp", "latent_gp.jl"))
    # println(" ")
    # @info "Ran latent_gp tests"

    # include("deprecations.jl")
    # println(" ")
    # @info "Ran deprecation tests"

    # @testset "doctests" begin
    #     DocMeta.setdocmeta!(
    #         AbstractGPs,
    #         :DocTestSetup,
    #         :(using AbstractGPs, Random, Documenter, LinearAlgebra);
    #         recursive=true,
    #     )
    #     doctest(AbstractGPs)
    # end

    @static if VERSION >= v"1.5"
        Pkg.activate("compat")
        Pkg.develop(PackageSpec(; path=".."))
        Pkg.instantiate()
        include(joinpath("compat", "runtests.jl"))
    end
end
