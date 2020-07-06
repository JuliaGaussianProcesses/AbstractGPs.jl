using AbstractGPs
using AbstractGPs: AbstractGP, MeanFunction, FiniteGP, ConstMean, GP, ZeroMean, 
    ConstMean, CustomMean, Xt_A_X, Xt_A_Y, Xt_invA_Y, Xt_invA_X, diag_At_A, diag_At_B, 
    diag_Xt_A_X, diag_Xt_A_Y, diag_Xt_invA_X, diag_Xt_invA_Y, Xtinv_A_Xinv, tr_At_A,
    mean_and_cov_diag
using Documenter
using Distributions: MvNormal, PDMat
using KernelFunctions
using KernelFunctions: Kernel, ColVecs, RowVecs
using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using Random
using Plots
using Test
using FiniteDifferences
using FiniteDifferences: j′vp, to_vec
using Statistics
using Zygote


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

    include(joinpath("latent_gp", "latent_gp.jl"))

    include(joinpath("util", "plotting.jl"))
    
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
