using AbstractGPs
using AbstractGPs:
    AbstractGP,
    MeanFunction,
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
    tr_At_A,
    inducing_points,
    TestUtils

using Aqua
using Documenter
using Distributions: MvNormal, PDMat, loglikelihood, Distributions
using FillArrays
using FiniteDifferences
using FiniteDifferences: j′vp, to_vec
using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using PDMats: ScalMat
using Pkg
using Plots
using Random
using Statistics
using Test
using Zygote

const GROUP = get(ENV, "GROUP", "All")
const PKGDIR = dirname(dirname(pathof(AbstractGPs)))

include("test_util.jl")

@testset "AbstractGPs" begin
    if GROUP == "All" || GROUP == "AbstractGPs"
        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(AbstractGPs; ambiguities=false)
            # Ref https://github.com/JuliaTesting/Aqua.jl/issues/77
            Aqua.test_ambiguities(AbstractGPs; recursive=false)
        end
        @testset "util" begin
            include("util/common_covmat_ops.jl")
            include("util/plotting.jl")
        end
        println(" ")
        @info "Ran util tests"

        @testset "abstract_gp" begin
            include("abstract_gp.jl")
            include("finite_gp_projection.jl")
        end
        println(" ")
        @info "Ran abstract_gp tests"

        @testset "gp" begin
            include("mean_function.jl")
            include("base_gp.jl")
        end
        println(" ")
        @info "Ran gp tests"

        @testset "posterior_gp" begin
            include("exact_gpr_posterior.jl")
            include("sparse_approximations.jl")
        end
        println(" ")
        @info "Ran posterior_gp tests"

        include("latent_gp.jl")
        println(" ")
        @info "Ran latent_gp tests"

        include("vector_valued_gp.jl")
        println(" ")
        @info "Ran vector_valued_gp tests"

        include("deprecations.jl")
        println(" ")
        @info "Ran deprecation tests"

        @testset "doctests" begin
            DocMeta.setdocmeta!(
                AbstractGPs,
                :DocTestSetup,
                :(using AbstractGPs, Random, Documenter, LinearAlgebra);
                recursive=true,
            )
            doctest(AbstractGPs)
        end
    end

    if (GROUP == "All" || GROUP == "PPL") && VERSION >= v"1.5"
        Pkg.activate(joinpath(@__DIR__, "ppl"))
        Pkg.develop(PackageSpec(; path=PKGDIR))
        Pkg.instantiate()
        include("ppl/runtests.jl")
    end
end
