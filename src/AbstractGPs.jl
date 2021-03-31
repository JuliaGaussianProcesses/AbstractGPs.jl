module AbstractGPs

    using ChainRulesCore
    using Distributions
    using FillArrays
    using LinearAlgebra
    using Reexport
    @reexport using KernelFunctions
    using Random
    using Statistics
    using RecipesBase

    using KernelFunctions: ColVecs, RowVecs

    export GP, mean, cov, var, std, mean_and_cov, mean_and_var, marginals,
        logpdf, elbo, dtc, posterior, approx_posterior, VFE, DTC, sampleplot,
        update_approx_posterior, LatentGP, ColVecs, RowVecs

    # Various bits of utility functionality.
    include(joinpath("util", "common_covmat_ops.jl"))

    # AbstractGP interface and FiniteGP interface.
    include(joinpath("abstract_gp", "abstract_gp.jl"))
    include(joinpath("abstract_gp", "finite_gp.jl"))

    # Basic GP object.
    include(joinpath("gp", "mean_function.jl"))
    include(joinpath("gp", "gp.jl"))

    # Efficient exact and approximate posterior GP implementations.
    include(joinpath("posterior_gp", "posterior_gp.jl"))
    include(joinpath("posterior_gp", "approx_posterior_gp.jl"))

    # LatentGP object to accomodate GPs with non-gaussian likelihoods.
    include(joinpath("latent_gp", "latent_gp.jl"))

    # Plotting utilities.
    include(joinpath("util", "plotting.jl"))

    # Deprecations.
    include("deprecations.jl")
end # module
