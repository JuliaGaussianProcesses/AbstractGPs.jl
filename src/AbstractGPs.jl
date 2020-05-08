module AbstractGPs

    using Distributions
    using FillArrays
    using LinearAlgebra
    using KernelFunctions
    using Random
    using Statistics

    export GP, mean, cov, std, cov_diag, mean_and_cov, mean_and_cov_diag, marginals, rand,
        logpdf, elbo, dtc, posterior, approx_posterior, VFE, DTC, AbstractGP

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

    # Plotting utilities.
    include(joinpath("util", "plotting.jl"))
end # module
