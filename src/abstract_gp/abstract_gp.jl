# Define the AbstractGP type and its API.

"""
    abstract type AbstractGP end

Supertype for various Gaussian process (GP) types. A common interface is provided for
interacting with each of these objects. See [1] for an overview of GPs.

[1] - C. E. Rasmussen and C. Williams. "Gaussian processes for machine learning". MIT Press.
2006.
"""
abstract type AbstractGP end

"""
    mean(f::AbstractGP, x::AbstractVector)

Computes the mean vector of the multivariate Normal `f(x)`.
"""
Statistics.mean(::AbstractGP, ::AbstractVector)

"""
    cov(f::AbstractGP, x::AbstractVector)

Compute the `length(x)` by `length(x)` covariance matrix of the multivariate Normal `f(x)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector)

"""
    cov_diag(f::AbstractGP, x::AbstractVector)

Compute only the diagonal elements of `cov(f(x))`.
"""
cov_diag(::AbstractGP, x::AbstractVector)

"""
    cov(f::AbstractGP, x::AbstractVector, y::AbstractVector)

Compute the `length(x)` by `length(y)` cross-covariance matrix between `f(x)` and `f(y)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector, y::AbstractVector)

"""
    mean_and_cov(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and `cov(f(x))`. Sometimes more efficient than separately
computation, particularly for posteriors.
"""
mean_and_cov(::AbstractGP, ::AbstractVector)

"""
    mean_and_cov_diag(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and the diagonal elements of `cov(f(x))`. Sometimes more efficient
than separately computation, particularly for posteriors.
"""
mean_and_cov_diag(f::AbstractGP, x::AbstractVector)
