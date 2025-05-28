# Define the AbstractGP type and its API.

"""
    abstract type AbstractGP end

Supertype for various Gaussian process (GP) types. A common interface is provided for
interacting with each of these objects. See [1] for an overview of GPs.

[1] - C. E. Rasmussen and C. Williams. "Gaussian processes for machine learning". 
MIT Press. 2006.
"""
abstract type AbstractGP end

"""
    mean(f::AbstractGP, x::AbstractVector)

Compute the mean vector of the multivariate Normal `f(x)`.
"""
Statistics.mean(::AbstractGP, ::AbstractVector)

# https://github.com/JuliaDiff/ChainRules.jl/pull/615 introduced a `rrule` that causes errors
ChainRulesCore.@opt_out ChainRulesCore.rrule(
    ::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
    ::typeof(mean),
    ::AbstractGP,
    ::AbstractVector,
)

"""
    cov(f::AbstractGP, x::AbstractVector)

Compute the `length(x)` by `length(x)` covariance matrix of the multivariate Normal `f(x)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector)

"""
    var(f::AbstractGP, x::AbstractVector)

Compute only the diagonal elements of `cov(f(x))`.
"""
Statistics.var(::AbstractGP, ::AbstractVector)

"""
    cov(f::AbstractGP, x::AbstractVector, y::AbstractVector)

Compute the `length(x)` by `length(y)` cross-covariance matrix between `f(x)` and `f(y)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector, y::AbstractVector)

"""
    mean_and_cov(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and `cov(f(x))`. Sometimes more efficient than
computing them separately, particularly for posteriors.
"""
StatsBase.mean_and_cov(f::AbstractGP, x::AbstractVector) = (mean(f, x), cov(f, x))

"""
    mean_and_var(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and the diagonal elements of `cov(f(x))`. Sometimes
more efficient than computing them separately, particularly for posteriors.
"""
StatsBase.mean_and_var(f::AbstractGP, x::AbstractVector) = (mean(f, x), var(f, x))

for (m, f) in [
    (:Statistics, :mean),
    (:Statistics, :var),
    (:Statistics, :cov),
    (:StatsBase, :mean_and_cov),
    (:StatsBase, :mean_and_var),
]
    @eval function $m.$f(::AbstractGP)
        return error(
            "`",
            $f,
            "(f::AbstractGP)` is not defined (on purpose!).\n",
            "Please provide an `AbstractVector` of locations `x` at which you wish to compute your ",
            $f,
            $((f === :mean_and_cov || f === :mean_and_var) ? " vectors" : " vector"),
            ", and call `",
            $f,
            "(f(x))`\n",
            "For more details please have a look at the AbstractGPs docs.",
        )
    end
end

"""
    approx_log_evidence(approx::<Approximation>, lfx::LatentFiniteGP, ys)

Compute an approximation to the log of the marginal likelihood (also known as
"evidence") under the given `approx`imation to the posterior. The return value
of `approx_log_evidence` can be used to optimise the hyperparameters of `lfx`.
"""
function approx_log_evidence end

"""
    posterior(fx::FiniteGP, y::AbstractVector{<:Real})
    posterior(approx::<Approximation>, fx::FiniteGP, y::AbstractVector{<:Real})
    posterior(approx::<Approximation>, lfx::LatentFiniteGP, y::AbstractVector)

Construct the posterior distribution over the latent Gaussian process (`fx.f`
or `lfx.fx.f`), given the observations `y` corresponding to the process's
finite projection (`fx` or `lfx`).

In the two-argument form, this describes exact GP regression with `y` observed
under a Gaussian likelihood, and returns a `PosteriorGP`.

In the three-argument form, the first argument specifies the approximation to
be used (e.g. `VFE` or defined in other packages such as ApproximateGPs.jl),
and returns an `ApproxPosteriorGP`.
"""
function posterior end
