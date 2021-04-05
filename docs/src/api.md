# FiniteGPs and AbstractGPs

AbstractGPs provides one main abstract type, the `AbstractGP`.
Instances of subtypes of `AbstractGP` represent Gaussian processes -- collections of jointly-Gaussian random variables, which may be infinite-dimensional.
AbstractGPs only become useful computationally when you specify a finite-dimensional marginal distribution to work with.
It is for this reason that we provide the `FiniteGP` type.
Roughly speaking, this type comprises an `AbstractGP` `f` and an `AbstractVector` `x`, and represents the multivariate Gaussian distribution over `f` at `x`.
This distribution, being finite-dimensional, is something that can be used to compute useful things.
For example, `rand(f(x))` generates a sample from the multivariate Gaussian that is `f` at `x`, and `logpdf(f(x), y)` computes the log (marginal) probability of observing the vector `y` under this distribution.

Consequently, if you create a new `AbstractGP` subtype, say you called it `MyNewGP`, the methods that you must ensure are implemented are not those directly involving `MyNewGP`, but rather
```julia
FiniteGP{<:MyNewGP}
```
We provide two ways in which to do this.
The first is to implement methods directly on `Finite{<:MyNewGP}` -- this is detailed in the [FiniteGP APIs](@ref).
The second is to implement some methods directly involving `MyNewGP`, and utilise default `FiniteGP` methods implemented in terms of these -- this is detailed in the [Internal AbstractGPs API](@ref).

The second interface is generally easier to implement, but isn't applicable for all subtypes of `AbstractGP`.
See [Which API should I implement?](@ref) for further discussion.

Note that neither `AbstractGP` nor `FiniteGP` are exported as this package also provides user-facing concrete types which are generally what users should interact with.
Package developers, and anyone writing code that is intended to work with any GP in the ecosystem, should import what they need.

## Intended Audience

If you do not plan to implement your own `AbstractGP` subtype because e.g. you just wish to build a model involving a GP, then the useful part of the API for you is the [FiniteGP APIs](@ref).
Please avoid working directly with the [Internal AbstractGPs API](@ref).
In fact, you can safely ignore its existence.





## FiniteGP APIs

A `FiniteGP` represents the multivariate Gaussian induced by "indexing" into an AbstractGP `f` at an `AbstractVector` of points `x`, and adding independent zero-mean noise with covariance matrix `Σ`:
```julia
fx = f(x, Σ)

# The code below is equivalent to the above, and is just for reference.
# When writing code, prefer the above syntax.
fx = AbstractGPs.FiniteGP(f, x, Σ)
```

The `FiniteGP` has two API levels.
The [Primary Public API](@ref) should be supported by all `FiniteGP`s, while the [Secondary Public API](@ref) will only be supported by a subset.
Use only the primary API when possible.



### Primary Public API

These are user-facing methods.
You can expect them to be implemented whenever you encounter a `FiniteGP`.
If you are building something on top of AbstractGPs, try to implement it in terms of these functions.

#### Required Methods

```@docs
rand
marginals
logpdf(::AbstractGPs.FiniteGP, ::AbstractVector{<:Real})
posterior(::AbstractGPs.FiniteGP, ::AbstractVector{<:Real})
mean(::AbstractGPs.FiniteGP)
var(::AbstractGPs.FiniteGP)

```

#### Optional methods
Default implementations are provided for these, but you may wish to specialise for performance.
```@docs
mean_and_var(::AbstractGPs.FiniteGP)
```



### Secondary Public API

While the covariance matrix of any multivariate Gaussian is defined, it is not always a good idea to actually compute it.
Fortunately, it's often the case that you're not actually interested in the covariance matrix per-se, rather the other quantities that you might use it to compute (`logpdf`, `rand`, `posterior`).
This is similar to the well-known observation that you rarely need the inverse of a matrix, you just need to compute the inverse multiplied by something, so it's considered good practice to avoid ever explicitly computing the inverse of a matrix so as to avoid the numerical issues associated with it.
This is important, for example, as [TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl) is able to [implement](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/blob/master/src/gp/lti_sde.jl) all of the [Primary Public API](@ref) in linear time in the dimension of the `FiniteGP`, as it never needs to evaluate the covariance matrix.

However, for many (probably the majority of) GPs, this acceleration isn't possible, and there is really nothing lost by explicitly evaluating the covariance matrix.
We call this the [Secondary Public API](@ref), because it's available a large proportion of the time, but should be avoided if at all possible.

#### Required Methods

```@docs
cov(::AbstractGPs.FiniteGP)
```

#### Optional Methods
Default implementations are provided for these, but you may wish to specialise for performance.
```@docs
mean_and_cov(::AbstractGPs.FiniteGP)
```


## Internal AbstractGPs API

This functionality is not intended to be used directly by the users, or those building functionality on top of `AbstractGP` -- they should interact with [Primary Public API](@ref).

Implementing the following API for your own GP type automatically implements both the Primary and Secondary public APIs above in terms of them.

Existing implementations of this interface include
1. [`GP`](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/blob/3b5de4f4da80e4e3a7dcf716764b298d953a0b37/src/gp/gp.jl#L56)
1. [`PosteriorGP`](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/blob/3b5de4f4da80e4e3a7dcf716764b298d953a0b37/src/posterior_gp/posterior_gp.jl#L1)
1. [`ApproxPosteriorGP`](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/blob/3b5de4f4da80e4e3a7dcf716764b298d953a0b37/src/posterior_gp/approx_posterior_gp.jl#L4)
1. [`WrappedGP`](https://github.com/JuliaGaussianProcesses/Stheno.jl/blob/b4e2d20f973a0816272fdf07bdd5896a614b99e1/src/gp/gp.jl#L11)
1. [`CompositeGP`](https://github.com/JuliaGaussianProcesses/Stheno.jl/blob/b4e2d20f973a0816272fdf07bdd5896a614b99e1/src/composite/composite_gp.jl#L7)
1. [`GaussianProcessProbabilisticProgramme`](https://github.com/JuliaGaussianProcesses/Stheno.jl/blob/b4e2d20f973a0816272fdf07bdd5896a614b99e1/src/gaussian_process_probabilistic_programme.jl#L8)


#### Required Methods

```@docs
mean(::AbstractGPs.AbstractGP, ::AbstractVector)
cov(::AbstractGPs.AbstractGP, ::AbstractVector, ::AbstractVector)
var(::AbstractGPs.AbstractGP, ::AbstractVector)
```

#### Optional Methods
Default implementations are provided for these, but you may wish to specialise for performance.
```@docs
cov(::AbstractGPs.AbstractGP, ::AbstractVector)
mean_and_cov(::AbstractGPs.AbstractGP, ::AbstractVector)
mean_and_var(::AbstractGPs.AbstractGP, ::AbstractVector)
```

Note that, while we _could_ provide a default implementation for `var(f, x)` as `diag(cov(f, x))`, this is generally such an inefficient fallback, that we find it preferable to error if it's not implemented than to ever hit a fallback.



## Which API should I implement?

To answer this question, you need to need to know whether or not the default implementations of the [FiniteGP APIs](@ref) work for your use case.
There are a couple of reasons of which we are aware for why this might not be the case -- possibly there are others.

If you are unsure, please open an issue to discuss.



### You want to avoid computing the covariance matrix

We've already discussed this a bit on this page.
The default implementations of the [FiniteGP APIs](@ref) rely on computing the covariance matrix.
If your `AbstractGP` subtype needs to avoid computing the covariance matrix for performance reasons, then do _not_ implement the [Internal AbstractGPs API](@ref).
_Do_ implement the [Primary Public API](@ref).
Do _not_ implement the [Secondary Public API](@ref).

[TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl) is an example of a package that does this -- see the [`LTISDE`](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/blob/24343744cf60a50e09b301dee6f14b03cba7ccba/src/gp/lti_sde.jl#L7) implementation for an example.




### You don't want to use the default implementations

Perhaps you just don't like the default implementations because you don't want to make use of Cholesky factorisations.
For example, [GPyTorch](https://gpytorch.ai/) avoids the Cholesky factorisation in favour of iterative solvers.

In this situation, considering implementing _both_ the [Internal AbstractGPs API](@ref) _and_ the [FiniteGP APIs](@ref).

In this situation you will benefit less from code reuse inside AbstractGPs, but will continue to benefit from the ability of others use your code, and to take advantage of any existing functionality which requires types which adhere to the AbstractGPs API.

We don't have an example of this yet.
