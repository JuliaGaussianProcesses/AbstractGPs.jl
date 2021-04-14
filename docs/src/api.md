# FiniteGP and AbstractGP

## Intended Audience

This page is intended for developers.
If you are a user, please refer to our other examples.




## Introduction

AbstractGPs provides the abstract type `AbstractGP`, and the concrete type `FiniteGP`.
An `AbstractGP`, `f`, should be thought of as a distribution over functions.
This means that the output of `rand(f)` would be a real-valued function.
It's not usually possible to implement this though, so we don't.

A `FiniteGP` `fx = f(x)` represents the distribution over functions at the finite collection of points specified in `x`.
`fx` is a multivariate Normal distribution, so `rand(fx)` produces a `Vector` of `Real`s.

A `FiniteGP` is the interesting object computationally, so if you create a new subtype `MyNewGP` of `AbstractGP`, and wish to make it interact well with the rest of the GP ecosystem, the methods that you must implement are not those directly involving `MyNewGP`, but rather those involving
```julia
FiniteGP{<:MyNewGP}
```
We provide two ways in which to do this.
The first is to implement methods directly on `Finite{<:MyNewGP}` -- this is detailed in the [FiniteGP APIs](@ref).
The second is to implement some methods directly involving `MyNewGP`, and utilise default `FiniteGP` methods implemented in terms of these -- this is detailed in the [Internal AbstractGPs API](@ref).
For example, the first method involves implementing methods like `AbstractGPs.mean(fx::FiniteGP{<:MyNewGP})`, while the second involves `AbstractGPs.mean(f::MyNewGP, x::AbstractVector)`.

The second interface is generally easier to implement, but sometimes it isn't always appropriate.
See [Which API should I implement?](@ref) for further discussion.







## FiniteGP APIs

Let `f` be an `AbstractGP`, `x` an `AbstractVector` representing a collection of inputs, and `Σ` a positive-definite matrix of size `(length(x), length(x))`.
A `FiniteGP` represents the multivariate Gaussian induced by "indexing" into `f` at each point in `x`, and adding independent zero-mean noise with covariance matrix `Σ`:
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

This functionality is not intended to be used directly by the users, or those building functionality on top of this package -- they should interact with [Primary Public API](@ref).

As discussed at the top of this page, instances of subtypes of `AbstractGP` represent Gaussian processes -- collections of jointly-Gaussian random variables, which may be infinite-dimensional.

Implementing the following API for your own `AbstractGP` subtype automatically implements both the Primary and Secondary public APIs above in terms of them.

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
There are a couple of reasons of which we are aware for why this might not be the case (see below) -- possibly there are others.

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
We don't have an example of this yet in Julia, however [GPyTorch](https://gpytorch.ai/) avoids the Cholesky factorisation in favour of iterative solvers.

In this situation, implement _both_ the [Internal AbstractGPs API](@ref) _and_ the [FiniteGP APIs](@ref).

In this situation you will benefit less from code reuse inside AbstractGPs, but will continue to benefit from the ability of others use your code, and to take advantage of any existing functionality which requires types which adhere to the AbstractGPs API.
