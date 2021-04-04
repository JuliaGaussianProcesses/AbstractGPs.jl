# FiniteGPs and AbstractGPs

There are two core types that AbstractGPs provides: `FiniteGP`, and `AbstractGP`.


## FiniteGP

AbstractGPs provides the `FiniteGP` type. It represents the multivariate Gaussian induced by "indexing" into a GP `f` at a collection of points `x`, and adding independent zero-mean noise with variance `Σ`:
```julia
fx = f(x, Σ)

# The above is equivalent to:
fx = AbstractGPs.FiniteGP(f, x, Σ)
```

The `FiniteGP` has two API levels.
The first should be supported by all `FiniteGPs`, while the second will only be supported by a subset.



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
This is important, for example, as `TemporalGPs.jl` is able to implement all of the Primary Public API in linear time in the dimension of the `FiniteGP`, as it never needs to evaluate the covariance matrix.

However, for many (probably the majority of) GPs, this acceleration isn't possible, and there is really nothing lost by explicitly evaluating the covariance matrix.
We call this the Secondary Public API, because it's available a large proportion of the time, but should be avoided if at all possible.

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

This functionality is not intended to be used directly be users, or those building functionality on top of `AbstractGP` -- they should interact with Primary Public API above, and the Seconary Public API if truly necessary.

The reason for this is that some `AbstractGP`s will not actually implement any of these methods, but they will ensure that the Primary Public API is implemented for `FiniteGP`s containing them.
See the next section for more info on this.

Implementing the following API for your own GP type automatically implements both the Primary and Secondary public APIs above in terms of them.


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

Note that, while we _could_ provide a default implementation for `var` as `diag(cov(f, x))`, this is generally such an inefficient fallback, that we view find it preferable to error if it's not implemented than to ever hit a fallback.



## When to implement FiniteGP methods

If you have a new subtype of `AbstractGP` and it implements the API above (i.e. you don't mind computing covariance matrices), then you'll not usually need to add new methods involving your own `FiniteGP` -- the fallback implementations will often be completely fine.

If, on the other hand, you don't want to implement the Internal AbstractGPs API for e.g. performance reasons, then you'll need to implement the Primary Public API directly.
This is the case in `TemporalGPs.jl` -- the covariance matrix is never actually needed, so we neglect to provide any implementations involving it, instead implementing specialised methods for the Primary Public API.

There are possibly other reasons why you might wish to modify the way in which e.g. `logpdf` works for GPs implementing the Primary and Secondary public APIs.
For example, you might wish to avoid ever computing Cholesky factorisations directly, instead implementing everything in terms of matrix-vector multiplies, conjugate gradients, etc.

In these cases, we advise that you use the type parameters in `FiniteGP` to dispatch appropriately to specialised Primary Public API methods for your type. E.g.
```julia
const MyFiniteGP = FiniteGP{<:MyGPType}
AbstractGPs.logpdf(::MyFiniteGP, ::AbstractVector{<:Real})
```

