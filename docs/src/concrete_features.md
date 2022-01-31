# Features

### Setup

```julia
using AbstractGPs, Random
rng = MersenneTwister(0)

# Construct a zero-mean Gaussian process with a matern-3/2 kernel.
f = GP(Matern32Kernel())

# Specify some input and target locations.
x = randn(rng, 10)
y = randn(rng, 10)
```

## Finite dimensional projection
Look at the finite-dimensional projection of `f` at `x`, under zero-mean observation noise with variance `0.1`.
```julia
fx = f(x, 0.1)
```

## Sample from GP from the prior at `x` under noise.
```julia
y_sampled = rand(rng, fx)
```

## Compute the log marginal probability of `y`.
```julia
logpdf(fx, y)
```

## Construct the posterior process implied by conditioning `f` at `x` on `y`.
```julia
f_posterior = posterior(fx, y)
```

## A posterior process follows the `AbstractGP` interface, so the same functions which work on the posterior as on the prior.

```julia
rand(rng, f_posterior(x))
logpdf(f_posterior(x), y)
```

## Compute the VFE approximation to the log marginal probability of `y`.
Here, z is a set of pseudo-points.
```julia
z = randn(rng, 4)
```

### Evidence Lower BOund (ELBO)
We provide a ready implentation of elbo w.r.t to the pseudo points. We can perform Variational Inference on pseudo-points by maximizing the ELBO term w.r.t pseudo-points `z` and any kernel parameters. For more information, see [examples](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/tree/master/examples).
```julia
elbo(VFE(f(z)), fx, y)
```

### Construct the approximate posterior process implied by the VFE approximation.
The optimal pseudo-points obtained above can be used to create a approximate/sparse posterior. This can be used like a regular posterior in many cases.
```julia
f_approx_posterior = posterior(VFE(f(z)), fx, y)
```

### An approximate posterior process is yet another `AbstractGP`, so you can do things with it like
```julia
marginals(f_approx_posterior(x))
```

## Sequential Conditioning
Sequential conditioning allows you to compute your posterior in an online fashion. We do this in an efficient manner by updating the cholesky factorisation of the covariance matrix and avoiding recomputing it from original covariance matrix.

```julia
# Define GP prior
f = GP(SqExponentialKernel())
```

### Exact Posterior

Generate posterior with the first batch of data by conditioning the prior on them:
```julia
p_fx = posterior(f(x[1:3], 0.1), y[1:3])
```

Generate posterior with the second batch of data, considering the previous posterior `p_fx` as the prior:
```julia
p_p_fx = posterior(p_fx(x[4:10], 0.1), y[4:10])
```

### Approximate Posterior
#### Adding observations in a sequential fashion
```julia
Z1 = rand(rng, 4)
Z2 = rand(rng, 3)
Z = vcat(Z1, Z2)
p_fx1 = posterior(VFE(f(Z)), f(x[1:7], 0.1), y[1:7])
u_p_fx1 = update_posterior(p_fx1, f(x[8:10], 0.1), y[8:10])
```

#### Adding pseudo-points in a sequential fashion
```julia
p_fx2 = posterior(VFE(f(Z1)), f(x, 0.1), y)
u_p_fx2 = update_posterior(p_fx2, f(Z2))
```

## Plotting

### Plots.jl

We provide functions for plotting samples and predictions of Gaussian processes with [Plots.jl](https://github.com/JuliaPlots/Plots.jl). You can see some examples in the [One-dimensional regression](@ref) tutorial.

```@docs
AbstractGPs.RecipesBase.plot(::AbstractVector, ::AbstractGPs.FiniteGP)
AbstractGPs.RecipesBase.plot(::AbstractGPs.FiniteGP)
AbstractGPs.RecipesBase.plot(::AbstractVector, ::AbstractGPs.AbstractGP)
sampleplot
```

### Makie.jl

You can use the Julia package [AbstractGPsMakie.jl](https://github.com/JuliaGaussianProcesses/AbstractGPsMakie.jl) to plot Gaussian processes with [Makie.jl](https://github.com/JuliaPlots/Makie.jl).

![posterior animation](https://juliagaussianprocesses.github.io/AbstractGPsMakie.jl/stable/posterior_animation.mp4)
