# AbstractGPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/AbstractGPs.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/AbstractGPs.jl/dev)
[![Build Status](https://travis-ci.com/JuliaGaussianProcesses/AbstractGPs.jl.svg?branch=master)](https://travis-ci.com/JuliaGaussianProcesses/AbstractGPs.jl)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

AbstractGPs.jl is a package that defines a low-level API for working with Gaussian processes (GPs), and basic functionality for working with them in the simplest cases. As such it is aimed more at developers and researchers who are interested in using it as a building block than end-users of GPs.


## Example Usage

```julia
using AbstractGPs, KernelFunctions, Random

# Construct a zero-mean Gaussian process with a matern-3/2 kernel.
f = GP(Matern32Kernel())

rng = MersenneTwister(0)

# Specify some input locations.
x = randn(rng, 10)

# Look at the finite-dimensional marginals of `f` at `x`, under zero-mean observation noise with variance `s`.
s = 0.1
fx = f(x, s)

# Sample from the prior at `x` under noise.
y = rand(rng, fx)

# Compute the log marginal probability of `y`.
logpdf(fx, y)

# Construct the posterior process implied by conditioning `f` at `x` on `y`.
f_posterior = posterior(fx, y)

# A posterior process follows the `AbstractGP` interface, so the same
# functions work on the posterior as the prior.
rand(rng, f_posterior(x))
logpdf(f_posterior(x), y)

# Compute the VFE approximation to the log marginal probability of `y`.
z = randn(rng, 4)
u = f(z)
elbo(fx, y, u)

# Construct the approximate posterior process implied by the VFE approximation.
f_approx_posterior = approx_posterior(VFE(), fx, y, u)

# An approximate posterior process is yet another `AbstractGP`, so you can do things with it like
marginals(f_approx_posterior(x))
```
