# AbstractGPs

[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/AbstractGPs.jl/stable)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/AbstractGPs.jl/dev)
[![CI](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/254674526.svg)](https://zenodo.org/badge/latestdoi/254674526)



AbstractGPs.jl is a package that defines a low-level API for working with Gaussian processes (GPs), and basic functionality for working with them in the simplest cases. As such it is aimed more at developers and researchers who are interested in using it as a building block than end-users of GPs. You may want to go through the main [API design documentation](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/api/).

![GP](gp.gif)

## Installation 

AbstractGPs is an officially registered Julia package, so the following will install AbstractGPs using the Julia's package manager:

```julia
] add AbstractGPs
```

## Example
```julia
# Import packages.
using AbstractGPs, Plots

# Generate toy synthetic data.
x = rand(10)
y = sin.(x)

# Define GP prior with Matern-3/2 kernel
f = GP(Matern32Kernel())

# Finite projection of `f` at inputs `x`.
# Added Gaussian noise with variance 0.001.
fx = f(x, 0.001)

# Log marginal probability of `y` under `f` at `x`.
# Quantity typically maximised to train hyperparameters.
logpdf(fx, y)

# Exact posterior given `y`. This is another GP.
p_fx = posterior(fx, y)

# Log marginal posterior predictive probability.
logpdf(p_fx(x), y)

# Plot posterior.
scatter(x, y; label="Data")
plot!(-0.5:0.001:1.5, p_fx; label="Posterior")
```


## Related Julia packages

- [AbstractGPsMakie.jl](https://github.com/JuliaGaussianProcesses/AbstractGPsMakie.jl/) - Plotting GPs with [Makie.jl](https://github.com/JuliaPlots/Makie.jl/).
- [ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/) - Approximate inference for GPs, both for sparse approximations and non-Gaussian likelihoods. Built on types which implement this package's APIs.
- [BayesianLinearRegressors.jl](https://github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl) - Accelerated inference in GPs with a linear kernel. Built on types which implement this package's APIs.
- [GPLikelihoods.jl](https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/) - Non-Gaussian likelihood functions to use with GPs.
- [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/) - Kernel functions for machine learning.
- [Stheno.jl](https://github.com/JuliaGaussianProcesses/Stheno.jl) - Building probabilistic programmes involving GPs. Built on types which implement this package's APIs.
- [TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl) - Accelerated inference in GPs involving time. Built on types which implement this package's APIs.


## Issues/Contributing

If you notice a problem or would like to contribute by adding more kernel functions or features please [submit an issue](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues).
