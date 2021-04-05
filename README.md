# AbstractGPs

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/AbstractGPs.jl/dev)
![CI](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/workflows/CI/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/AbstractGPs.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

AbstractGPs.jl is a package that defines a low-level API for working with Gaussian processes (GPs), and basic functionality for working with them in the simplest cases. As such it is aimed more at developers and researchers who are interested in using it as a building block than end-users of GPs.

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
X = rand(10)
Y = sin.(rand(10))

# Define GP prior with Matern32 kernel
f = GP(Matern32Kernel())

# Finite projection at the inputs `X`
fx = f(X, 0.001)

# Data's log-likelihood w.r.t prior GP `f`. 
logpdf(fx, Y)

# Exact posterior given `Y`.
p_fx = posterior(fx, Y)

# Data's log-likelihood w.r.t posterior GP `p_fx`. 
logpdf(p_fx(X), Y)

# Plot posterior.
scatter(X, Y; label="Data")
plot!(-0.5:0.001:1.5, p_fx; label="Posterior")
```


## Related Projects
- [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/) -  Julia Package for kernel functions for machine learning 
- [LikelihoodFunctions.jl](https://github.com/JuliaGaussianProcesses/LikelihoodFunctions.jl/) - Julia package for non-gaussian likelihood functions to use with Gaussian Processes.
- [TemporalGPs.jl](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl) - Julia package for accelerated inference in GPs involving time. Implements the `AbstractGP` API.


## Issues/Contributing

If you notice a problem or would like to contribute by adding more kernel functions or features please [submit an issue](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/issues).
