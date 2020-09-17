# -*- coding: utf-8 -*-
# # Multi-Output data modelling using Multi-Output kernels

# Loading the necessary packages and setting seed.

using AbstractGPs, KernelFunctions, Plots, Random, Distributions
Random.seed!(1234);

# Generate synthetic multi-dim regression dataset with 1-dim inputs and 2-dim outputs.

N = 100
in_dim = 3
out_dim = 2
x = [rand(Uniform(-1, 1), 3) for _ in 1:N] # 1-dim input 
y = [[sin(xi[1]) + exp(xi[2]) , cos(xi[3])] .+ 0.1 * randn(2) for xi in x];  # 2-dim output 
plt = multidataplot(x, y, in_dim, out_dim, layout=(1,3))
plot!(plt, size=(900, 300), label="")

# Make inputs and outputs compatible with multi-output GPs.

X, Y = mo_transform(x, y, 2);

# Define a multi-output kernel which assumes the outputs are independent of each other.

k = IndependentMOKernel(Matern32Kernel())

# Instantiate a Gaussian Process with the given kernel.

f = GP(k)

# Instantiate a `FiniteGP`, a finite dimentional projection at the inputs of the dataset 
# observed under Gaussian Noise with $\sigma = 0.1$ .

fx = f(X, 0.1);

# Data's log-likelihood w.r.t prior `GP`. 

logpdf(fx, Y)

# Calculating the exact posterior over `f` given `Y`. The GP's kernel currently has 
# some arbitrary fixed parameters. 

p_fx = posterior(fx, Y);

# Data's log-likelihood under the posterior `GP`. We see that it drastically increases.

logpdf(p_fx(X, 0.1), Y)

# Plotting posterior

X_Test = MOInput(reshape(collect.(collect(Iterators.product(-1:0.05:1, -1:0.05:1, -1:0.05:1))), :), out_dim)
p_fx_x = p_fx(X_Test, 0.01)
plt = multigpplot(p_fx_x, in_dim, out_dim, layout=(1,3))
plot!(plt, size=(900, 300), label="", )


