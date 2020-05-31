# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Julia 1.4.1
#     language: julia
#     name: julia-1.4
# ---

# # An example showcasing the usage of EllipticalSliceSampling.jl with AbstractGPs.jl

# Loading the necessary packages.

using AbstractGPs, KernelFunctions, Plots, CSV, DataFrames
include("utils.jl")

# Loading [toy regression dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv) taken from GPFlow examples.

df = CSV.read("data/regression_1D.csv", header=false);
x = df[:, 1];
y = df[:, 2];

# Making a custom kernel with two parameters.

k = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(1.0)), 1.0)

# Instantiating a Gaussian Process with the given kernel `k`.

f = GP(k)

# Instantiating a `FiniteGP`, a finite dimentional projection at the inputs of the dataset observed under Gaussian Noise with $\sigma = 0.001$ .

fx = f(x, 0.001)

# Data's log-likelihood w.r.t prior `GP`. 

logpdf(fx, y)

# Calculating the exact posterior with the given expected output `y` and `FiniteGP`. THe GP's kernel currently has fixed parameters. 

p_fx = posterior(fx, y)

# Data's log-likelihood w.r.t exact posterior `GP`. We see that it drastically increases.

logpdf(p_fx(x), y)

# Plotting the functions sampled from the exact posterior `p_fx` along with the data points.

plt = scatter(x, y, label = "Data")
sampleplot!(plt, p_fx(sort(x), 0.001), 100, alph=0.1)

# # Elliptical Slice Sampler

# Previously, we computed the the exact posterior GP without tuning the kernel parameters and achieved a loglikelihood on exact posterior of $-1.285$. We now attempt get a better posterior by sampling for kernel parameters using Elliptical Slice Sampler provided by [EllipticalSliceSampling.jl](https://github.com/TuringLang/EllipticalSliceSampling.jl/)

# We start of by loading necessary packages.

using EllipticalSliceSampling, Distributions

# We define a function which returns log-likelihood of of data w.r.t an exact posterior with given set of kernel parameters.

function logp(params)
    exp_params = exp.(params)
    kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_params[1])), exp_params[2])
    f = GP(kernel)
    fx = f(x, 0.1)
    p_fx = posterior(fx, y)
    return logpdf(p_fx(x, 0.1), y)
end

# We define a Gaussian prior over the joint distribution on kernel parameters space. Since we have only two parameters, we define a multi-variate Gaussian of dimension two.

prior = MvNormal(zeros(2), ones(2) .* 1)

# Sanity check for the defined `logp` function and `prior` distribution.

logp(rand(prior))

# Sampling 2,000 samples using `ESS_mcmc` provided by `EllipticalSliceSampling.jl`. 

samples = ESS_mcmc(prior, logp, 2_000);
samples_mat = hcat(samples...)';

# Plotting a histogram of the samples for the two parameters. The vertical line in each graph indicates the mean of the samples.

plt = histogram(samples_mat, layout=2, labels= "Param")
vline!(plt, mean(samples_mat, dims=1), layout=2, label="Mean")

# Mean of samples of both the parameters.

mean_params = mean(samples_mat, dims=1)

# Conditional log-probability of exact posterior with kernel's parameters tuned using ESS. We can observe that there is significant improvement over exact posterior with default kernel parameters. 

logp(mean_params)

# Plotting sampled functions from posterior with tuned parameters

# +

plt = scatter(x, y, label="data")

exp_mean_params = exp.(mean_params)
kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_mean_params[1])), exp_mean_params[2])
f = GP(kernel)
fx = f(x, 0.1)
p_fx = posterior(fx, y)
sampleplot!(plt, p_fx(sort(x), 0.001), 100, alph=0.1)
plt
# -


