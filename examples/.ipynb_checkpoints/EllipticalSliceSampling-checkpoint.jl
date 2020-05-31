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

]activate ..

using AbstractGPs, KernelFunctions, Plots, CSV, DataFrames
include("utils.jl")

df = CSV.read("data/regression_1D.csv", header=false);
x = df[:, 1];
y = df[:, 2];

k = ScaledKernel(transform(Matern52Kernel(), ScaleTransform()))
f = GP(k)
fx = f(x, 0.001);

p_fx = posterior(fx, y);

plt = scatter(x, y, label = "Data")
sampleplot!(plt, p_fx(sort(x), 0.001), 100, alph=0.1)

logpdf(fx, y)

logpdf(p_fx(x), y)

# # Elliptical Slice Sampler

using EllipticalSliceSampling, Distributions

function logp(params)
    exp_params = exp.(params)
    kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_params[1])), exp_params[2])
    f = GP(kernel)
    fx = f(x, 0.1)
    p_fx = posterior(fx, y)
    return logpdf(p_fx(x, 0.1), y)
end

prior = MvNormal(zeros(2), ones(2) .* 1)

logp(rand(prior))

samples = ESS_mcmc(prior, logp, 2_000);
samples_mat = hcat(samples...)';

plt = histogram(samples_mat, layout=2, labels= "Param")
vline!(plt, mean(samples_mat, dims=1), layout=2, label="Mean")

mean(samples_mat, dims=1)

logp(mean(samples_mat, dims=1))

# +

plt = scatter(x, y, label="data")
for params in rand(samples, 100)
    exp_params = exp.(params)
    kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_params[1])), exp_params[2])
    f = GP(kernel)
    fx = f(x, 0.1)
    p_fx = posterior(fx, y)
    plot!(plt, p_fx(sort(x)), label="", linewidth=2, alpha=0.1, ribbon=nothing, color="red")
end
plt
# -


