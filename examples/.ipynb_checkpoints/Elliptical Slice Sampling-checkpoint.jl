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

using AbstractGPs, KernelFunctions, Plots, CSV

df = CSV.read("data/regression_1D.csv", header=false);
x = df[:, 1];
y = df[:, 2];

?transform

k = ScaledKernel(transform(Matern52Kernel(), ScaleTransform()))
f = GP(k)
fx = f(x, 0.001);

p_fx = posterior(fx, y);

plt = scatter(x, y, label = "True Function")
plot!(plt, sort(x), rand(p_fx(sort(x), 0.001)), label="Sampled Function")

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

prior = MvNormal(zeros(2), ones(2) .* 10)

rand(prior)

logp(rand(prior))

ESS_mcmc(prior, logp, 2_000)


