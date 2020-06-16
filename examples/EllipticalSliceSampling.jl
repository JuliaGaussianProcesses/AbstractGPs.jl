# # An example showcasing the usage of EllipticalSliceSampling.jl with AbstractGPs.jl

# Loading the necessary packages.

using AbstractGPs, KernelFunctions, Plots

# Loading [toy regression dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv) taken from GPFlow examples.

x = [0.8658165855998895, 0.6661700880180962, 0.8049218148148531, 0.7714303440386239, 0.14790478354654835, 0.8666105548197428, 0.007044577166530286, 0.026331737288148638, 0.17188596617099916, 0.8897812990554013, 0.24323574561119998, 0.028590102134105955];
y = [1.5255314337144372, 3.6434202968230003, 3.010885733911661, 3.774442382979625, 3.3687639483798324, 1.5506452040608503, 3.790447985799683, 3.8689707574953, 3.4933565751758713, 1.4284538820635841, 3.8715350915692364, 3.7045949061144983];
scatter(x, y, xlabel="x", ylabel="y")

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

# Plotting the posterior `p_fx` along with the data points.

plt = scatter(x, y, label = "Data")
plot!(plt, p_fx, 0:0.001:1, label="Posterior")

# # Elliptical Slice Sampler

# Previously, we computed the the exact posterior GP without tuning the kernel parameters and achieved a loglikelihood on exact posterior of $-1.285$. We now attempt get a better posterior by tuning for kernel parameters using Elliptical Slice Sampler provided by [EllipticalSliceSampling.jl](https://github.com/TuringLang/EllipticalSliceSampling.jl/) instead of computing the exact posterior.

# We start of by loading necessary packages.

using EllipticalSliceSampling, Distributions

# We define a function which returns log-likelihood of of data w.r.t a GP with the given set of kernel parameters.

function logp(params)
    exp_params = exp.(params)
    kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_params[1])), exp_params[2])
    f = GP(kernel)
    fx = f(x, 0.1)
    return logpdf(fx, y)
end

# We define a Gaussian prior over the joint distribution on kernel parameters space. Since we have only two parameters, we define a multi-variate Gaussian of dimension two.

prior = MvNormal(2, 1)

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

# Conditional log-probability of GP with kernel's parameters tuned using ESS. We can observe that there is significant improvement over exact posterior with default kernel parameters. 

logp(mean_params)

# Plotting sampled functions from posterior with tuned parameters


plt = scatter(x, y, label="data")
for params in eachrow(samples_mat[end-100:end,:])
    exp_params = exp.(params)
    opt_kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp_params[1])), exp_params[2])
    f = GP(opt_kernel)
    p_fx = posterior(f(x, 0.1), y)
    sampleplot!(plt, p_fx(collect(0:0.02:1)), 1)
end
plt


