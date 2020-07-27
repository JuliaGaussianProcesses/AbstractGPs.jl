# -*- coding: utf-8 -*-
# # Example: Approximate Inference with NUTS-HMC

# Loading the necessary packages and setting seed.

using AbstractGPs, KernelFunctions, Plots, Random
Random.seed!(1234);

# Load toy regression 
# [dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv) 
# taken from GPFlow examples.

x = [0.8658165855998895, 0.6661700880180962, 0.8049218148148531, 0.7714303440386239, 
    0.14790478354654835, 0.8666105548197428, 0.007044577166530286, 0.026331737288148638, 
    0.17188596617099916, 0.8897812990554013, 0.24323574561119998, 0.028590102134105955];
y = [1.5255314337144372, 3.6434202968230003, 3.010885733911661, 3.774442382979625, 
    3.3687639483798324, 1.5506452040608503, 3.790447985799683, 3.8689707574953, 
    3.4933565751758713, 1.4284538820635841, 3.8715350915692364, 3.7045949061144983];
scatter(x, y; xlabel="x", ylabel="y")

# Split the observations into train and test set.

(x_train, y_train) = (x[begin:8], y[begin:8]);
(x_test, y_test) = (x[9:end], y[9:end]);

# Instantiate the kernel.

k = Matern52Kernel()

# Instantiate a Gaussian Process with the given kernel `k`.

f = GP(k)

# Instantiate a `FiniteGP`, a finite dimentional projection at the inputs of the dataset 
# observed under Gaussian Noise with $\sigma = 0.001$ .

fx = f(x_train, 0.001)

# Data's log-likelihood w.r.t prior `GP`. 

logpdf(fx, y_train)

# Calculating the exact posterior over `f` given `y`. The GP's kernel currently has 
# some arbitrary fixed parameters. 

p_fx = posterior(fx, y_train)

# Data's log-likelihood under the posterior `GP`. We see that it drastically increases.

logpdf(p_fx(x_test), y_test)

# Plot the posterior `p_fx` along with the observations.

plt = scatter(x_train, y_train, label="Train Data")
scatter!(plt, x_test, y_test, label="Test Data")
plot!(plt, p_fx, 0:0.001:1; label="Posterior")

# # Hamiltonian Monte Carlo Sampler

# Previously we computed the log likelihood of the untuned kernel parameters of the GP, 
# $-1.285$. We now also perform approximate inference over said kernel parameters using 
# the No-U-Turn Sampler - Hamiltonian Monte Carlo(NUTS-HMC) Sampler 
# ([paper](http://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf))
# provided by [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl/).
# We start of by loading necessary packages.

using AdvancedHMC, Distributions, ForwardDiff

# We define a function which returns log-probability of the data under the 
# GP / log-likelihood of the parameters of the GP.

function logp(params; x=x_train, y=y_train)
    kernel = ScaledKernel(
        transform(
            Matern52Kernel(), 
            ScaleTransform(exp(params[1]))
        ), 
        exp(params[2])
    )
    f = GP(kernel)
    fx = f(x, 0.01)
    return logpdf(fx, y)
end

# Set the number of samples to draw and warmup iterations

n_samples, n_adapts = 2_000, 1_000
initial_params = rand(2)

# Define a Hamiltonian system

metric = DiagEuclideanMetric(2)
hamiltonian = Hamiltonian(metric, logp, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically

initial_ϵ = find_good_stepsize(hamiltonian, initial_params)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix

proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))


# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample

samples, stats = sample(
    hamiltonian, 
    proposal, 
    initial_params, 
    n_samples, 
    adaptor, 
    n_adapts; 
    progress=false
)
samples_mat = reduce(hcat, samples)'; 

# Mean of samples of both the parameters.

mean_params = mean(samples_mat; dims=1)

# Plotting a histogram of the samples for the two parameters. 
# The vertical line in each graph indicates the mean of the samples.

plt = histogram(samples_mat; layout=2, labels= "Param")
vline!(plt, mean_params; layout=2, label="Mean")

# Average log-marginal-probability of data with posterior kernel parameter samples 
# sampled using ESS. We can observe that there is significant improvement over exact 
# posterior with default kernel parameters.

mean(logp(param; x=x_test, y=y_test) for param in samples)

# Plotting sampled functions from posterior with tuned parameters

plt = scatter(x_train, y_train, label="Train Data")
scatter!(plt, x_test, y_test, label="Test Data")
for params in @view(samples[(end-100):end,:])
    opt_kernel = ScaledKernel(
        transform(
            Matern52Kernel(), 
            ScaleTransform(exp(params[1]))
        ), 
        exp(params[2])
    )
    f = GP(opt_kernel)
    p_fx = posterior(f(x_train, 0.1), y_train)
    sampleplot!(plt, p_fx(collect(0:0.02:1)), 1)
end
plt


