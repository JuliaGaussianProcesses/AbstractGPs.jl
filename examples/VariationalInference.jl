# # Example: Approximate Inference on Sparse GPs using VI

# Loading necessary packages.

using AbstractGPs, Plots, KernelFunctions

# Loading [toy regression dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv) taken from GPFlow examples.

x = [0.8658165855998895, 0.6661700880180962, 0.8049218148148531, 0.7714303440386239, 0.14790478354654835, 0.8666105548197428, 0.007044577166530286, 0.026331737288148638, 0.17188596617099916, 0.8897812990554013, 0.24323574561119998, 0.028590102134105955];
y = [1.5255314337144372, 3.6434202968230003, 3.010885733911661, 3.774442382979625, 3.3687639483798324, 1.5506452040608503, 3.790447985799683, 3.8689707574953, 3.4933565751758713, 1.4284538820635841, 3.8715350915692364, 3.7045949061144983];
scatter(x, y, xlabel="x", ylabel="y")

# Making a custom kernel with two parameters.

k = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(1.0)), 1.0)

# Instantiating a Gaussian Process with the given kernel `k`.

f = GP(k);

# Instantiating a `FiniteGP`, a finite dimentional projection at the inputs of the dataset observed under Gaussian Noise with $\sigma = 0.001$ .

fx = f(x, 0.001);

# Sanity checking the Evidence Lower BOund (ELBO) implemented according to M. K. Titsias. _Variational learning of inducing variables in sparse Gaussian processes_

elbo(fx, y, f(rand(7)))

# We will be using `Optim.jl` package's `LBFGS` algorithm to maximize the given ELBO.

using Optim

# Creatign a helper function which would be used for optimization. It takes in the parameters (both variational and model parameters) and returns the negative of the ELBO.  

function optim_function(params)
    kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp.(params[1]))), exp.(params[2]))
    f = GP(kernel)
    fx = f(x, 0.1)
    return -elbo(fx, y, f(params[3:end]))
end

# Initializing the parameters

x0 = rand(7)

# Sanity testing the helper function.

optim_function(x0)

# Optimizing using `Optim.jl` package's `LBFGS` algorithm.

opt = optimize(optim_function, x0, LBFGS())

# Optimal negative ELBO:

optim_function(opt.minimizer)

# Visualizing the posterior.

opt_kernel = ScaledKernel(transform(Matern52Kernel(), ScaleTransform(exp.(opt.minimizer[1]))), exp.(opt.minimizer[2]))
opt_f = GP(opt_kernel)
opt_fx = opt_f(x, 0.1)
ap = approx_posterior(VFE(), opt_fx, y, opt_f(opt.minimizer[3:end]));

plt = plot(ap, 0:0.001:1, label="Approx Posterior")
scatter!(plt, opt.minimizer[3:end], mean(rand(ap(opt.minimizer[3:end], 0.1), 100), dims=2), label="Pseudo-points")
scatter!(plt, x, y, label="Data")
plt


