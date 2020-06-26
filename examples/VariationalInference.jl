# # Example: Approximate Inference on Sparse GPs using VI

# Loading the necessary packages and setting seed.

using AbstractGPs, KernelFunctions, Plots, Random
pyplot()
Random.seed!(1234);

# Loading toy regression 
# [dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv) 
# taken from GPFlow examples.

x = [0.8658165855998895, 0.6661700880180962, 0.8049218148148531, 0.7714303440386239, 
    0.14790478354654835, 0.8666105548197428, 0.007044577166530286, 0.026331737288148638, 
    0.17188596617099916, 0.8897812990554013, 0.24323574561119998, 0.028590102134105955];
y = [1.5255314337144372, 3.6434202968230003, 3.010885733911661, 3.774442382979625, 
    3.3687639483798324, 1.5506452040608503, 3.790447985799683, 3.8689707574953, 
    3.4933565751758713, 1.4284538820635841, 3.8715350915692364, 3.7045949061144983];
scatter(x, y, xlabel="x", ylabel="y")

# Split the observations into train and test set.

(x_train, y_train) = (x[begin:8], y[begin:8]);
(x_test, y_test) = (x[9:end], y[9:end]);

# Instantiate the kernel.

k = Matern52Kernel()

# Instantiate a Gaussian Process with the given kernel `k`.

f = GP(k);

# Instantiate a `FiniteGP`, a finite dimentional projection at the inputs of the dataset 
# observed under Gaussian Noise with $\sigma = 0.001$ .

fx = f(x_train, 0.001);

# Calculating the exact posterior over `f` given `y`. The GP's kernel currently has some 
# arbitrary fixed parameters. 

p_fx = posterior(fx, y_train)

logpdf(p_fx(x_test), y_test)

# Sanity check for the Evidence Lower BOund (ELBO) implemented according to 
# M. K. Titsias's _Variational learning of inducing variables in sparse Gaussian processes_

elbo(fx, y_train, f(rand(7)))

# We will be using `Optim.jl` package's `LBFGS` algorithm to maximize the given ELBO.

using Optim

# Create a helper function for optimization. It takes in the parameters (both variational 
# and model parameters) and returns the negative of the ELBO.  

function optim_function(params; x=x_train, y=y_train)
    kernel = ScaledKernel(
        transform(
            Matern52Kernel(), 
            ScaleTransform(exp.(params[1]))
            ), 
        exp.(params[2])
    )
    f = GP(kernel)
    fx = f(x, 0.1)
    return -elbo(fx, y, f(params[3:end]))
end

# Initialize the parameters (Varitational and Model parameters)
x0 = rand(7)

# Sanity check for the helper function. We intend to minimize the result of this function.

-optim_function(x0)

# Optimize using `Optim.jl` package's `LBFGS` algorithm.

opt = optimize(optim_function, x0, LBFGS())

# Optimal parameters:

opt.minimizer

# ELBO with optimal parameters. We see that there is significant improvement when compared 
# to the initial parameters.

-optim_function(opt.minimizer; x=x_test, y=y_test)

# Visualize the posterior.

opt_kernel = ScaledKernel(
    transform(
        Matern52Kernel(), 
        ScaleTransform(exp.(opt.minimizer[1]))
        ), 
    exp.(opt.minimizer[2])
)
opt_f = GP(opt_kernel)
opt_fx = opt_f(x_train, 0.1)
ap = approx_posterior(VFE(), opt_fx, y_train, opt_f(opt.minimizer[3:end]));

# Average log-marginal-probability of data with posterior kernel parameter samples sampled 
# using ESS. We can observe that there is significant improvement over exact posterior with 
# default kernel parameters.

logpdf(ap(x_test), y_test)

plt = plot(ap, 0:0.001:1, label="Approx Posterior")
plot!(plt, p_fx, 0:0.001:1, label="Exact Posterior")
scatter!(
    plt, 
    opt.minimizer[3:end], 
    mean(rand(ap(opt.minimizer[3:end], 0.1), 100), dims=2), 
    label="Pseudo-points"
)
scatter!(plt, x_train, y_train, label="Train Data")
scatter!(plt, x_test, y_test, label="Test Data")
plt


