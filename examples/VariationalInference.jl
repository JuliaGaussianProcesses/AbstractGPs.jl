# # Approximate Inference on Sparse GPs using VI
#
# Loading the necessary packages and setting seed.

using AbstractGPs
using Distributions
using Plots
using StatsFuns
using Random
Random.seed!(1234)
nothing #hide

# Load toy regression
# [dataset](https://github.com/GPflow/docs/blob/master/doc/source/notebooks/basics/data/regression_1D.csv)
# taken from GPFlow examples.

x = [0.8658165855998895, 0.6661700880180962, 0.8049218148148531, 0.7714303440386239, 
    0.14790478354654835, 0.8666105548197428, 0.007044577166530286, 0.026331737288148638, 
    0.17188596617099916, 0.8897812990554013, 0.24323574561119998, 0.028590102134105955]
y = [1.5255314337144372, 3.6434202968230003, 3.010885733911661, 3.774442382979625, 
    3.3687639483798324, 1.5506452040608503, 3.790447985799683, 3.8689707574953, 
    3.4933565751758713, 1.4284538820635841, 3.8715350915692364, 3.7045949061144983]
scatter(x, y; xlabel="x", ylabel="y")

# We split the observations into train and test data.

x_train = x[1:8]
y_train = y[1:8]
x_test = x[9:end]
y_test = y[9:end]
nothing #hide

# We instantiate a Gaussian process with a Matern kernel. The kernel has
# fixed variance and length scale parameters of default value 1.

f = GP(Matern52Kernel())

# We create a finite dimentional projection at the inputs of the training dataset
# observed under Gaussian noise with standard deviation $\sigma = 0.1$, and compute the
# log-likelihood of the outputs of the training dataset.

fx = f(x_train, 0.1)
logpdf(fx, y_train)

# We compute the posterior Gaussian process given the training data, and calculate the
# log-likelihood of the test dataset.

p_fx = posterior(fx, y_train)
logpdf(p_fx(x_test), y_test)

# We plot the posterior Gaussian process along with the observations.

plt = scatter(x_train, y_train; title="posterior (default parameters)", label="Train Data")
scatter!(plt, x_test, y_test; label="Test Data")
plot!(plt, p_fx, 0:0.001:1; label="Posterior")

# ## Variational Inference
#
# Sanity check for the Evidence Lower BOund (ELBO) implemented according to
# M. K. Titsias's _Variational learning of inducing variables in sparse Gaussian processes_.

elbo(fx, y_train, f(rand(7)))

# We use the LBFGS algorithm to maximize the given ELBO. It is provided by the Julia
# package [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

using Optim

# We define a function which returns the negative ELBO for different variance and inverse
# lengthscale parameters of the Matern kernel and different pseudo-points. We ensure that
# the kernel parameters are positive with the softplus function
# ```math
# f(x) = \log (1 + \exp x).
# ```

struct NegativeELBO{X,Y}
    x::X
    y::Y
end

function (g::NegativeELBO)(params)
    kernel = ScaledKernel(
        transform(
            Matern52Kernel(), 
            ScaleTransform(softplus(params[1]))
        ), 
        softplus(params[2]),
    )
    f = GP(kernel)
    fx = f(g.x, 0.1)
    return -elbo(fx, g.y, f(@view(params[3:end])))
end

# We randomly initialize the kernel parameters and 5 pseudo points, and minimize the
# negative ELBO with the LBFGS algorithm and obtain the following optimal parameters:

x0 = rand(7)
opt = optimize(NegativeELBO(x_train, y_train), x0, LBFGS())
opt.minimizer

# The optimized value of the inverse lengthscale is

softplus(opt.minimizer[1])

# and of the variance is

softplus(opt.minimizer[2])

# We compute the log-likelihood of the test data for the resulting approximate
# posterior. We can observe that there is a significant improvement over the
# log-likelihood with the default kernel parameters of value 1.

opt_kernel = ScaledKernel(
    transform(
        Matern52Kernel(),
        ScaleTransform(softplus(opt.minimizer[1]))
    ),
    softplus(opt.minimizer[2]),
)
opt_f = GP(opt_kernel)
opt_fx = opt_f(x_train, 0.1)
ap = approx_posterior(VFE(), opt_fx, y_train, opt_f(opt.minimizer[3:end]))
logpdf(ap(x_test), y_test)

# We visualize the approximate posterior with optimized parameters.

plot(ap, 0:0.001:1; label="Approximate Posterior")
scatter!(
    opt.minimizer[3:end], 
    mean(rand(ap(opt.minimizer[3:end], 0.1), 100), dims=2);
    label="Pseudo-points",
)
scatter!(x_train, y_train; label="Train Data")
scatter!(x_test, y_test; label="Test Data")
Plots.current() #hide
