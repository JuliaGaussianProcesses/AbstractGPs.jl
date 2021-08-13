# # A Minimal Working Example of AbstractGPs on CUDA
#
# ## Setup

using AbstractGPs
using CUDA
using Optim
using Functors
using LogExpFunctions

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1234)
#md nothing #hide

# First, a function to create the kernel.
# N.B. we can't use a ScaledKernel because of the hardcoded Vector -
# https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/299

make_kernel(ℓ_inv) = Matern52Kernel() ∘ ScaleTransform(ℓ_inv)
#md nothing #hide

# Next, sample some data from a GP

lik_noise = 0.01
f = GP(make_kernel(1.0))
x = rand(20)
fx = f(x, lik_noise)
y = rand(fx)
#md nothing #hide

# Move everything to the GPU and compute the covariance

gpu(f) = fmap(cu, f)

x, y = gpu(x), gpu(y)
fx = f(x)
cov(fx) isa CuArray

# Create an objective function and optimise the kernel parameters and inducing
# points

jitter = 1e-5
function objective_function(x, y)
    function negative_elbo(params)
        kernel = make_kernel(params[1])
        f = GP(kernel)
        fx = f(x, lik_noise)
        z = logistic.(params[2:end])
        fz = f(z, jitter)  # "observing" the latent process with some (small) amount of jitter improves numerical stability
        return -elbo(fx, y, fz)
    end
    return negative_elbo
end

x0 = cu(rand(6) * 2)
# TODO: this doesn't work with CUDA.allowscalar(false) because of Optim
opt = optimize(objective_function(x, y), x0, LBFGS())
#md nothing #hide

# Construct the approximate posterior using the optimised parameters

opt_kernel = make_kernel(opt.minimizer[1])
opt_f = GP(opt_kernel)
opt_fx = opt_f(x, lik_noise)
ap = approx_posterior(VFE(), opt_fx, y, opt_f(logistic.(opt.minimizer[2:end]), jitter))

# Plot the optimised posterior (requires some casting to and from CuArrays)

scatter(
    Array(x),
    Array(y);
    xlim=(0, 1),
    xlabel="x",
    ylabel="y",
    title="posterior (VI with sparse grid)",
    label="Train Data",
)
x_plot = cu(collect(0:0.001:1))
plot!(x_plot, Array(mean(ap(x_plot))); label=false)
vline!(logistic.(opt.minimizer[3:end]); label="Pseudo-points")
