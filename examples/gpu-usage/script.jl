using AbstractGPs
import AbstractGPs: FiniteGP
using StatsFuns
using CUDA
using Optim
using Functors

# %%
# N.B. can't use a ScaledKernel because of the hardcoded Vector -
# https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/299
make_kernel(ℓ_inv) = Matern52Kernel() ∘ ScaleTransform(ℓ_inv)

f = GP(make_kernel(2.0))

x = rand(10)
fx = f(x)
y = rand(fx)

# %%
gpu(f) = fmap(cu, f)

x, y = gpu(x), gpu(y)
fx = fx |> gpu

cov(fx)

# %%
function objective_function(x, y)
    function negative_elbo(params)
        kernel = make_kernel(params[1])
        f = GP(kernel)
        fx = f(x, 0.1)
        z = logistic.(params[2:end])
        fz = f(z, 1e-6)  # "observing" the latent process with some (small) amount of jitter improves numerical stability
        return -elbo(fx, y, fz)
    end
    return negative_elbo
end

# %%
x0 = cu(rand(6))
# TODO: this doesn't work with CUDA.allowscalar(false)
opt = optimize(objective_function(x, y), x0, LBFGS())


# %%
opt_kernel = make_kernel(opt.minimizer[1])
opt_f = GP(opt_kernel)
opt_fx = opt_f(x, 0.1)
ap = approx_posterior(VFE(), opt_fx, y, opt_f(logistic.(opt.minimizer[2:end])))

# %%
using Plots
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
