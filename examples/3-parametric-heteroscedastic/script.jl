# # Parametric Heteroscedastic Model
#
# This example is a small extension of the standard GP regression problem, in which the
# observation noise variance is a function of the input.
# It is assumed to be a simple quadratic form, with a single unknown
# scaling parameter, in addition to the usual lengthscale and variance parameters
# of the GP.
# A point estimate of all parameters is obtained using type-II maximum likelihood,
# as per usual.

using AbstractGPs
using AbstractGPsMakie
using CairoMakie
using KernelFunctions
using Optim
using ParameterHandling
using Zygote

using LinearAlgebra
using Random
Random.seed!(1234)  # setting the seed for reproducibility of this notebook
#md nothing #hide

# Specify simple GP:
build_gp(θ) = GP(0, θ.s * with_lengthscale(SEKernel(), θ.l));

# Observation variance is some scaling of $x^2$:
observation_variance(θ, x::AbstractVector{<:Real}) = Diagonal(θ.σ² .* x .^ 2);

# Specify hyperparameters:
const flat_θ, unflatten = ParameterHandling.value_flatten((
    s=positive(1.0), l=positive(3.0), σ²=positive(0.1)
));
θ = unflatten(flat_θ);

# Build inputs:
const x = range(0.0, 10.0; length=100)
const y = rand(build_gp(θ)(x, observation_variance(θ, x)));

# We specify the objective function:
function objective(flat_θ)
    θ = unflatten(flat_θ)
    f = build_gp(θ)
    Σ = observation_variance(θ, x)
    return -logpdf(f(x, Σ), y)
end;

# We use L-BFGS for optimising the objective function.
# It is a first-order method and hence requires computing the gradient of the objective function.
# We do not derive and implement the gradient function manually here but instead use reverse-mode automatic differentiation with Zygote.
# When computing gradients with Zygote, the objective function is evaluated as well.
# We can exploit this and [avoid re-evaluating the objective function](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations) in such cases.
function objective_and_gradient(F, G, flat_θ)
    if G !== nothing
        val_grad = Zygote.withgradient(objective, flat_θ)
        copyto!(G, only(val_grad.grad))
        if F !== nothing
            return val_grad.val
        end
    end
    if F !== nothing
        return objective(flat_θ)
    end
    return nothing
end;

# Optimise the hyperparameters. They've been initialised near the correct values, so
# they ought not to deviate too far.
flat_θ_init = flat_θ + 0.01 * randn(length(flat_θ))
result = optimize(
    Optim.only_fg!(objective_and_gradient),
    flat_θ_init,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; show_every=100),
)
θ_final = unflatten(result.minimizer)

# Construct the posterior GP with the optimal model parameters:
Σ_obs_final = observation_variance(θ_final, x)
fx_final = build_gp(θ_final)(x, Σ_obs_final)
f_post = posterior(fx_final, y);

# Plot the results, making use of [AbstractGPsMakie](https://github.com/JuliaGaussianProcesses/AbstractGPsMakie.jl):
using CairoMakie.Makie.ColorSchemes: Set1_4

with_theme(
    Theme(;
        palette=(color=Set1_4,),
        patchcolor=(Set1_4[2], 0.2),
        Axis=(limits=((0, 10), nothing),),
    ),
) do
    ## Fix numerical issues when computing the Cholesky decomposition of the covariance matrix
    ## of the finite projection of the posterior GP by introducing artifical noise
    f_post_jitter = f_post(x, 1e-8)

    plot(
        x,
        f_post(x, Σ_obs_final);
        bandscale=3,
        label="posterior + noise",
        color=(:orange, 0.3),
    )
    plot!(x, f_post_jitter; bandscale=3, label="posterior")
    gpsample!(x, f_post_jitter; samples=10, color=Set1_4[3])
    scatter!(x, y; label="y")
    axislegend()
    current_figure()
end
