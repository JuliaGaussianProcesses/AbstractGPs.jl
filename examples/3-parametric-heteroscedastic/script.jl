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
using DifferentiationInterface
using KernelFunctions
using Mooncake
using Optim
using ParameterHandling

using LinearAlgebra
using Random
Random.seed!(42)  # setting the seed for reproducibility of this notebook
#md nothing #hide

# In this example we work with a simple GP with a Gaussian kernel and heteroscedastic observation variance.
observation_variance(θ, x::AbstractVector{<:Real}) = Diagonal(0.01 .+ θ.σ² .* x .^ 2)
function build_gpx(θ, x::AbstractVector{<:Real})
    Σ = observation_variance(θ, x)
    return GP(0, θ.s * with_lengthscale(SEKernel(), θ.l))(x, Σ)
end;

# We specify the following hyperparameters:
const flat_θ, unflatten = ParameterHandling.value_flatten((
    s=positive(1.0), l=positive(3.0), σ²=positive(0.1)
));
θ = unflatten(flat_θ);

# We generate some observations:
const x = 0.0:0.1:10.0
const y = rand(build_gpx(θ, x));

# We specify the objective function:
function objective(flat_θ)
    θ = unflatten(flat_θ)
    fx = build_gpx(θ, x)
    return -logpdf(fx, y)
end;

# We use L-BFGS for optimising the objective function.
# It is a first-order method and hence requires computing the gradient of the objective function.
# We do not derive and implement the gradient function manually here but instead use reverse-mode automatic differentiation with DifferentiationInterface + Mooncake.
# When computing gradients, the objective function is evaluated as well.
# We can exploit this and [avoid re-evaluating the objective function](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations) in such cases.
backend = AutoMooncake()
function objective_and_gradient(F, G, flat_θ)
    if G !== nothing
        val = objective(flat_θ)
        grad = only(gradient(objective, backend, flat_θ))
        copyto!(G, grad)
        if F !== nothing
            return val
        end
    end
    if F !== nothing
        return objective(flat_θ)
    end
    return nothing
end;

# We optimise the hyperparameters using initializations close to the values that the observations were generated with.
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

# The optimal model parameters are:
θ_final = unflatten(result.minimizer)

# We compute the posterior GP with these optimal model parameters:
fx_final = build_gpx(θ_final, x)
f_post = posterior(fx_final, y);

# We visualize the results with [AbstractGPsMakie](https://github.com/JuliaGaussianProcesses/AbstractGPsMakie.jl):
using CairoMakie.Makie.ColorSchemes: Set1_4

with_theme(
    Theme(;
        palette=(color=Set1_4,),
        patchcolor=(Set1_4[2], 0.2),
        Axis=(limits=((0, 10), nothing),),
    ),
) do
    plot(
        x,
        f_post(x, observation_variance(θ_final, x));
        bandscale=3,
        label="posterior + noise",
        color=(:orange, 0.3),
    )
    plot!(x, f_post; bandscale=3, label="posterior")
    gpsample!(x, f_post; samples=10, color=Set1_4[3])
    scatter!(x, y; label="y")
    axislegend()
    current_figure()
end
