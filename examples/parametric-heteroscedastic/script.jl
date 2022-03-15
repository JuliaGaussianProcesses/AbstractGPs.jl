# Parameteric Heteroscedsatic

using AbstractGPs
using AbstractGPsMakie
using CairoMakie
using KernelFunctions
using LinearAlgebra
using Optim
using ParameterHandling
using Random
using Zygote

# Specify simple GP.
build_gp(θ) = GP(0, θ.s * with_lengthscale(SEKernel(), θ.l))
observation_variance(θ, x::AbstractVector{<:Real}) = Diagonal(θ.σ² .* x.^2)

# Specify hyperparameters.
flat_init_params, unflatten = ParameterHandling.value_flatten((
    s=positive(1.0),
    l=positive(3.0),
    σ²=positive(0.1),
));
θ_init = unflatten(flat_init_params);

# Build inputs.
const x = range(0.0, 10.0; length=100)
const y = rand(Xoshiro(123456), build_gp(θ_init)(x, observation_variance(θ_init, x)))

function objective(θ)
    f = build_gp(θ)
    Σ = observation_variance(θ, x)
    return -logpdf(f(x, Σ), y)
end

# Optimise the hyperparameters. They've been initialised to the correct values, so
# they ought not to deviate too far.
result = optimize(
    objective ∘ unflatten,
    θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
    flat_init_params,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=4_000);
    inplace=false,
);
θ_final = unflatten(result.minimizer);

# Construct the posterior GP with the optimal model parameters.
Σ_obs_final = observation_variance(θ_final, x);
fx_final = build_gp(θ_final)(x, Σ_obs_final);
f_post = posterior(fx_final, y);

# Plot the results, making use of AbstractGPsMakie.
using CairoMakie.Makie.ColorSchemes: Set1_4

set_theme!(
    palette=(color=Set1_4,),
    patchcolor=(Set1_4[2], 0.2),
    Axis=(limits=((0, 10), nothing),),
)

let
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, x, f_post(x, Σ_obs_final); bandscale=3)
    plot!(ax, x, f_post(x, 1e-9); bandscale=3)
    gpsample!(ax, x, f_post(x, 1e-9); samples=10, color=Set1_4[3])
    scatter!(ax, x, y)
    fig
end
