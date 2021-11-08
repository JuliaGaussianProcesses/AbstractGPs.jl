# # Mauna Loa time series example

using CSV, DataFrames
using AbstractGPs
using ParameterHandling
using Optim, Zygote
using Plots

# Let's load and visualize the dataset:

data = CSV.read("CO2_data.csv", Tables.matrix; header=0)
year = data[:, 1]
co2 = data[:, 2]
#md nothing #hide

# Split the data into training and testing data
xtrain = year[year .< 2004]
ytrain = co2[year .< 2004]
xtest = year[year .>= 2004]
ytest = co2[year .>= 2004]
#md nothing #hide

function plotdata()
    plot(; xlabel="year", ylabel="CO2", legend=:bottomright)
    scatter!(xtrain, ytrain; ms=2, label="train")
    return scatter!(xtest, ytest; ms=2, label="test")
end

plotdata()

# We will use ParameterHandling.jl for handling the (hyper)parameters of our model. We represent all required parameters as a nested NamedTuple:

#! format: off
θ_init = (;
    se1 = (;
        σ = positive(exp(4.0)),
        ℓ = positive(exp(4.0)),
    ),
    per = (;
        σ = positive(exp(0.0)),
        ℓ = positive(exp(1.0)),
        p = positive(exp(0.0)),
    ),
    se2 = (;
        σ = positive(exp(4.0)),
        ℓ = positive(exp(0.0)),
    ),
    rq = (;
        σ = positive(exp(0.0)),
        ℓ = positive(exp(0.0)),
        α = positive(exp(-1.0)),
    ),
    se3 = (;
        σ = positive(exp(-2.0)),
        ℓ = positive(exp(-2.0)),
    ),
    noise_scale = positive(exp(-2.0)),
)
#! format: on

# We define a couple of helper functions to simplify the kernel construction:

SE(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ)
## PeriodicKernel is broken, see https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/389
##Per(θ) = θ.σ^2 * with_lengthscale(PeriodicKernel(; r=[θ.ℓ/2]), θ.p)  # NOTE- discrepancy with GaussianProcesses.jl
function Per(θ)
    return θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ) ∘ PeriodicTransform(1 / θ.p)
end
RQ(θ) = θ.σ^2 * with_lengthscale(RationalQuadraticKernel(; α=θ.α), θ.ℓ)

function build_gp_prior(θ)
    ## The kernel is represented as a sum of kernels:
    kernel = SE(θ.se1) + Per(θ.per) * SE(θ.se2) + RQ(θ.rq) + SE(θ.se3)
    return GP(kernel)
end

function build_finite_gp(θ)
    f = build_gp_prior(θ)
    return f(xtrain, θ.noise_scale^2)
end

function build_posterior_gp(θ)
    fx = build_finite_gp(θ)
    return posterior(fx, ytrain)
end

# We can now construct the posterior GP.
# The call to `ParameterHandling.value` is required to replace the constraints (such as `positive`) with concrete numbers:

fpost_init = build_posterior_gp(ParameterHandling.value(θ_init))

# This is what the GP fitted to the data looks like for the initial choice of kernel hyperparameters:

let
    plotdata()
    plot!(fpost_init(1900:0.2:2050))
end

# To improve the fit, we want to maximize the (log) marginal likelihood with respect to the hyperparameters.
# Optim.jl expects to minimize a loss, so we define it as the negative log marginal likelihood:

function loss(θ)
    fx = build_finite_gp(θ)
    lml = logpdf(fx, ytrain)
    return -lml
end

# In the future, we are planning to provide the following utility function as
# part of JuliaGaussianProcesses -- for now, we just define it inline.
# The L-BFGS parameters were chosen because they seem to work well empirically.
# You could also try with the defaults.

default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function optimize_loss(loss, θ_init; optimizer=default_optimizer, maxiter=1_000)
    options = Optim.Options(; iterations=maxiter, show_trace=true)

    θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init)
    loss_packed = loss ∘ unflatten

    ## https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations
    function fg!(F, G, x)
        if F != nothing && G != nothing
            val, grad = Zygote.withgradient(loss_packed, x)
            G .= only(grad)
            return val
        elseif G != nothing
            grad = Zygote.gradient(loss_packed, x)
            G .= only(grad)
            return nothing
        elseif F != nothing
            return loss_packed(x)
        end
    end

    result = optimize(Optim.only_fg!(fg!), θ_flat_init, optimizer, options; inplace=false)

    return unflatten(result.minimizer), result
end

# We now run the optimisation:

θ_opt, result = optimize_loss(loss, θ_init)
result

# The final value of the log marginal likelihood is:

-result.minimum

# We now construct the posterior GP:

fpost_opt = build_posterior_gp(ParameterHandling.value(θ_opt))

# This is the kernel with the point-estimated hyperparameters:

fpost_opt.prior.kernel

# And, finally, this is the visualization of the posterior GP:

let
    plotdata()
    plot!(fpost_opt(1900:0.2:2050))
end
