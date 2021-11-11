# # Mauna Loa time series example
#
# In this notebook, we apply Gaussian process regression to the Mauna Loa CO₂
# dataset. This showcases a rich combination of kernels, and how to handle and
# optimize all their parameters.

# ## Setup
#
# We make use of the following packages:

using CSV, DataFrames  # data loading
using AbstractGPs  # exact GP regression
using ParameterHandling  # for nested and constrained parameters
using Optim  # optimization
using Zygote  # auto-diff gradient computation
using Plots  # visualisation

# Let's load and visualize the dataset:

data = CSV.read(joinpath(@__DIR__, "CO2_data.csv"), Tables.matrix; header=0)
year = data[:, 1]
co2 = data[:, 2]

## We split the data into training and testing set:
idx_train = year .< 2004
xtrain = year[idx_train]
ytrain = co2[idx_train]
idx_test = .!idx_train
xtest = year[idx_test]
ytest = co2[idx_test]

function plotdata()
    plot(; xlabel="year", ylabel="CO₂ [ppm]", legend=:bottomright)
    scatter!(xtrain, ytrain; label="training data", ms=2, markerstrokewidth=0)
    return scatter!(xtest, ytest; label="test data", ms=2, markerstrokewidth=0)
end

plotdata()

# ## Prior
#
# We will model this dataset using a sum of several kernels which describe
#
# - smooth trend: squared exponential kernel with long lengthscale;
# - seasonal component: periodic covariance function with period of one year,
#   multiplied with a squared exponential kernel to allow decay away from exact
#   periodicity;
# - medium-term irregularities: rational quadratic kernel;
# - noise terms: squared exponential kernel with short lengthscale
#   and uncorrelated observation noise.
#
# For more details, see [Rasmussen & Williams (2005), chapter 5](http://www.gaussianprocess.org/gpml/chapters/RW5.pdf).

# We will use
# [ParameterHandling.jl](https://invenia.github.io/ParameterHandling.jl/) for
# handling the (hyper)parameters of our model. It provides functions such as
# `positive` with which we can put constraints on the hyperparameters, and
# allows us to represent all required parameters as a nested NamedTuple:

#! format: off
## initial values to match http://stor-i.github.io/GaussianProcesses.jl/latest/mauna_loa/
θ_init = (;
    se_long = (;
        σ = positive(exp(4.0)),
        ℓ = positive(exp(4.0)),
    ),
    seasonal = (;
        ## product kernels only need a single overall signal variance
        per = (;
            ℓ = positive(exp(0.0)),  # relative to period!
            p = fixed(1.0),  # 1 year, do not optimize over
        ),
        se = (;
            σ = positive(exp(1.0)),
            ℓ = positive(exp(4.0)),
        ),
    ),
    rq = (;
        σ = positive(exp(0.0)),
        ℓ = positive(exp(0.0)),
        α = positive(exp(-1.0)),
    ),
    se_short = (;
        σ = positive(exp(-2.0)),
        ℓ = positive(exp(-2.0)),
    ),
    noise_scale = positive(exp(-2.0)),
)
#! format: on
#md nothing #hide

# We define a couple of helper functions to simplify the kernel construction:

SE(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ)
## PeriodicKernel is broken, see https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/389
##Per(θ) = with_lengthscale(PeriodicKernel(; r=[θ.ℓ/2]), θ.p)  # NOTE- discrepancy with GaussianProcesses.jl
Per(θ) = with_lengthscale(SqExponentialKernel(), θ.ℓ) ∘ PeriodicTransform(1 / θ.p)
RQ(θ) = θ.σ^2 * with_lengthscale(RationalQuadraticKernel(; α=θ.α), θ.ℓ)
#md nothing #hide

# This allows us to write a function that, given the nested tuple of parameter values, constructs the GP prior:

function build_gp_prior(θ)
    k_smooth_trend = SE(θ.se_long)
    k_seasonality = Per(θ.seasonal.per) * SE(θ.seasonal.se)
    k_medium_term_irregularities = RQ(θ.rq)
    k_noise_terms = SE(θ.se_short) + θ.noise_scale^2 * WhiteKernel()
    kernel = k_smooth_trend + k_seasonality + k_medium_term_irregularities + k_noise_terms
    return GP(kernel)  # [`ZeroMean`](@ref) mean function by default
end
#md nothing #hide

# ## Posterior
#
# To construct the posterior, we need to first build a [`FiniteGP`](@ref),
# which represents the infinite-dimensional GP at a finite number of input
# features:

function build_finite_gp(θ)
    f = build_gp_prior(θ)
    return f(xtrain)
end
#md nothing #hide

# !!! info "`WhiteKernel` vs `FiniteGP` observation noise"
#     In this notebook, we already included observation noise through the
#     `WhiteKernel` as part of the GP prior covariance in `build_gp_prior`. We
#     therefore call `f(xtrain)` which implies zero (additional) observation
#     noise.
#
#     Alternatively, we could have omitted the `θ.noise_scale^2 *
#     WhiteKernel()` term and instead passed the noise variance as a second
#     argument to the GP call in `build_finite_gp`, `f(xtrain,
#     θ.noise_scale^2)`.
#
#     These two approaches have slightly different semantics: In the first one,
#     the `WhiteKernel` contributes non-zero variance to the `[i, j]` element
#     of the covariance matrix of the `FiniteGP` if `xtrain[i] == xtrain[j]`
#     (based on the values of the features). In the second one, the observation
#     noise variance passed to `FiniteGP` only contributes to the diagonal
#     elements of the covariance matrix, i.e. for `i == j`.
#
#     Moreover, the variance (uncertainty) of the posterior predictions
#     includes the variance from the `WhiteKernel`, but does not include the
#     variance of the observation noise passed to the `FiniteGP`. To include
#     the observation noise in posterior predictions from the second approach,
#     call `fpost_opt(xtest, noise_scale^2)`.
#
# !!! tip
#     In practice, we recommend that you pass in observation noise to the
#     `FiniteGP`, and omit the explicit `WhiteKernel`.

# We obtain the posterior, conditioned on the (finite) observations, by calling
# [`posterior`](@ref):

function build_posterior_gp(θ)
    fx = build_finite_gp(θ)
    return posterior(fx, ytrain)
end
#md nothing #hide

# Now we can put it all together to obtain a [`PosteriorGP`](@ref).
# The call to `ParameterHandling.value` is required to replace the constraints
# (such as `positive` in our case) with concrete numbers:

fpost_init = build_posterior_gp(ParameterHandling.value(θ_init))

# Let's visualize what the GP fitted to the data looks like, for the initial choice of kernel hyperparameters.

plot_gp!(f; label) = plot!(f(1920:0.2:2030); ribbon_scale=2, linewidth=1, label)

let
    ## The `let` block creates a new scope, so any utility variables we define in here won't leak outside.
    ## The return value of this block is given by its last expression.
    plotdata()
    plot_gp!(fpost_init; label="posterior f(⋅)")  ## this returns the current plot object
end  ## and so the plot object will be shown

# A reasonable fit to the data, but awful extrapolation away from the observations!

# ## Hyperparameter Optimization
#
# To improve the fit, we want to maximize the (log) marginal likelihood with
# respect to the hyperparameters.
# [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) expects to
# minimize a loss, so we define it as the negative log marginal likelihood:

function loss(θ)
    fx = build_finite_gp(θ)
    lml = logpdf(fx, ytrain)  # this computes the log marginal likelihood
    return -lml
end
#md nothing #hide

# !!! note "Work-in-progress"
#     In the future, we are planning to provide the `optimize_loss` utility
#     function as part of JuliaGaussianProcesses -- for now, we just define it
#     inline.
#
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
        if F !== nothing && G !== nothing
            val, grad = Zygote.withgradient(loss_packed, x)
            G .= only(grad)
            return val
        elseif G !== nothing
            grad = Zygote.gradient(loss_packed, x)
            G .= only(grad)
            return nothing
        elseif F !== nothing
            return loss_packed(x)
        end
    end

    result = optimize(Optim.only_fg!(fg!), θ_flat_init, optimizer, options; inplace=false)

    return unflatten(result.minimizer), result
end
#md nothing #hide

# We now run the optimization:

θ_opt, result = optimize_loss(loss, θ_init)
result

# The final value of the log marginal likelihood is:

-result.minimum

# !!! warning
#     To avoid bad local optima, we could (and should) have carried out several
#     random restarts with different initial values for the hyperparameters,
#     and then picked the result with the highest marginal likelihood. We omit
#     this for simplicity. For more details on how to fit GPs in practice,
#     check out [A Practical Guide to Gaussian
#     Processes](https://tinyurl.com/guide2gp).
#
# Let's construct the posterior GP with the optimized hyperparameters:

fpost_opt = build_posterior_gp(ParameterHandling.value(θ_opt))
#md nothing #hide

# This is the kernel with the point-estimated hyperparameters:

fpost_opt.prior.kernel

# Let's print the optimized values of the hyperparameters in a more helpful format:

# !!! note "Work-in-progress"
#     This is another utility function we would eventually like to move out of this notebook:

using Printf

function show_params(nt::NamedTuple, pre=0)
    res = ""
    for (s, v) in pairs(nt)
        if typeof(v) <: NamedTuple
            res *= join(fill(" ", pre)) * "$(s):\n" * show_params(v, pre+4)
        else
            res *= join(fill(" ", pre)) * "$s = $(@sprintf("%.3f", v))\n"
        end
    end
    return res
end

print(show_params(ParameterHandling.value(θ_opt)))

# And, finally, we can visualize our optimized posterior GP:

let
    plotdata()
    plot_gp!(fpost_opt; label="optimized posterior f(⋅)")
end
