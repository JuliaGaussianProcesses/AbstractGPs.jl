# # Deep Kernel Learning with Flux
# ## Package loading
# We use a couple of useful packages to plot and optimize
# the different hyper-parameters
using KernelFunctions
using Flux
using Distributions, LinearAlgebra
using Plots
using ProgressMeter
using AbstractGPs
default(; legendfontsize=15.0, linewidth=3.0);

# ## Data creation
# We create a simple 1D Problem with very different variations
xmin = -3;
xmax = 3; # Limits
N = 150
noise = 0.01
x_train = collect(eachrow(rand(Uniform(xmin, xmax), N))) # Training dataset
target_f(x) = sinc(abs(x)^abs(x)) # We use sinc with a highly varying value
target_f(x::AbstractArray) = target_f(first(x))
y_train = target_f.(x_train) + randn(N) * noise
x_test = collect(eachrow(range(xmin, xmax; length=200))) # Testing dataset
spectral_mixture_kernel()
# ## Model definition
# We create a neural net with 2 layers and 10 units each
# The data is passed through the NN before being used in the kernel
neuralnet = Chain(Dense(1, 20), Dense(20, 30), Dense(30, 5))
# We use two cases :
# - The Squared Exponential Kernel
k = transform(SqExponentialKernel(), FunctionTransform(neuralnet))

# We use AbstractGPs.jl to define our model
gpprior = GP(k) # GP Prior
fx = AbstractGPs.FiniteGP(gpprior, x_train, noise) # Prior on f
fp = posterior(fx, y_train) # Posterior of f

# This compute the log evidence of `y`,
# which is going to be used as the objective
loss(y) = -logpdf(fx, y)

@info "Init Loss = $(loss(y_train))"

# Flux will automatically extract all the parameters of the kernel
ps = Flux.params(k)

# We show the initial prediction with the untrained model
p_init = Plots.plot(
    vcat(x_test...), target_f; lab="true f", title="Loss = $(loss(y_train))"
)
Plots.scatter!(vcat(x_train...), y_train; lab="data")
pred = marginals(fp(x_test))
Plots.plot!(vcat(x_test...), mean.(pred); ribbon=std.(pred), lab="Prediction")
# ## Training
anim = Animation()
nmax = 1000
opt = Flux.ADAM(0.1)
@showprogress for i in 1:nmax
    global grads = gradient(ps) do
        loss(y_train)
    end
    Flux.Optimise.update!(opt, ps, grads)
    if i % 100 == 0
        @info "$i/$nmax"
        L = loss(y_train)
        # @info "Loss = $L"
        p = Plots.plot(
            vcat(x_test...), target_f; lab="true f", title="Loss = $(loss(y_train))"
        )
        p = Plots.scatter!(vcat(x_train...), y_train; lab="data")
        pred = marginals(posterior(fx, y_train)(x_test))
        Plots.plot!(vcat(x_test...), mean.(pred); ribbon=std.(pred), lab="Prediction")
        frame(anim)
        display(p)
    end
end
gif(anim; fps=5)
