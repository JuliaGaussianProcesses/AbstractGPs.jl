# # Deep Kernel Learning with Flux
# ## Package loading
# We use a couple of useful packages to plot and optimize
# the different hyper-parameters
using KernelFunctions
using Flux
using Distributions, LinearAlgebra
using Plots
using AbstractGPs
default(; legendfontsize=15.0, linewidth=3.0);

# ## Data creation
# We create a simple 1D Problem with very different variations

xmin, xmax = (-3, 3)  # Limits
N = 150
noise_std = 0.01
x_train = collect(eachrow(rand(Uniform(xmin, xmax), N))) # Training dataset
target_f(x) = sinc(abs(x)^abs(x)) # We use sinc with a highly varying value
target_f(x::AbstractArray) = target_f(only(x))
y_train = target_f.(x_train) + randn(N) * noise_std
x_test = collect(eachrow(range(xmin, xmax; length=200))) # Testing dataset

plot(xmin:0.01:xmax, target_f; label="ground truth")
scatter!(map(only, x_train), y_train; label="training data")

# ## Model definition
# We create a neural net with 2 layers and 10 units each.
# The data is passed through the NN before being used in the kernel.
neuralnet = Chain(Dense(1, 20), Dense(20, 30), Dense(30, 5))

# We use the Squared Exponential Kernel:
k = SqExponentialKernel() ∘ FunctionTransform(neuralnet)

# We now define our model:
gpprior = GP(k)  # GP Prior
fx = AbstractGPs.FiniteGP(gpprior, x_train, noise_std^2)  # Prior at the observations
fp = posterior(fx, y_train)  # Posterior of f given the observations

# This computes the log evidence of `y`, which is going to be used as the objective:
loss(y) = -logpdf(fx, y)

@info "Initial loss = $(loss(y_train))"

# Flux will automatically extract all the parameters of the kernel
ps = Flux.params(k)

# We show the initial prediction with the untrained model
p_init = plot(; title="Loss = $(round(loss(y_train); sigdigits=6))")
plot!(vcat(x_test...), target_f; label="true f")
scatter!(vcat(x_train...), y_train; label="data")
pred = marginals(fp(x_test))
plot!(vcat(x_test...), mean.(pred); ribbon=std.(pred), label="Prediction")

# ## Training
anim = Animation()
nmax = 1000
opt = Flux.ADAM(0.1)
for i in 1:nmax
    global grads = gradient(ps) do
        loss(y_train)
    end
    Flux.Optimise.update!(opt, ps, grads)
    if i % 10 == 0
        L = loss(y_train)
        @info "$i/$nmax; loss = $L"
        p = plot(; title="iteration $i/$nmax: loss = $(round(L; sigdigits=6))")
        plot!(vcat(x_test...), target_f; label="true f")
        scatter!(vcat(x_train...), y_train; label="data")
        pred = marginals(posterior(fx, y_train)(x_test))
        plot!(vcat(x_test...), mean.(pred); ribbon=std.(pred), label="Prediction")
        frame(anim)
        display(p)
    end
end
gif(anim, "train-dkl.gif"; fps=5)
nothing #hide

# ![](train-dkl.gif)
