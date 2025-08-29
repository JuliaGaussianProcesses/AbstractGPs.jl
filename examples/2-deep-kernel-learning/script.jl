# # Deep Kernel Learning with Lux

## Background

# This example trains a GP whose inputs are passed through a neural network.
# This kind of model has been considered previously [^Calandra] [^Wilson], although it has been shown that some care is needed to avoid substantial overfitting [^Ober].
# In this example we make use of the `FunctionTransform` from [KernelFunctions.jl](github.com/JuliaGaussianProcesses/KernelFunctions.jl/) to put a simple Multi-Layer Perceptron built using Lux.jl inside a standard kernel.

# [^Calandra]: Calandra, R., Peters, J., Rasmussen, C. E., & Deisenroth, M. P. (2016, July). [Manifold Gaussian processes for regression.](https://ieeexplore.ieee.org/abstract/document/7727626) In 2016 International Joint Conference on Neural Networks (IJCNN) (pp. 3338-3345). IEEE.

# [^Wilson]: Wilson, A. G., Hu, Z., Salakhutdinov, R. R., & Xing, E. P. (2016). [Stochastic variational deep kernel learning.](https://proceedings.neurips.cc/paper/2016/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) Advances in Neural Information Processing Systems, 29.

# [^Ober]: Ober, S. W., Rasmussen, C. E., & van der Wilk, M. (2021, December). [The promises and pitfalls of deep kernel learning.](https://proceedings.mlr.press/v161/ober21a.html) In Uncertainty in Artificial Intelligence (pp. 1206-1216). PMLR.

# ### Package loading
# We use a couple of useful packages to plot and optimize
# the different hyper-parameters
using AbstractGPs
using Distributions
using KernelFunctions
using LinearAlgebra
using Lux
using Optimisers
using Plots
using Random
using Zygote
default(; legendfontsize=15.0, linewidth=3.0);

Random.seed!(42)  # for reproducibility

# ## Data creation
# We create a simple 1D Problem with very different variations

xmin, xmax = (-3, 3)  # Limits
N = 150
noise_std = 0.01
x_train_vec = rand(Uniform(xmin, xmax), N) # Training dataset
x_train = collect(eachrow(x_train_vec)) # vector-of-vectors for neural network compatibility
target_f(x) = sinc(abs(x)^abs(x)) # We use sinc with a highly varying value
y_train = target_f.(x_train_vec) + randn(N) * noise_std
x_test_vec = range(xmin, xmax; length=200) # Testing dataset
x_test = collect(eachrow(x_test_vec)) # vector-of-vectors for neural network compatibility

plot(xmin:0.01:xmax, target_f; label="ground truth")
scatter!(x_train_vec, y_train; label="training data")

# ## Model definition
# We create a neural net with 2 layers and 10 units each.
# The data is passed through the NN before being used in the kernel.
neuralnet = Chain(Dense(1 => 20), Dense(20 => 30), Dense(30 => 5))

# Initialize the neural network parameters
rng = Random.default_rng()
ps, st = Lux.setup(rng, neuralnet)

smodel = StatefulLuxLayer(neuralnet, ps, st)

# We use the Squared Exponential Kernel:
k = SqExponentialKernel() ∘ FunctionTransform(smodel)

# We now define our model:
gpprior = GP(k)  # GP Prior
fx = AbstractGPs.FiniteGP(gpprior, x_train, noise_std^2)  # Prior at the observations
fp = posterior(fx, y_train)  # Posterior of f given the observations

# This computes the negative log evidence of `y` (the negative log marginal likelihood of
# the neural network parameters), which is going to be used as the objective:
loss(y) = -logpdf(fx, y)

@info "Initial loss = $(loss(y_train))"

# We show the initial prediction with the untrained model
p_init = plot(; title="Loss = $(round(loss(y_train); sigdigits=6))")
plot!(vcat(x_test...), target_f; label="true f")
scatter!(vcat(x_train...), y_train; label="data")
pred_init = marginals(fp(x_test))
plot!(vcat(x_test...), mean.(pred_init); ribbon=std.(pred_init), label="Prediction")

# ## Training
nmax = 200

# Create a wrapper function that updates the kernel with current parameters
function update_kernel_and_loss(model, ps, st, data)
    smodel = StatefulLuxLayer(model, ps, st)
    k_updated = SqExponentialKernel() ∘ FunctionTransform(smodel)
    fx_updated = AbstractGPs.FiniteGP(GP(k_updated), x_train, noise_std^2)
    return -logpdf(fx_updated, y_train), smodel.st, (;)
end

anim = Animation()
let tstate = Training.TrainState(neuralnet, ps, st, Optimisers.Adam(0.005))
    for i in 1:nmax
        _, loss_val, _, tstate = Training.single_train_step!(
            AutoZygote(), update_kernel_and_loss, (), tstate
        )

        if i % 10 == 0
            k =
                SqExponentialKernel() ∘ FunctionTransform(
                    StatefulLuxLayer(neuralnet, tstate.parameters, tstate.states)
                )
            fx = AbstractGPs.FiniteGP(GP(k), x_train, noise_std^2)

            @info "iteration $i/$nmax: loss = $loss_val"

            p = plot(; title="Loss[$i/$nmax] = $(round(loss_val; sigdigits=6))")
            plot!(vcat(x_test...), target_f; label="true f")
            scatter!(vcat(x_train...), y_train; label="data")
            pred = marginals(posterior(fx, y_train)(x_test))
            plot!(vcat(x_test...), mean.(pred); ribbon=std.(pred), label="Prediction")
            frame(anim)
            display(p)
        end
    end
end

gif(anim, "train-dkl.gif"; fps=3)
nothing #hide

# ![](train-dkl.gif)
