# # Example: Multi-Output data modelling using Multi-Output kernels

# Loading the necessary packages and setting seed.

using AbstractGPs, KernelFunctions, Plots, Random, Distributions
Random.seed!(1234);

# Generate synthetic multi-dim regression dataset with 1-dim inputs and 2-dim outputs.

N = 100
x = rand(Uniform(-1, 1), N) # 1-dim input 
y = [[sin(xi) + exp(xi) , cos(xi)] .+ 0.1 * randn(2) for xi in x]  # 2-dim output 
plt = scatter(
    x, 
    [yi[1] for yi in y], 
    [yi[2] for yi in y]; 
    label="Data Points",
    xlabel="x",
    ylabel="y[1]",
    zlabel="y[2]",
    )

# Create input for multi-output GPs.

X = MOInput(x, 2)

# Utility function to transform back flattened outputs to original form

transform_back(y) = [y[[i, i+Int(length(y)/2)]] for i in 1:Int(length(y)/2)]


# Flatten outputs

Y = vcat([yi[1] for yi in y], [yi[2] for yi in y])

# Define a multi-output kernel which assumes the outputs are independent of each other.

k = IndependentMOKernel(MaternKernel())

# Instantiate a Gaussian Process with the given kernel.

f = GP(k)

# Instantiate a `FiniteGP`, a finite dimentional projection at the inputs of the dataset 
# observed under Gaussian Noise with $\sigma = 0.1$ .

fx = f(X, 0.1)

# Data's log-likelihood w.r.t prior `GP`. 

logpdf(fx, Y)

# Calculating the exact posterior over `f` given `Y`. The GP's kernel currently has 
# some arbitrary fixed parameters. 

p_fx = posterior(fx, Y)

# Data's log-likelihood under the posterior `GP`. We see that it drastically increases.

logpdf(p_fx(X, 0.1), Y)

mean_y = mean(p_fx(MOInput(collect(-1:0.01:1),2), 0.01))
mean_y = transform_back(mean_y)
plot!(
    plt,
    collect(-1:0.01:1), 
    [yi[1] for yi in mean_y], 
    [yi[2] for yi in mean_y]; 
    c="black",
    label="Mean Posterior",
    smooth=false,
    xlabel="x",
    ylabel="y[1]",
    zlabel="y[2]",
    )
for i = 1:10
    sample = transform_back(rand(p_fx(MOInput(collect(-1:0.01:1),2), 0.001)))
    plot!(
    plt,
    collect(-1:0.01:1), 
    [yi[1] for yi in sample], 
    [yi[2] for yi in sample]; 
    smooth=false,
    alpha=0.2,
    c="red",
    label="",
    xlabel="x",
    ylabel="y[1]",
    zlabel="y[2]",
    )    
end
plt