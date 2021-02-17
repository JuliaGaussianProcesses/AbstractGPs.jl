using RecipesBase

@recipe f(gp::AbstractGP, x::AbstractArray) = gp(x)
@recipe f(gp::AbstractGP, x::AbstractRange) = gp(x)
@recipe f(gp::AbstractGP, xmin::Real, xmax::Real) = gp(collect(range(xmin, xmax, length=1000)))
@recipe function f(gp::FiniteGP)
    x = gp.x
    f = gp.f
    ms = marginals(gp)

    @series begin
        μ = mean.(ms)
        σ = std.(ms)
        ribbon := σ
        fillalpha --> 0.3
        linewidth --> 2
        x, μ
    end
end

"""
    sampleplot(GP::FiniteGP, samples)

Plot samples from the given `FiniteGP`. Make sure to run `using Plots` before using this 
function. 

# Example
```julia
using Plots
f = GP(SqExponentialKernel())
sampleplot(f(rand(10)), 10; markersize=5)
```
The given example plots 10 samples from the given `FiniteGP`. The `markersize` is modified
from default of 0.5 to 5.
"""
@userplot SamplePlot
@recipe function f(sp::SamplePlot)
    x = sp.args[1].x
    f = sp.args[1].f
    num_samples = sp.args[2]
    @series begin
        samples = rand(f(x, 1e-9), num_samples)
        seriestype --> :line
        linealpha --> 0.2
        markershape --> :circle
        markerstrokewidth --> 0.0
        markersize --> 0.5
        markeralpha --> 0.3
        seriescolor --> "red"
        label --> ""
        x, samples
    end
end

"""
    multigpplot(GP::FiniteGP, in_dim::Int, out_dim::Int)

"""
@userplot MultiGPPlot
@recipe function f(mgp::MultiGPPlot)
    gp = mgp.args[1]
    x = gp.x.x
    f = gp.f
    in_dim = mgp.args[2]
    out_dim = mgp.args[3]
    ms = mo_inverse_transform(;Y=marginals(gp), out_dim=out_dim)
    x = [[[x[i][j] for i in 1:length(x)] for j in 1:in_dim] for k in 1:out_dim]
    μ = [[[mean(ms[i][k]) for i in 1:length(ms)] for j in 1:in_dim] for k in 1:out_dim]
    σ = [[[std(ms[i][k]) for i in 1:length(ms)] for j in 1:in_dim] for k in 1:out_dim]
    @series begin
        ribbon := σ
        layout --> in_dim
        label --> reshape(["out_dim=$j" for j in 1:out_dim], 1, :)
        title --> reshape(["in_dim=$j" for j in 1:in_dim], 1, :)
        fillalpha --> 0.3
        linewidth --> 2
        x, μ
    end
end

"""
    multidataplot(x, y, in_dim::Int, out_dim::Int)

"""
@userplot MultiDataPlot
@recipe function f(mdp::MultiDataPlot)
    x = mdp.args[1]
    y = mdp.args[2]
    length(x) == length(y) || error("`x` and `y` should be of the same length")
    
    in_dim = mdp.args[3]
    out_dim = mdp.args[4]
    x = [[[x[i][j] for i in 1:length(x)] for j in 1:in_dim] for k in 1:out_dim]
    y = [[[y[i][k] for i in 1:length(y)] for j in 1:in_dim] for k in 1:out_dim]
    
    @series begin
        seriestype --> :scatter
        layout --> in_dim
        label --> reshape(["out_dim=$j" for j in 1:out_dim], 1, :)
        title --> reshape(["in_dim=$j" for j in 1:in_dim], 1, :)
        x, y
    end
end