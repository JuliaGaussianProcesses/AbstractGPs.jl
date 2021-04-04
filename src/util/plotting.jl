@recipe f(x::AbstractVector, gp::AbstractGP) = gp(x)
@recipe f(gp::FiniteGP) = (gp.x, gp)
@recipe function f(x::AbstractVector, gp::FiniteGP)
    length(x) == length(gp.x) ||
        throw(DimensionMismatch("length of `x` and `gp.x` has to be equal"))
    scale::Float64 = pop!(plotattributes, :ribbon_scale, 1.0)
    scale > 0.0 || error("`bandwidth` keyword argument must be non-negative")

    # compute marginals
    μ, σ2 = mean_and_var(gp)

    ribbon := scale .* sqrt.(σ2)
    fillalpha --> 0.3
    linewidth --> 2
    return x, μ
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
