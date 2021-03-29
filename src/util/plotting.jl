@recipe f(gp::AbstractGP, x::AbstractArray) = gp(x)
@recipe f(gp::AbstractGP, xmin::Real, xmax::Real) = gp(range(xmin, xmax; length=1_000))

@recipe f(z::AbstractVector, gp::AbstractGP, x::AbstractArray) = (z, gp(x))
@recipe function f(z::AbstractVector, gp::AbstractGP, xmin::Real, xmax::Real)
    return (z, gp(range(xmin, xmax; length=1_000)))
end

@recipe f(gp::FiniteGP) = (gp.x, gp)
@recipe function f(z::AbstractVector, gp::FiniteGP)
    length(z) == length(gp.x) ||
        throw(DimensionMismatch("length of `z` and `gp.x` has to be equal"))

    # compute marginals
    μ, σ2 = mean_and_cov_diag(gp)
    σ = map(sqrt, σ2)

    ribbon := σ
    fillalpha --> 0.3
    linewidth --> 2
    return z, μ
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

