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
    plot([x::AbstractVector, ]f::FiniteGP; ribbon_scale=1)

Plot the predictive mean as well as a ribbon with a width equal to `ribbon_scale` times the standard deviation versus x.

Make sure to run `using Plots` before using this function

# Example
```julia
using Plots
gp = GP(SqExponentialKernel())
plot(gp(rand(5)); ribbon_scale=3)
```
The given example plots the mean with 3 std. dev. from the projection of the GP `gp`.

--- 
    plot(x::AbstractVector, gp::AbstractGP; ribbon_scale=1)

Plot mean and std. dev from the finite projection `gp(x, 1e-9)` versus `x`.
"""
RecipesBase.plot(f::FiniteGP)

"""
    sampleplot([x::AbstractVector, ]f::FiniteGP; samples=1)

Plot samples from `f` versus `x` (default value: `f.x`).

Make sure to run `using Plots` before using this function.

# Example
```julia
using Plots
gp = GP(SqExponentialKernel())
sampleplot(gp(rand(5)); samples=10, markersize=5)
```
The given example plots 10 samples from the projection of the GP `gp`. The `markersize` is modified
from default of 0.5 to 5.

---
    sampleplot(x::AbstractVector, gp::AbstractGP; samples=1)

Plot samples from the finite projection `gp(x, 1e-9)` versus `x`.
"""
@userplot struct SamplePlot{X<:AbstractVector,F<:FiniteGP}
    x::X
    f::F
end

# default constructor (recipe forwards arguments as tuple)
SamplePlot((x, f)::Tuple{<:AbstractVector,<:FiniteGP}) = SamplePlot(x, f)

# `FiniteGP`s without explicit `x`
SamplePlot((f,)::Tuple{<:FiniteGP}) = SamplePlot((f.x, f))

# `AbstractGP`s with explicit `x`
# zero mean observation noise with variance 1e-9 to avoid numerical issues in
# Cholesky decomposition
SamplePlot((x, gp)::Tuple{<:AbstractVector,<:AbstractGP}) = SamplePlot((gp(x, 1e-9),))

@recipe function f(sp::SamplePlot)
    nsamples::Int = get(plotattributes, :samples, 1)
    samples = rand(sp.f, nsamples)

    # Set default attributes
    seriestype --> :line
    linealpha --> 0.2
    markershape --> :circle
    markerstrokewidth --> 0.0
    markersize --> 0.5
    markeralpha --> 0.3
    seriescolor --> "red"
    label --> ""

    return sp.x, samples
end
