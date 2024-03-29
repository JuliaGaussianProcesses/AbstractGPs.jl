@recipe f(x::AbstractVector, gp::AbstractGP) = gp(x)
@recipe f(gp::FiniteGP) = (gp.x, gp)
@recipe function f(x::AbstractVector, gp::FiniteGP)
    length(x) == length(gp.x) ||
        throw(DimensionMismatch("length of `x` and `gp.x` has to be equal"))
    scale::Float64 = pop!(plotattributes, :ribbon_scale, 1.0)
    scale >= 0.0 || error("`ribbon_scale` keyword argument must be non-negative")

    # compute marginals
    μ, σ2 = mean_and_var(gp)

    ribbon := scale .* sqrt.(σ2)
    fillalpha --> 0.3
    linewidth --> 2
    return x, μ
end

"""
    plot(x::AbstractVector, f::FiniteGP; ribbon_scale=1, kwargs...)
    plot!([plot, ]x::AbstractVector, f::FiniteGP; ribbon_scale=1, kwargs...)

Plot the predictive mean for the projection `f` of a Gaussian process and a ribbon of
`ribbon_scale` standard deviations around it versus `x`.

!!! note
    Make sure to load [Plots.jl](https://github.com/JuliaPlots/Plots.jl) before you use
    this function.

# Examples

Plot the mean and a ribbon of 3 standard deviations:

```julia
using Plots

gp = GP(SqExponentialKernel())
plot(gp(rand(5)); ribbon_scale=3)
```
"""
RecipesBase.plot(::AbstractVector, ::FiniteGP; kwargs...)
@doc (@doc RecipesBase.plot(::AbstractVector, ::FiniteGP)) RecipesBase.plot!(
    ::AbstractVector, ::FiniteGP; kwargs...
)
@doc (@doc RecipesBase.plot(::AbstractVector, ::FiniteGP)) RecipesBase.plot!(
    ::RecipesBase.AbstractPlot, ::AbstractVector, ::FiniteGP; kwargs...
)

"""
    plot(f::FiniteGP; kwargs...)
    plot!([plot, ]f::FiniteGP; kwargs...)

Plot the predictive mean and a ribbon around it for the projection `f` of a Gaussian
process versus `f.x`.
"""
RecipesBase.plot(::FiniteGP; kwargs...)
@doc (@doc RecipesBase.plot(::FiniteGP)) RecipesBase.plot!(::FiniteGP; kwargs...)
@doc (@doc RecipesBase.plot(::FiniteGP)) RecipesBase.plot!(
    ::RecipesBase.AbstractPlot, ::FiniteGP; kwargs...
)

"""
    plot(x::AbstractVector, gp::AbstractGP; kwargs...)
    plot!([plot, ]x::AbstractVector, gp::AbstractGP; kwargs...)

Plot the predictive mean and a ribbon around it for the projection `gp(x)` of the Gaussian
process `gp`.
"""
RecipesBase.plot(::AbstractVector, ::AbstractGP; kwargs...)
@doc (@doc RecipesBase.plot(::AbstractVector, ::AbstractGP)) RecipesBase.plot!(
    ::AbstractVector, ::AbstractGP; kwargs...
)
@doc (@doc RecipesBase.plot(::AbstractVector, ::AbstractGP)) RecipesBase.plot!(
    ::RecipesBase.AbstractPlot, ::AbstractVector, ::AbstractGP; kwargs...
)

"""
    sampleplot([x::AbstractVector=f.x, ]f::FiniteGP; samples=1, kwargs...)

Plot samples from the projection `f` of a Gaussian process versus `x`.

!!! note
    Make sure to load [Plots.jl](https://github.com/JuliaPlots/Plots.jl) before you use
    this function.

When plotting multiple samples, these are treated as a _single_ series (i.e.,
only a single entry will be added to the legend when providing a `label`).

# Example

```julia
using Plots

gp = GP(SqExponentialKernel())
sampleplot(gp(rand(5)); samples=10, linealpha=1.0)
```
The given example plots 10 samples from the projection of the GP `gp`.
The `linealpha` is modified from default of 0.35 to 1.

---
    sampleplot(x::AbstractVector, gp::AbstractGP; samples=1, kwargs...)

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
    nsamples::Int = pop!(plotattributes, :samples, 1)
    samples = rand(sp.f, nsamples)

    flat_x = repeat(vcat(sp.x, NaN), nsamples)
    flat_f = vec(vcat(samples, fill(NaN, 1, nsamples)))

    # Set default attributes
    linealpha --> 0.35
    label --> ""

    return flat_x, flat_f
end
