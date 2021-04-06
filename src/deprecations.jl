@deprecate sampleplot(gp::FiniteGP, n::Int; kwargs...) sampleplot(gp; samples=n, kwargs...)
@deprecate sampleplot!(gp::FiniteGP, n::Int; kwargs...) sampleplot!(gp; samples=n, kwargs...)
@deprecate sampleplot!(plt::RecipesBase.AbstractPlot, gp::FiniteGP, n::Int; kwargs...) sampleplot!(plt, gp; samples=n, kwargs...)
