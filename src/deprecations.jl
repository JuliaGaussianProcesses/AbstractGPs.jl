@deprecate sampleplot(gp::FiniteGP, n::Int; kwargs...) sampleplot(gp; samples=n, kwargs...)
@deprecate sampleplot!(gp::FiniteGP, n::Int; kwargs...) sampleplot!(
    gp; samples=n, kwargs...
)
@deprecate sampleplot!(plt::RecipesBase.AbstractPlot, gp::FiniteGP, n::Int; kwargs...) sampleplot!(
    plt, gp; samples=n, kwargs...
)

@deprecate elbo(dtc::DTC, fx, y) approx_log_evidence(dtc, fx, y)
@deprecate dtc(vfe::Union{VFE,DTC}, fx, y) approx_log_evidence(vfe, fx, y)
