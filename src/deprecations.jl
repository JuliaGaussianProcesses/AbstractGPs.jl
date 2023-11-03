@deprecate sampleplot(gp::FiniteGP, n::Int; kwargs...) sampleplot(gp; samples=n, kwargs...)
@deprecate sampleplot!(gp::FiniteGP, n::Int; kwargs...) sampleplot!(
    gp; samples=n, kwargs...
)
@deprecate sampleplot!(plt::RecipesBase.AbstractPlot, gp::FiniteGP, n::Int; kwargs...) sampleplot!(
    plt, gp; samples=n, kwargs...
)

@deprecate dtc(dtc::DTC, fx, y) approx_log_evidence(dtc, fx, y)

function _warn_elbo_called_with_DTC(dtc::DTC, fx, y)
    @warn "`elbo` was called with an object of type `DTC`, but should only be called with the `VFE` type instead"
    return elbo(VFE(dtc.fz), fx, y)
end

function _warn_dtc_called_with_VFE(vfe::VFE, fx, y)
    @warn "`dtc` was called with an object of type `VFE`, but instead you should call `approx_log_evidence` with an object of type `DTC`"
    return approx_log_evidence(DTC(vfe.fz), fx, y)
end

@deprecate elbo(dtc::DTC, fx, y) _warn_elbo_called_with_DTC(dtc, fx, y)
@deprecate dtc(vfe::VFE, fx, y) _warn_dtc_called_with_VFE(vfe, fx, y)
