import ChainRulesCore: ProjectTo, Tangent
using PDMats: ScalMat

ProjectTo(x::T) where {T<:ScalMat} = ProjectTo{T}(; dim=x.dim, value=ProjectTo(x.value))
(pr::ProjectTo{<:ScalMat})(dx::ScalMat) = ScalMat(pr.dim, pr.value(dx.value))
(pr::ProjectTo{<:ScalMat})(dx::Tangent{<:ScalMat}) = ScalMat(pr.dim, pr.value(dx.value))
