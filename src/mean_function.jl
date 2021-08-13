abstract type MeanFunction end

"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end

"""
This is an AbstractGPs-internal workaround for AD issues; ideally we would just extend Base.map
"""
_map_meanfunction(::ZeroMean{T}, x::AbstractVector) where {T} = Zeros{T}(length(x))

function ChainRulesCore.rrule(::typeof(_map_meanfunction), m::ZeroMean, x::AbstractVector)
    map_ZeroMean_pullback(Δ) = (NoTangent(), NoTangent(), ZeroTangent())
    return _map_meanfunction(m, x), map_ZeroMean_pullback
end

ZeroMean() = ZeroMean{Float64}()

"""
    ConstMean{T<:Real} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end

_map_meanfunction(m::ConstMean, x::AbstractVector) = Fill(m.c, length(x))

"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy. Must be able to be mapped over an
`AbstractVector` of inputs.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end

_map_meanfunction(f::CustomMean, x::AbstractVector) = map(f.f, x)
