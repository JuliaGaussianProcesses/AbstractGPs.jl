abstract type MeanFunction end

"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end

_map(::ZeroMean{T}, x::AbstractVector) where {T} = zeros(T, length(x))

function ChainRulesCore.rrule(::typeof(_map), m::ZeroMean, x::AbstractVector)
    map_ZeroMean_pullback(Δ) = (NO_FIELDS, NO_FIELDS, Zero())
    return _map(m, x), map_ZeroMean_pullback
end

ZeroMean() = ZeroMean{Float64}()

"""
    ConstMean{T<:Real} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end

_map(m::ConstMean, x::AbstractVector) = fill(m.c, length(x))

"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy. Must be able to be mapped over an
`AbstractVector` of inputs.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end

_map(f::CustomMean, x::AbstractVector) = map(f.f, x)
