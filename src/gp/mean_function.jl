abstract type MeanFunction end

"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere.
"""
struct ZeroMean{T<:Real} <: MeanFunction end

Base.map(::ZeroMean{T}, x::AbstractVector) where T = zeros(T, length(x))

ZeroMean() = ZeroMean{Float64}()


"""
    ConstMean{T<:Real} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end

Base.map(m::ConstMean, x::AbstractVector) = fill(m.c, length(x))


"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy. Must be able to be mapped over an
`AbstractVector` of inputs.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end

Base.map(f::CustomMean, x::AbstractVector) = map(f.f, x)
