"""
    abstract type MeanFunction end

`MeanFunction` introduces an API for treating the prior mean function appropriately.
On the abstract level, all `MeanFunction` are functions.
However we generally want to evaluate them on a collection of inputs.
To this effect, we provide the `mean_vector(::MeanFunction, ::AbstractVector)`
function, which is equivalent to `map` but with possibilities of optimizations
(for [`ZeroMean`](@ref) and [`ConstMean`](@ref) for example).
"""
abstract type MeanFunction end

"""
    mean_vector(m::MeanFunction, x::AbstractVector)::AbstractVector{<:Real}

`mean_vector` is the function to call to apply a `MeanFunction` to a collection of inputs.
"""
mean_vector

"""
    ZeroMean{T<:Real} <: MeanFunction

Returns `zero(T)` everywhere, `T` is `Float64` by default.
"""
struct ZeroMean{T<:Real} <: MeanFunction end

mean_vector(::ZeroMean{T}, x::AbstractVector) where {T} = Zeros{T}(length(x))

ZeroMean() = ZeroMean{Float64}()

"""
    ConstMean{T<:Real} <: MeanFunction

Returns `c` everywhere.
"""
struct ConstMean{T<:Real} <: MeanFunction
    c::T
end

mean_vector(m::ConstMean, x::AbstractVector) = Fill(m.c, length(x))

"""
    CustomMean{Tf} <: MeanFunction

A wrapper around whatever unary function you fancy. Must be able to be mapped over an
`AbstractVector` of inputs.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end

mean_vector(m::CustomMean, x::AbstractVector) = map(m.f, x)

mean_vector(m::CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
mean_vector(m::CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))
