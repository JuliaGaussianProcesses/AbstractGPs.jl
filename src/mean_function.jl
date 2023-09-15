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

# Warning
`CustomMean` is generally sufficient for testing purposes, but care should be taken if
attempting to differentiate through `mean_vector` with a `CustomMean` when using
`Zygote.jl`. In particular, `mean_vector(m::CustomMean, x)` is implemented as `map(m.f, x)`,
which when `x` is a `ColVecs` or `RowVecs` will not differentiate correctly.

In such cases, you should implement `mean_vector` directly for your custom mean.
For example, if `f(x) = sum(x)`, you might implement `mean_vector` as
```julia
mean_vector(::CustomMean{typeof(f)}, x::ColVecs) = vec(sum(x.X; dims=1))
mean_vector(::CustomMean{typeof(f)}, x::RowVecs) = vec(sum(x.X; dims=2))
```
which avoids ever applying `map` to a `ColVecs` or `RowVecs`.
"""
struct CustomMean{Tf} <: MeanFunction
    f::Tf
end

mean_vector(m::CustomMean, x::AbstractVector) = map(m.f, x)
