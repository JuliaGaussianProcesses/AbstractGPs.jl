"""
    MOutput(x<:AbstractVector, out_dim::Int)

A data type to handle multi-dimensional outputs.
"""
struct MOutput{T<:Real,X<:AbstractVector} <: AbstractVector{T}
    x::X
    out_dim::Int
end

"""
    MOutput(x::AbstractMatrix)

If the input is a `Matrix`, it expects the size of the input `x` to be `(out_dim, N)`
where `out_dim` is the output dimension and `N` is the number of data-points.
"""
MOutput(x::AbstractMatrix) = vec(permutedims(x))

"""
    MOutput(x::AbstractVector)

If the input is a `Vector` of `Vector`s, it expects input `x` to be `Vector` of
`N` `Vector`s each of length `out_dim` where `out_dim` is the output dimension and 
`N` is the number of data-points.
"""
function MOutput(x::X) where X <: AbstractVector
    return MOutput{eltype(first(x)), X}(x, length(first(x)))
end

Base.size(out::MOutput) = (out.out_dim * size(out.x, 1),)

@inline function Base.getindex(out::MOutput, ind::Integer)
    @boundscheck checkbounds(out, ind)
    (ind2, ind1) = fldmod1(ind, length(out.x))
    return out.x[ind1][ind2]
end

"""
    mo_transform

A utility function to transform multi-dimensional data in the form of Vector of Vectors
to data compatible with Multi-Output Kernels. This is the inverse of 
[`mo_inverse_transform`](@ref). 

...
"""
mo_transform

"""
    mo_transform(x::AbstractVector, y::AbstractVector, out_dim::Int)

`x` and `y` is are `Vector` of `Vector`s/`Real`s. Where each element of the `Vector` 
is a input/target for one observation. 

...
# Arguments
- `x`: the input `Vector`.
- `y`: the output `Vector`.
- `out_dim::Int`: the output dimension.
...
"""
function mo_transform(x::AbstractVector, y::AbstractVector, out_dim::Int)
    return MOInput(x, out_dim), MOutput(y)
end

"""
    mo_transform(x::AbstractMatrix, y::AbstractMatrix)

...
# Arguments
- `x`: the input matrix of size `(in_dim, N)`.
- `y`: the output matrix of size `(out_dim, N)`.
...
"""
function mo_transform(x::AbstractMatrix, y::AbstractMatrix)
    size(x, 2) == size(y, 2) || error("`x` and `y` are do not have compatible sizes.")
    out_dim = size(y, 1) 
    return MOInput(ColVecs(x), out_dim), MOutput(y)
end

"""
    mo_inverse_transform

A utility function to transform back Multi-Output Kernel comptatible data 
to Vector of Vectors form. This is the inverse of [`mo_transform`](@ref). 

...
# Arguments
- `X`: the inputs.
- `Y::AbstractVector`: the outputs.
- `out_dim::Int`: the output dimension.
...
"""
mo_inverse_transform

"""
    mo_inverse_transform(X::MOInput, Y::AbstractVector)
"""
function mo_inverse_transform(X::MOInput, Y::MOutput)
    return X.x, Y.x
end

"""
    mo_inverse_transform(X::MOInput)
"""
mo_inverse_transform(X::MOInput) = X.x

"""
    mo_inverse_transform(Y::MOutput)
"""
mo_inverse_transform(Y::MOutput) = Y.x

"""
    mo_inverse_transform(X::AbstractVector, Y::AbstractVector, out_dim::Int)
"""
function mo_inverse_transform(X::AbstractVector, Y::AbstractVector, out_dim::Int)
    N = length(Y) ÷ out_dim
    x = [first(xi) for xi in X[1:N]]
    y = [Y[[i + j*N for j in 0:(out_dim - 1)]] for i in 1:N]
    return x, y
end
