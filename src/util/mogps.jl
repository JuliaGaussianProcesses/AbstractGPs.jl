"""
    MOutput(x::AbstractVector, out_dim::Integer)

A data type to handle multi-dimensional outputs.
"""
struct MOutput{T<:AbstractVector} <: AbstractVector{Real}
    x::T
    out_dim::Integer
end

Base.length(out::MOutput) = out.out_dim * length(out.x)

Base.size(out::MOutput, d) = d::Integer == 1 ? out.out_dim * size(out.x, 1) : 1 
Base.size(out::MOutput) = (out.out_dim * size(out.x, 1),)

function Base.getindex(out::MOutput, ind::Integer)
    if ind > 0
        len = length(out.x)
        ind1 = ind % len
        ind2 = (ind - 1) ÷ len + 1
        if ind1 == 0 ind1 = len end
        return out.x[ind1][ind2]
    else
        throw(BoundsError(string("Trying to access at ", ind)))
    end
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
    return MOInput(x, out_dim), MOutput(y, out_dim)
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
    return MOInput(ColVecs(x), out_dim), MOutput(ColVecs(y), out_dim)
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
