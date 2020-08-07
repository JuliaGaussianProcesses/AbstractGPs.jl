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

    X = MOInput(x, out_dim)
    Y = vcat(([yi[i] for yi in y] for i in 1:out_dim)...) 
    return X, Y
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
    X = MOInput(ColVecs(x), out_dim)
    y_ = ColVecs(y)
    Y = vcat(([yi[i] for yi in y_] for i in 1:out_dim)...) 
    return X, Y
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
function mo_inverse_transform(X::MOInput, Y::AbstractVector)
    N = length(Y) ÷ X.out_dim
    y = [Y[[i + j*N for j in 0:(X.out_dim - 1)]] for i in 1:N]
    return X.x, y
end

"""
    mo_inverse_transform(X::AbstractVector, Y::AbstractVector, out_dim::Int)
"""
function mo_inverse_transform(X::AbstractVector, Y::AbstractVector, out_dim::Int)
    N = length(Y) ÷ out_dim
    x = [first(xi) for xi in X[1:N]]
    y = [Y[[i + j*N for j in 0:(out_dim - 1)]] for i in 1:N]
    return x, y
end

"""
    mo_inverse_transform(Y::AbstractVector, out_dim::Int)
"""
function mo_inverse_transform(Y::AbstractVector, out_dim::Int)
    N = length(Y) ÷ out_dim
    y = [Y[[i + j*N for j in 0:(out_dim - 1)]] for i in 1:N]
    return y
end
