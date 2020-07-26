"""
    mo_transform(x::AbstractVector, y::AbstractVector, out_dim::Int)

A utility function to transform multi-dimensional data in the form of Vector of Vectors
to data compatible with Multi-Output Kernels. This is the inverse of 
[`mo_inverse_transform`](@ref). 

...
# Arguments
- `x::AbstractVector`: the inputs.
- `y::AbstractVector`: the outputs.
- `out_dim::Int`: the output dimension.
...
"""
function mo_transform(x::AbstractVector, y::AbstractVector, out_dim::Int)

    X = MOInput(x, out_dim)
    Y = vcat(([yi[i] for yi in y] for i in 1:out_dim)...) 
    return X, Y
end

"""
    mo_inverse_transform(X, [Y::AbstractVector, out_dim::Int])

A utility function to transform back Multi-Output Kernel comptatible data 
to Vector of Vectors form. This is the inverse of [`mo_transform`](@ref). 

...
# Arguments
- `X::AbstractVector`: the inputs.
- `Y::AbstractVector`: the outputs.
- `out_dim::Int`: the output dimension.
...
"""
mo_inverse_transform

function mo_inverse_transform(X::MOInput, Y::AbstractVector)
    N = length(Y) ÷ X.out_dim
    y = [Y[[i + j*N for j in 0:(X.out_dim - 1)]] for i in 1:N]
    return X.x, y
end

function mo_inverse_transform(X::AbstractVector, Y::AbstractVector, out_dim::Int)
    N = length(Y) ÷ out_dim
    x = [first(xi) for xi in X[1:N]]
    y = [Y[[i + j*N for j in 0:(out_dim - 1)]] for i in 1:N]
    return x, y
end

function mo_inverse_transform(Y::AbstractVector, out_dim::Int)
    N = length(Y) ÷ out_dim
    y = [Y[[i + j*N for j in 0:(out_dim - 1)]] for i in 1:N]
    return y
end
