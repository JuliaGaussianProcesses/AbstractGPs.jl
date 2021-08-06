# TODO: these should be in KernelFunctions
function KernelFunctions.pairwise(::SqEuclidean, x::AbstractGPUArray{<:Real, 2}, y::AbstractGPUArray{<:Real, 2}, kwargs...)
    return sum(abs2, x; dims=1)' .+ sum(abs2, y; dims=1) .- 2x'y
end

KernelFunctions.pairwise(d::SqEuclidean, x::AbstractGPUArray{<:Real, 2}, kwargs...) = KernelFunctions.pairwise(d, x, x)

KernelFunctions.pairwise(::Euclidean, x::AbstractGPUArray{<:Real, 2}, y::AbstractGPUArray{<:Real, 2}; kwargs...) = .√KernelFunctions.pairwise(SqEuclidean(), x, y)
KernelFunctions.pairwise(d::Euclidean, x::AbstractGPUArray{<:Real, 2}; kwargs...) = KernelFunctions.pairwise(d, x, x)

function KernelFunctions.pairwise(::Euclidean, x::AbstractGPUArray, y::AbstractGPUArray; kwargs...)
    return .√KernelFunctions.pairwise(SqEuclidean(), reshape(x, 1, :), reshape(y, 1, :))
end
KernelFunctions.pairwise(d::Euclidean, x::AbstractGPUArray; kwargs...) = KernelFunctions.pairwise(d, x, x)

#TODO: overload colwise for var(fx)

# Temporarily needed until var(fx) is working
function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal{<:Any, <:AbstractGPUArray}, chol_Σy::Cholesky)
    return sum(diag(cov(f.f, f.x)) ./ diag(Σy))
end

# Fixes a bug with cholesky of Diagonal CuArrays
function _cholesky(X::Diagonal{<:Real,T}) where T<:AbstractGPUArray
    # Bit of a hack - should maybe specialise to CuArray?
    return cholesky(T.name.wrapper(X))
end

## If we want to use FillArrays for fx.Σy, something like the following is needed in cov(fx):

# _add_broadcasted(A, B) = A .+ B

# # TODO: this is potentially much harder for var(fx)
# function _add_broadcasted(A::AbstractGPUArray, B::Diagonal{T, <:FillArrays.AbstractFill}) where T
#     return Base.materialize(Base.broadcasted(Base.BroadcastStyle(typeof(A)), +, A, B))
# end
# _add_broadcasted(B::Diagonal{T, <:FillArrays.AbstractFill}, A::AbstractGPUArray) where T = _add_broadcasted(A, B)

## A more general approach?
# const WrappedFillArray{T,N} = Adapt.WrappedArray{T, N, FillArrays.AbstractFill,FillArrays.AbstractFill{T,N}}
# function _add_broadcasted(A::AbstractGPUArray, B::WrappedFillArray) where T
#     Base.materialize(Base.broadcasted(Base.BroadcastStyle(typeof(A)), +, A, B))
# end
