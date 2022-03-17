# If a matrix is `Diagonal` or if it is a `Cholesky` object,
# we generally don't need to wrap it in a `Symmetric`, because
# it's already symmetric. This is used in a couple of places to avoid precisely this and
# having to add specialised methods of e.g. `_cholesky` for complicated wrapped types.
_symmetric(X) = Symmetric(X)
_symmetric(X::Diagonal) = X
_symmetric(X::Cholesky) = X

# Small bit of indirection to work around a cholesky-related bug whereby the interaction
# between `FillArrays` and `Diagonal` and `Cholesky` causes problems.
_cholesky(X) = cholesky(X)
function _cholesky(X::Diagonal{<:Real,<:FillArrays.AbstractFill})
    return cholesky(Diagonal(collect(diag(X))))
end

"""
     update_chol(chol::Cholesky, C12::AbstractMatrix, C22::AbstractMatrix)

Let `C` be the positive definite matrix comprising blocks
```julia
C = [C11 C12;
     C21 C22]
```
with upper-triangular cholesky factorisation comprising blocks
```julia
U = [U11 U12;
     0   U22]
```
where `U11` and `U22` are themselves upper-triangular, and `U11 = cholesky(C11).U`.
update_chol computes the updated Cholesky given original `chol`, `C12`, and `C22`.

# Arguments
     - chol::Cholesky: The original cholesky decomposition
     - C12::AbstractMatrix: matrix of size (size(chol.U, 1), size(C22, 1))
     - C22::AbstractMatrix: positive-definite matrix
"""
function update_chol(chol::Cholesky, C12::AbstractMatrix, C22::AbstractMatrix)
    U12 = chol.U' \ C12
    U22 = cholesky(Symmetric(C22 - U12'U12)).U
    return Cholesky([chol.U U12; zero(U12') U22], 'U', 0)
end

# Various specialised operations using the Cholesky factorisation.

Xt_A_X(A::Cholesky, x::AbstractVector) = sum(abs2, A.U * x)
function Xt_A_X(A::Cholesky, X::AbstractMatrix)
    V = A.U * X
    return Symmetric(V'V)
end

Xt_A_Y(X::AbstractVecOrMat, A::Cholesky, Y::AbstractVecOrMat) = (A.U * X)' * (A.U * Y)

Xt_invA_X(A::Cholesky, x::AbstractVector) = sum(abs2, A.U' \ x)
function Xt_invA_X(A::Cholesky, X::AbstractVecOrMat)
    V = A.U' \ X
    return Symmetric(V'V)
end

Xt_invA_Y(X::AbstractVecOrMat, A::Cholesky, Y::AbstractVecOrMat) = (A.U' \ X)' * (A.U' \ Y)

At_A(A::AbstractVecOrMat) = A'A

diag_At_A(A::AbstractMatrix) = vec(sum(abs2.(A); dims=1))
diag_At_A(x::AbstractVector) = [dot(x, x)]

tr_At_A(A::AbstractVecOrMat) = sum(abs2, A)

function diag_At_B(A::AbstractVecOrMat, B::AbstractVecOrMat)
    size(A) == size(B) || throw(
        DimensionMismatch(
            "A ($(size(A))) and B ($(size(B))) do not have the same dimensions "
        ),
    )
    return vec(sum(A .* B; dims=1))
end
diag_At_B(x::AbstractVector, y::AbstractVector) = [dot(x, y)]

diag_Xt_A_X(A::Cholesky, X::AbstractVecOrMat) = diag_At_A(A.U * X)

function diag_Xt_A_Y(X::AbstractVecOrMat, A::Cholesky, Y::AbstractVecOrMat)
    size(X) == size(Y) || throw(
        DimensionMismatch(
            "X ($(size(X))) and Y ($(size(Y))) do not have the same dimensions "
        ),
    )
    return diag_At_B(A.U * X, A.U * Y)
end

diag_Xt_invA_X(A::Cholesky, X::AbstractVecOrMat) = diag_At_A(A.U' \ X)

function diag_Xt_invA_Y(X::AbstractMatrix, A::Cholesky, Y::AbstractMatrix)
    size(X) == size(Y) || throw(
        DimensionMismatch(
            "X ($(size(X))) and Y ($(size(Y))) do not have the same dimensions "
        ),
    )
    return diag_At_B(A.U' \ X, A.U' \ Y)
end

tr_Xt_invA_X(A::Cholesky, X::AbstractVecOrMat) = tr_At_A(A.U' \ X)

function Xtinv_A_Xinv(A::Cholesky, X::Cholesky)
    size(A) == size(X) || throw(
        DimensionMismatch(
            "A ($(size(A))) and X ($(size(X))) do not have the same dimensions "
        ),
    )
    C = A.U \ (X.U' \ A.U')
    return Symmetric(C * C')
end
