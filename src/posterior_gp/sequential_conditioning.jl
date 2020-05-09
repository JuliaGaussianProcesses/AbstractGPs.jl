
function posterior(fx::FiniteGP{<:PosteriorGP}, y::AbstractVector{<:Real})
    m, C_mat = mean_and_cov(fx)
    C1 = cholesky(Symmetric(C_mat))
    C2 = cholesky(Symmetric(cov(fx.f, fx.f.data.x)))

    #TODO: CHeck if we always need to take covariance w.r.t to previous posterior.
    C_new = update_chol(C2.U, C_mat, cov(fx.f, fx.x, fx.f.data.x))
    #TODO: Better way to get Cholesky struct from the decomposition.
    C_new = cholesky(C_new * C_new')
    α = fx.f.data.α
    α1 = C1 \ (y - m)
    #TODO: Is it right to append old alphas with new?
    append!(α, α1)
    x = fx.f.data.x
    append!(x, fx.x)

    #TODO: Should we create a fresh PosteriorGP or keep the chain?
    return PosteriorGP(fx.f.prior , (α=α, C=C_new, x=x))
end

"""
     update_chol(U11::UpperTriangular, C12::AbstractMatrix, C22::AbstractMatrix)

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
update_chol computes the UpperTriangular matrix `U` given `U11`, `C12`, and `C22`.

## Arguments
     - U11::UpperTriangular: The cholesky decomposition C11
     - C12::AbstractMatrix: matrix of size (size(U11, 1), size(C22, 1))
     - C22::AbstractMatrix: positive-definite matrix
"""
function update_chol(U11::UpperTriangular, C12::AbstractMatrix, C22::AbstractMatrix)
    U12 = U11' \ C12
    U22 = cholesky(Symmetric(C22 - U12'U12)).U
    return UpperTriangular([U11 U12; zero(U12') U22])
end

