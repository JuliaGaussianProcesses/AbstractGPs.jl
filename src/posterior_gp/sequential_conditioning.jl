
function posterior(fx::FiniteGP{<:PosteriorGP}, y::AbstractVector{<:Real})
    m, C_mat = mean_and_cov(fx)
    C1 = cholesky(Symmetric(C_mat))
    C2 = cholesky(Symmetric(cov(fx.f, fx.f.data.x)))
    C_new = update_chol(C2.U, C_mat, cov(fx.f, fx.x, fx.f.data.x))
    C_new = cholesky(C_new * C_new')
    α = fx.f.data.α
    α1 = C1 \ (y - m)
    append!(α, α1)
    x = fx.f.data.x
    append!(x, fx.x)
    return PosteriorGP(fx.f.prior , (α=α, C=C_new, x=x))
end

"""
     update_chol!(chol, Kttq, Ktt0)

## Arguments

     - chol::UpperTriangular:    The cholesky decomposition of K_{0:t-1}+  I*Q
     - Kttq::AbstractMatrix:     The covariance matrix K_tt + I*Q
     - Ktt0::AbstractMatrix:     The covariance matrix Kt,0:t-1


Returns the updated cholesky decomposition of K_{0:t} + I*Q

"""

## Function for the update of the Cholesky decomposition - only increasing
function update_chol(chol::UpperTriangular, Kttq::AbstractMatrix, Ktt0::AbstractMatrix)
    S21 = Ktt0/chol
    S12 = S21'
    S22 = cholesky(Kttq - S21*S12).U
    return UpperTriangular([ chol S12; S21 S22])
end
