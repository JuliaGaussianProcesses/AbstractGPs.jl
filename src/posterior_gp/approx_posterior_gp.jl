struct VFE end
const DTC = VFE

struct ApproxPosteriorGP{Tapprox, Tprior, Tdata} <: AbstractGP
    approx::Tapprox
    prior::Tprior
    data::Tdata
end

LinearAlgebra.Symmetric(X::Diagonal) = X


"""
    approx_posterior(::VFE, fx::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)

Compute the optimal approximate posterior [1] over the process `f`, given observations `y`
of `f` at `x`, and inducing points `u`, where `u = f(z)` for some inducing inputs `z`.
"""
function approx_posterior(::VFE, fx::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)
    U_y = cholesky(Symmetric(fx.Σy)).U
    U = cholesky(Symmetric(cov(u))).U
    B_εf = U' \ (U_y' \ cov(fx, u))'
    b_y = U_y' \ (y - mean(fx))
    Λ_ε = cholesky(Symmetric(B_εf * B_εf' + I))
    m_ε = Λ_ε \ (B_εf * b_y)
    return ApproxPosteriorGP(VFE(), fx.f, (m_ε=m_ε, Λ_ε=Λ_ε, U=U, α=U \ m_ε, z=u.x))
end

function Statistics.mean(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    return mean(f.prior, x) + cov(f.prior, x, f.data.z) * f.data.α
end

function Statistics.cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, f.data.z, x)
    return cov(f.prior, x) - At_A(A) + Xt_invA_X(f.data.Λ_ε, A)
end

function cov_diag(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, f.data.z, x)
    return cov_diag(f.prior, x) - diag_At_A(A) + diag_Xt_invA_X(f.data.Λ_ε, A)
end

function Statistics.cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector, y::AbstractVector)
    A_zx = f.data.U' \ cov(f.prior, f.data.z, x)
    A_zy = f.data.U' \ cov(f.prior, f.data.z, y)
    return cov(f.prior, x, y) - A_zx'A_zy + Xt_invA_Y(A_zx, f.data.Λ_ε, A_zy)
end

function mean_and_cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, f.data.z, x)
    m_post = mean(f.prior, x) + A' * f.data.m_ε
    C_post = cov(f.prior, x) - At_A(A) + Xt_invA_X(f.data.Λ_ε, A)
    return m_post, C_post
end

function mean_and_cov_diag(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, f.data.z, x)
    m_post = mean(f.prior, x) + A' * f.data.m_ε
    c_post = cov_diag(f.prior, x) - diag_At_A(A) + diag_Xt_invA_X(f.data.Λ_ε, A)
    return m_post, c_post
end
