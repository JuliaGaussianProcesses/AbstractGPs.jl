struct PosteriorGP{Tprior, Tdata} <: AbstractGP
    prior::Tprior
    data::Tdata
end

"""
    posterior(fx::FiniteGP, y::AbstractVector{<:Real})

Constructs the posterior distribution over `fx.f` given observations `y` at `x` made under
noise `fx.Σy`. This is another `AbstractGP` object. See chapter 2 of [1] for a recap on
exact inference in GPs. This posterior process has mean function
```julia
m_posterior(x) = m(x) + k(x, fx.x) inv(cov(fx)) (y - mean(fx))
```
and kernel
```julia
k_posterior(x, z) = k(x, z) - k(x, fx.x) inv(cov(fx)) k(fx.x, z)
```
where `m` and `k` are the mean function and kernel of `fx.f` respectively.
"""
function posterior(fx::FiniteGP, y::AbstractVector{<:Real})
    m, C_mat = mean_and_cov(fx)
    C = cholesky(Symmetric(C_mat))
    δ = y - m
    α = C \ δ
    return PosteriorGP(fx.f, (α=α, C=C, x=fx.x, δ=δ))
end

"""
    posterior(fx::FiniteGP{<:PosteriorGP}, y::AbstractVector{<:Real})

Constructs the posterior distribution over `fx.f` when `f` is itself a `PosteriorGP` by
updating the cholesky factorisation of the covariance matrix and avoiding recomputing it
from original covariance matrix. It does this by using `update_chol` functionality.

Other aspects are similar to a regular posterior.
"""
function posterior(fx::FiniteGP{<:PosteriorGP}, y::AbstractVector{<:Real})
    m2 = mean(fx.f.prior, fx.x)
    δ2 = y - m2
    C12 = cov(fx.f.prior, fx.f.data.x, fx.x)
    C22 = cov(fx.f.prior, fx.x) + fx.Σy
    chol = update_chol(fx.f.data.C, C12, C22)
    δ = vcat(fx.f.data.δ, δ2)
    α = chol \ δ
    x = vcat(fx.f.data.x, fx.x)
    return PosteriorGP(fx.f.prior , (α=α, C=chol, x=x, δ=δ))
end

# AbstractGP interface implementation.

function Statistics.mean(f::PosteriorGP, x::AbstractVector)
    return mean(f.prior, x) + cov(f.prior, x, f.data.x) * f.data.α
end

function Statistics.cov(f::PosteriorGP, x::AbstractVector)
    return cov(f.prior, x) - Xt_invA_X(f.data.C, cov(f.prior, f.data.x, x))
end

function cov_diag(f::PosteriorGP, x::AbstractVector)
    return cov_diag(f.prior, x) - diag_Xt_invA_X(f.data.C, cov(f.prior, f.data.x, x))
end

function Statistics.cov(f::PosteriorGP, x::AbstractVector, z::AbstractVector)
    C_xcond_x = cov(f.prior, f.data.x, x)
    C_xcond_y = cov(f.prior, f.data.x, z)
    return cov(f.prior, x, z) - Xt_invA_Y(C_xcond_x, f.data.C, C_xcond_y)
end

function mean_and_cov(f::PosteriorGP, x::AbstractVector)
    C_xcond_x = cov(f.prior, f.data.x, x)
    m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
    C_post = cov(f.prior, x) - Xt_invA_X(f.data.C, C_xcond_x)
    return (m_post, C_post)
end

function mean_and_cov_diag(f::PosteriorGP, x::AbstractVector)
    C_xcond_x = cov(f.prior, f.data.x, x)
    m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
    C_post_diag = cov_diag(f.prior, x) - diag_Xt_invA_X(f.data.C, C_xcond_x)
    return (m_post, C_post_diag)
end
