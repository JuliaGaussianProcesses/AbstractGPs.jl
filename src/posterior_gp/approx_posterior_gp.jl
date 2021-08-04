struct VFE
    fz::FiniteGP
end
const DTC = VFE

struct ApproxPosteriorGP{Tapprox,Tprior,Tdata} <: AbstractGP
    approx::Tapprox
    prior::Tprior
    data::Tdata
end

"""
    posterior(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})

Compute the optimal approximate posterior [1] over the process `f = fx.f`, given observations `y`
of `f` at `x`, and inducing points `v.fz`.

```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=13);

julia> v = VFE(f(z));

julia> y = rand(f(x, 0.1));

julia> post = posterior(v, f(x, 0.1), y);

julia> post(z) isa AbstractGPs.FiniteGP
true
```

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function posterior(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})
    U_y = _cholesky(_symmetric(fx.Σy)).U
    U = cholesky(_symmetric(cov(v.fz))).U

    B_εf = U' \ (U_y' \ cov(fx, v.fz))'

    b_y = U_y' \ (y - mean(fx))

    D = B_εf * B_εf' + I
    Λ_ε = cholesky(_symmetric(D))

    m_ε = Λ_ε \ (B_εf * b_y)

    cache = (m_ε=m_ε, Λ_ε=Λ_ε, U=U, α=U \ m_ε, b_y=b_y, B_εf=B_εf, x=fx.x, Σy=fx.Σy)
    return ApproxPosteriorGP(v, fx.f, cache)
end

"""
    function update_posterior(
        f_post_approx::ApproxPosteriorGP{VFE},
        fx::FiniteGP,
        y::AbstractVector{<:Real}
    )

Update the `ApproxPosteriorGP` given a new set of observations. Here, we retain the same 
of pseudo-points.
"""
function update_posterior(
    f_post_approx::ApproxPosteriorGP{VFE}, fx::FiniteGP, y::AbstractVector{<:Real}
)
    U = f_post_approx.data.U
    z = inducing_points(f_post_approx)

    U_y₂ = _cholesky(_symmetric(fx.Σy)).U

    temp = zeros(size(f_post_approx.data.Σy, 1), size(fx.Σy, 2))
    Σy = [f_post_approx.data.Σy temp; temp' fx.Σy]

    b_y = vcat(f_post_approx.data.b_y, U_y₂ \ (y - mean(fx)))

    B_εf₂ = U' \ (U_y₂' \ cov(fx.f, fx.x, z))'
    B_εf = hcat(f_post_approx.data.B_εf, B_εf₂)

    Λ_ε = f_post_approx.data.Λ_ε

    for col in eachcol(B_εf₂)
        lowrankupdate!(Λ_ε, col)
    end

    m_ε = Λ_ε \ (B_εf * b_y)
    α = U \ m_ε
    x = vcat(f_post_approx.data.x, fx.x)

    cache = (m_ε=m_ε, Λ_ε=Λ_ε, U=U, α=α, z=z, b_y=b_y, B_εf=B_εf, x=x, Σy=Σy)
    return ApproxPosteriorGP(f_post_approx.approx, fx.f, cache)
end

"""
    function update_approx_posterior(
        f_post_approx::ApproxPosteriorGP,
        fz::FiniteGP,
    )

Update the `ApproxPosteriorGP` given a new set of pseudo-points to append to the existing 
set of pseudo points. 
"""
function update_posterior(f_post_approx::ApproxPosteriorGP{VFE}, fz::FiniteGP)
    z = inducing_points(f_post_approx)

    U11 = f_post_approx.data.U
    C12 = cov(fz.f, z, fz.x)
    C22 = _symmetric(cov(fz))
    U = update_chol(Cholesky(U11, 'U', 0), C12, C22).U
    U22 = U[(end - length(fz) + 1):end, (end - length(fz) + 1):end]
    U12 = U[1:length(z), (end - length(fz) + 1):end]

    B_εf₁ = f_post_approx.data.B_εf

    Cu1f = cov(f_post_approx.prior, z, f_post_approx.data.x)
    Cu2f = cov(f_post_approx.prior, fz.x, f_post_approx.data.x)

    U_y = _cholesky(_symmetric(f_post_approx.data.Σy)).U

    B_εf₂ = U22' \ (Cu2f * inv(U_y) - U12' * B_εf₁)
    B_εf = vcat(B_εf₁, B_εf₂)

    Λ_ε = update_chol(f_post_approx.data.Λ_ε, B_εf₁ * B_εf₂', B_εf₂ * B_εf₂' + I)

    m_ε = Λ_ε \ (B_εf * f_post_approx.data.b_y)

    α = U \ m_ε

    z_ = vcat(z, fz.x)

    cache = (
        m_ε=m_ε,
        Λ_ε=Λ_ε,
        U=U,
        α=α,
        b_y=f_post_approx.data.b_y,
        B_εf=B_εf,
        x=f_post_approx.data.x,
        Σy=f_post_approx.data.Σy,
    )
    return ApproxPosteriorGP(VFE(f_post_approx.prior(z_)), f_post_approx.prior, cache)
end

# AbstractGP interface implementation.

function Statistics.mean(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    return mean(f.prior, x) + cov(f.prior, x, inducing_points(f)) * f.data.α
end

function Statistics.cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, inducing_points(f), x)
    return cov(f.prior, x) - At_A(A) + Xt_invA_X(f.data.Λ_ε, A)
end

function Statistics.var(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, inducing_points(f), x)
    return var(f.prior, x) - diag_At_A(A) + diag_Xt_invA_X(f.data.Λ_ε, A)
end

function Statistics.cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector, y::AbstractVector)
    A_zx = f.data.U' \ cov(f.prior, inducing_points(f), x)
    A_zy = f.data.U' \ cov(f.prior, inducing_points(f), y)
    return cov(f.prior, x, y) - A_zx'A_zy + Xt_invA_Y(A_zx, f.data.Λ_ε, A_zy)
end

function StatsBase.mean_and_cov(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, inducing_points(f), x)
    m_post = mean(f.prior, x) + A' * f.data.m_ε
    C_post = cov(f.prior, x) - At_A(A) + Xt_invA_X(f.data.Λ_ε, A)
    return m_post, C_post
end

function StatsBase.mean_and_var(f::ApproxPosteriorGP{VFE}, x::AbstractVector)
    A = f.data.U' \ cov(f.prior, inducing_points(f), x)
    m_post = mean(f.prior, x) + A' * f.data.m_ε
    c_post = var(f.prior, x) - diag_At_A(A) + diag_Xt_invA_X(f.data.Λ_ε, A)
    return m_post, c_post
end

inducing_points(f::ApproxPosteriorGP{VFE}) = f.approx.fz.x

"""
    elbo(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})

The Titsias Evidence Lower BOund (ELBO) [1]. `y` are observations of `fx`, and `fz`
are inducing points, where `u = f(z)` for some `z`.


```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=13);

julia> v = VFE(f(z));

julia> y = rand(f(x, 0.1));

julia> elbo(v, f(x, 0.1), y) < logpdf(f(x, 0.1), y)
true
```

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function elbo(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})
    _dtc, A = _compute_intermediates(fx, y, v.fz)
    return _dtc - (tr_Cf_invΣy(fx, fx.Σy) - sum(abs2, A)) / 2
end

"""
    dtc(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})

The Deterministic Training Conditional (DTC) [1]. `y` are observations of `fx`, and `fz`
are inducing points.


```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=256);

julia> v = VFE(f(z));

julia> y = rand(f(x, 0.1));

julia> isapprox(dtc(v, f(x, 0.1), y), logpdf(f(x, 0.1), y); atol=1e-6, rtol=1e-6)
true
```

[1] - M. Seeger, C. K. I. Williams and N. D. Lawrence. "Fast Forward Selection to Speed Up
Sparse Gaussian Process Regression". In: Proceedings of the Ninth International Workshop on
Artificial Intelligence and Statistics. 2003
"""
function dtc(v::VFE, fx::FiniteGP, y::AbstractVector{<:Real})
    return first(_compute_intermediates(fx, y, v.fz))
end

# Factor out computations common to the `elbo` and `dtc`.
function _compute_intermediates(fx::FiniteGP, y::AbstractVector{<:Real}, fz::FiniteGP)
    consistency_check(fx, y)
    chol_Σy = _cholesky(fx.Σy)

    A = cholesky(_symmetric(cov(fz))).U' \ (chol_Σy.U' \ cov(fx, fz))'
    Λ_ε = cholesky(Symmetric(A * A' + I))
    δ = chol_Σy.U' \ (y - mean(fx))

    tmp = logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ))
    _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
    return _dtc, A
end

function consistency_check(fx, y)
    @assert length(fx) == size(y, 1)
end

function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal)
    return sum(var(f.f, f.x) ./ diag(Σy))
end
