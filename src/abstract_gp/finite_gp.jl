"""
    FiniteGP{Tf<:AbstractGP, Tx<:AbstractVector, TΣy}

The finite-dimensional projection of the AbstractGP `f` at `x`. Assumed to be observed under
Gaussian noise with zero mean and covariance matrix `Σ`
"""
struct FiniteGP{Tf<:AbstractGP, Tx<:AbstractVector, TΣ} <: ContinuousMultivariateDistribution
    f::Tf
    x::Tx
    Σy::TΣ
end

function FiniteGP(f::AbstractGP, x::AbstractVector, σ²::AbstractVector{<:Real})
    return FiniteGP(f, x, Diagonal(σ²))
end

FiniteGP(f::AbstractGP, x::AbstractVector, σ²::Real) = FiniteGP(f, x, Fill(σ², length(x)))

FiniteGP(f::AbstractGP, x::AbstractVector) = FiniteGP(f, x, 1e-18)

function FiniteGP(
    f::AbstractGP,
    X::AbstractMatrix;
    obsdim::Int = KernelFunctions.defaultobs,
)
    return FiniteGP(f, KernelFunctions.vec_of_vecs(X; obsdim=obsdim))
end

Base.length(f::FiniteGP) = length(f.x)

(f::AbstractGP)(x...) = FiniteGP(f, x...)

"""
    mean(fx::FiniteGP)

Compute the mean vector of `fx`.

```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(11);

julia> mean(f(x)) == zeros(11)
true
```
"""
Statistics.mean(fx::FiniteGP) = mean(fx.f, fx.x)

"""
    cov(f::FiniteGP)

Compute the covariance matrix of `fx`.

## Noise-free observations

```jldoctest cov_finitegp
julia> f = GP(Matern52Kernel());

julia> x = randn(11);

julia> cov(f(x)) == kernelmatrix(Matern52Kernel(), x)
true
```

## Isotropic observation noise

```jldoctest cov_finitegp
julia> cov(f(x, 0.1)) == kernelmatrix(Matern52Kernel(), x) + 0.1 * I
true
```

## Independent anisotropic observation noise

```jldoctest cov_finitegp
julia> s = rand(11);

julia> cov(f(x, s)) == kernelmatrix(Matern52Kernel(), x) + Diagonal(s)
true
```

## Correlated observation noise

```jldoctest cov_finitegp
julia> A = randn(11, 11); S = A'A;

julia> cov(f(x, S)) == kernelmatrix(Matern52Kernel(), x) + S
true
```
"""
Statistics.cov(f::FiniteGP) = cov(f.f, f.x) + f.Σy

"""
    mean_and_cov(f::FiniteGP)

Equivalent to `(mean(f), cov(f))`, but sometimes more efficient to compute them jointly than
separately.

```jldoctest
julia> fx = GP(SqExponentialKernel())(range(-3.0, 3.0; length=10), 0.1);

julia> mean_and_cov(fx) == (mean(fx), cov(fx))
true
```
"""
function mean_and_cov(f::FiniteGP)
    m, C = mean_and_cov(f.f, f.x)
    return m, C + f.Σy
end

"""
    cov(fx::FiniteGP, gx::FiniteGP)

Compute the cross-covariance matrix between `fx` and `gx`.

```jldoctest
julia> f = GP(Matern32Kernel());

julia> x1 = randn(11);

julia> x2 = randn(13);

julia> cov(f(x1), f(x2)) == kernelmatrix(Matern32Kernel(), x1, x2)
true
```
"""
function Statistics.cov(fx::FiniteGP, gx::FiniteGP)
    @assert fx.f == gx.f
    return cov(fx.f, fx.x, gx.x)
end

"""
    marginals(f::FiniteGP)

Compute a vector of Normal distributions representing the marginals of `f` efficiently.
In particular, the off-diagonal elements of `cov(f(x))` are never computed.

```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(11);

julia> fs = marginals(f(x));

julia> mean.(fs) == mean(f(x))
true

julia> std.(fs) == sqrt.(diag(cov(f(x))))
true
```
"""
marginals(f::FiniteGP) = Normal.(mean(f), sqrt.(cov_diag(f.f, f.x) .+ diag(f.Σy)))

"""
    rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

Obtain `N` independent samples from the marginals `f` using `rng`. Single-sample methods
produce a `length(f)` vector. Multi-sample methods produce a `length(f)` x `N` `Matrix`.

```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(11);

julia> rand(f(x)) isa Vector{Float64}
true

julia> rand(MersenneTwister(123456), f(x)) isa Vector{Float64}
true

julia> rand(f(x), 3) isa Matrix{Float64}
true

julia> rand(MersenneTwister(123456), f(x), 3) isa Matrix{Float64}
true
```
"""
function Random.rand(rng::AbstractRNG, f::FiniteGP, N::Int)
    m, C_mat = mean_and_cov(f)
    C = cholesky(Symmetric(C_mat))
    return m .+ C.U' * randn(rng, promote_type(eltype(m), eltype(C)), length(m), N)
end
Random.rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
Random.rand(rng::AbstractRNG, f::FiniteGP) = vec(rand(rng, f, 1))
Random.rand(f::FiniteGP) = vec(rand(f, 1))

"""
    logpdf(f::FiniteGP, y::AbstractVecOrMat{<:Real})

The logpdf of `y` under `f` if is `y isa AbstractVector`. logpdf of each column of `y` if
`y isa Matrix`.

```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(11);

julia> y = rand(f(x));

julia> logpdf(f(x), y) isa Real
true

julia> Y = rand(f(x), 3);

julia> logpdf(f(x), Y) isa AbstractVector{<:Real}
true
```
"""
function Distributions.logpdf(f::FiniteGP, y::AbstractVector{<:Real})
    return first(logpdf(f, reshape(y, :, 1)))
end

function Distributions.logpdf(f::FiniteGP, Y::AbstractMatrix{<:Real})
    m, C_mat = mean_and_cov(f)
    C = cholesky(Symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ diag_Xt_invA_X(C, Y .- m)) ./ 2
end

"""
   elbo(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)

The Titsias Evidence Lower BOund (ELBO) [1]. `y` are observations of `f`, and `u`
are pseudo-points, where `u = f(z)` for some `z`.

```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=13);

julia> y = rand(f(x, 0.1));

julia> elbo(f(x, 0.1), y, f(z)) < logpdf(f(x, 0.1), y)
true
```

[1] - M. K. Titsias. "Variational learning of inducing variables in sparse Gaussian
processes". In: Proceedings of the Twelfth International Conference on Artificial
Intelligence and Statistics. 2009.
"""
function elbo(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)
    _dtc, chol_Σy, A = _compute_intermediates(f, y, u)
    return _dtc - (tr_Cf_invΣy(f, f.Σy, chol_Σy) - sum(abs2, A)) / 2
end

"""
    dtc(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)

The Deterministic Training Conditional (DTC) [1]. `y` are observations of `f`, and `u`
are pseudo-points.

```jldoctest
julia> f = GP(Matern52Kernel());

julia> x = randn(1000);

julia> z = range(-5.0, 5.0; length=256);

julia> y = rand(f(x, 0.1));

julia> isapprox(dtc(f(x, 0.1), y, f(z)), logpdf(f(x, 0.1), y); atol=1e-3, rtol=1e-3)
true
```

[1] - M. Seeger, C. K. I. Williams and N. D. Lawrence. "Fast Forward Selection to Speed Up
Sparse Gaussian Process Regression". In: Proceedings of the Ninth International Workshop on
Artificial Intelligence and Statistics. 2003
"""
function dtc(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)
    return first(_compute_intermediates(f, y, u))
end

# Factor out computations common to the `elbo` and `dtc`.
function _compute_intermediates(f::FiniteGP, y::AbstractVector{<:Real}, u::FiniteGP)
    consistency_check(f, y, u)
    chol_Σy = cholesky(f.Σy)

    A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
    Λ_ε = cholesky(Symmetric(A * A' + I))
    δ = chol_Σy.U' \ (y - mean(f))

    tmp = logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ))
    _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
    return _dtc, chol_Σy, A
end

function consistency_check(f, y, u)
    @assert length(f) == size(y, 1)
end

function tr_Cf_invΣy(f::FiniteGP, Σy::Diagonal, chol_Σy::Cholesky)
    return sum(cov_diag(f.f, f.x) ./ diag(Σy))
end
