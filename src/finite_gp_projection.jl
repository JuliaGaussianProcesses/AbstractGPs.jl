"""
    FiniteGP{Tf<:AbstractGP, Tx<:AbstractVector, TΣy}

The finite-dimensional projection of the AbstractGP `f` at `x`. Assumed to be observed under
Gaussian noise with zero mean and covariance matrix `Σy`
"""
struct FiniteGP{Tf<:AbstractGP,Tx<:AbstractVector,TΣ<:AbstractMatrix{<:Real}} <:
       AbstractMvNormal
    f::Tf
    x::Tx
    Σy::TΣ
end

function FiniteGP(f::AbstractGP, x::AbstractVector, σ²::AbstractVector{<:Real})
    return FiniteGP(f, x, Diagonal(σ²))
end

const default_σ² = 1e-18

function FiniteGP(f::AbstractGP, x::AbstractVector, σ²::Real=default_σ²)
    return FiniteGP(f, x, ScalMat(length(x), σ²))
end

function FiniteGP(f::AbstractGP, x::AbstractVector, σ²::UniformScaling)
    return FiniteGP(f, x, σ²[1, 1])
end

## conversions
Base.convert(::Type{MvNormal}, f::FiniteGP) = MvNormal(mean_and_cov(f)...)
function Base.convert(::Type{MvNormal{T}}, f::FiniteGP) where {T}
    μ, Σ = mean_and_cov(f)
    return MvNormal(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

Base.length(f::FiniteGP) = length(f.x)

(f::AbstractGP)(x...) = FiniteGP(f, x...)
function (f::AbstractGP)(
    X::AbstractMatrix, args...; obsdim::Union{Int,Nothing}=KernelFunctions.defaultobs
)
    return f(KernelFunctions.vec_of_vecs(X; obsdim=obsdim), args...)
end

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

Distributions.invcov(f::FiniteGP) = inv(cov(f))

"""
    var(f::FiniteGP)

Compute only the diagonal elements of [`cov(f)`](@ref).

# Examples

```jldoctest
julia> fx = GP(Matern52Kernel())(randn(10), 0.1);

julia> var(fx) == diag(cov(fx))
true
```
"""
function Statistics.var(f::FiniteGP)
    Σy = f.Σy
    return var(f.f, f.x) + view(Σy, diagind(Σy))
end

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
function StatsBase.mean_and_cov(f::FiniteGP)
    m, C = mean_and_cov(f.f, f.x)
    return m, C + f.Σy
end

"""
    mean_and_var(f::FiniteGP)

Compute both `mean(f)` and the diagonal elements of `cov(f)`.

Sometimes more efficient than computing them separately, particularly for posteriors.

# Examples

```jldoctest
julia> fx = GP(SqExponentialKernel())(range(-3.0, 3.0; length=10), 0.1);

julia> mean_and_var(fx) == (mean(fx), var(fx))
true
```
"""
function StatsBase.mean_and_var(f::FiniteGP)
    m, c = mean_and_var(f.f, f.x)
    Σy = f.Σy
    return m, c + view(Σy, diagind(Σy))
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
function marginals(f::FiniteGP)
    m, c = mean_and_var(f)
    return Normal.(m, sqrt.(c))
end

"""
    rand(rng::AbstractRNG, f::FiniteGP, N::Int=1)

Obtain `N` independent samples from the marginals `f` using `rng`. Single-sample methods
produce a `length(f)` vector. Multi-sample methods produce a `length(f)` × `N` `Matrix`.


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
    C = cholesky(_symmetric(C_mat))
    return m .+ C.U' * randn(rng, promote_type(eltype(m), eltype(C)), length(m), N)
end
Random.rand(f::FiniteGP, N::Int) = rand(Random.GLOBAL_RNG, f, N)
Random.rand(rng::AbstractRNG, f::FiniteGP) = vec(rand(rng, f, 1))
Random.rand(f::FiniteGP) = vec(rand(f, 1))

# in-place sampling
"""
    rand!(rng::AbstractRNG, f::FiniteGP, y::AbstractVecOrMat{<:Real})

Obtain sample(s) from the marginals `f` using `rng` and write them to `y`.

If `y` is a matrix, then each column corresponds to an independent sample.

```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(11);

julia> y = similar(x);

julia> rand!(f(x), y);

julia> rand!(MersenneTwister(123456), f(x), y);

julia> ys = similar(x, length(x), 3);

julia> rand!(f(x), ys);

julia> rand!(MersenneTwister(123456), f(x), ys);
```
"""
Random.rand!(::AbstractRNG, ::FiniteGP, ::AbstractVecOrMat{<:Real})

# Distributions defines methods for `rand!` (and `rand`) that fall back to `_rand!`
function Distributions._rand!(rng::AbstractRNG, f::FiniteGP, x::AbstractVecOrMat{<:Real})
    m, C_mat = mean_and_cov(f)
    C = cholesky(_symmetric(C_mat))
    lmul!(C.U', randn!(rng, x))
    x .+= m
    return x
end

"""
    logpdf(f::FiniteGP, y::AbstractVecOrMat{<:Real})

The logpdf of `y` under `f` if `y isa AbstractVector`. The logpdf of each column of `y` if
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
logpdf(f::FiniteGP, y::AbstractVecOrMat{<:Real})

Distributions.loglikelihood(f::FiniteGP, Y::AbstractMatrix{<:Real}) = sum(logpdf(f, Y))

function Distributions.logpdf(f::FiniteGP, Y::AbstractVecOrMat{<:Real})
    m, C_mat = mean_and_cov(f)
    C = cholesky(_symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log2π) + logdet(C)) .+ _sqmahal(m, C, Y)) ./ 2
end

Distributions.logdetcov(f::FiniteGP) = logdet(cov(f))

function Distributions.sqmahal(f::FiniteGP, x::AbstractVector)
    m, C = mean_and_cov(f)
    return _sqmahal(m, cholesky(_symmetric(C)), x)
end

function Distributions.sqmahal(f::FiniteGP, X::AbstractMatrix)
    m, C = mean_and_cov(f)
    return _sqmahal(m, cholesky(_symmetric(C)), X)
end

_sqmahal(m::AbstractVector, C::Cholesky, x::AbstractVector) = tr_Xt_invA_X(C, x - m)
_sqmahal(m::AbstractVector, C::Cholesky, x::AbstractMatrix) = diag_Xt_invA_X(C, x .- m)

function Distributions.sqmahal!(r::AbstractArray, f::FiniteGP, x::AbstractArray)
    return r .= sqmahal(f, x) # TODO write a more efficient implementation
end

function Distributions.gradlogpdf(f::FiniteGP, x::AbstractArray)
    m, C = mean_and_cov(f)
    return _gradlogpdf(m, C, x)
end

_gradlogpdf(m, C, x) = _symmetric(C) \ (m .- x)

Distributions.params(f::FiniteGP) = (f.f, f.x, f.Σy)
