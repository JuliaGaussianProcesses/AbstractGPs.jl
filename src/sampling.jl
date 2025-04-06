abstract type AbstractGPSamplingMethod end

SeedableRNG = Union{Xoshiro,MersenneTwister}

_rand(rng, d) = Random.rand(rng, d)
function _rand(rng::AbstractRNG, ::Type{T}) where {T<:SeedableRNG}
    return T(Random.rand(rng, 1:typemax(Int)))
end

# ## Interface

struct GPSample{F,S}
    fun::F
    sample::S
end

(gs::GPSample)(x::AbstractArray) = eval_at(gs.fun, gs.sample, x)

# This may become more challenging once we extend to multi-input GPS
(gs::GPSample)(x::Number) = only(gs([x]))

"""
    GPSampler(gp::AbstractGPs.AbstractGP, method::AbstractGPSamplingMethod)
Creates a sampler for the given `gp` using the specified `method`. 

```jldoctest
julia> f = GP(Matern32Kernel());

julia> gps = GPSampler(f, CholeskySampling())

julia> rand(gps)
```   
"""
struct GPSampler{F,S} <: Random.Sampler{GPSample}
    fun::F
    sampler::S

    # Specify input types here, since it is a "public" interface
    function GPSampler(gp::AbstractGPs.AbstractGP, method::AbstractGPSamplingMethod)
        fun, sampler = method(gp)
        return new{typeof(fun),typeof(sampler)}(fun, sampler)
    end
end

# Don't love the deepcopy here
# issue is "pass by sharing" and the mutable struct in CholeskySampling
function Random.rand(rng::AbstractRNG, gs::GPSampler)
    return GPSample(deepcopy(gs.fun), _rand(rng, gs.sampler))
end

# ## Utils

_get_prior(gp::AbstractGPs.GP) = gp
_get_prior(pgp::AbstractGPs.PosteriorGP) = pgp.prior

function get_obs_variance(pgp::AbstractGPs.PosteriorGP)
    σk = pgp.prior.kernel(0, 0)
    v = diag(pgp.data.C.L * pgp.data.C.U) .- σk
    return v
end

function get_target_prior(pgp::AbstractGPs.PosteriorGP)
    m = pgp.data.δ
    σ2 = get_obs_variance(pgp)
    return MvNormal(m, sqrt.(σ2))
end

#########################
# Function Space/ Cholesky

"""
    CholeskySampling(s=Conditional, generator=Xoshiro)
Sampling by using the standard way, via Cholesky decomposition. 
Arguments:
- `s`: Sampling type, either `Conditional` or `Independent`. Default is `Conditional`.
- `generator`: Random number generator to use in each sample. Default is `Xoshiro`.
"""
struct CholeskySampling{M,G} <: AbstractGPSamplingMethod
    function CholeskySampling(s=Conditional, generator=Xoshiro)
        return new{s,generator}()
    end
end

function (cs::CholeskySampling{M,G})(gp) where {M,G}
    return M(gp), G
end

"""
    Conditional
Generates a GP sample that conditions function samples on all previous samples.
"""
mutable struct Conditional{GPT<:AbstractGPs.AbstractGP}
    gp::GPT
end

function Conditional(gp::AbstractGPs.GP)
    data = (
        α=Vector{Float64}(undef, 0),
        C=Cholesky(UpperTriangular(Matrix{Float64}(undef, 0, 0))),
        x=Vector{Float64}(undef, 0),
        δ=Vector{Float64}(undef, 0),
    )
    pgp = AbstractGPs.PosteriorGP(gp, data)
    return Conditional(pgp)
end

function eval_at(s::Conditional, rng, x::AbstractArray)
    if isassigned(s.gp.data.x, 1)
        pgp = s.gp
    else
        pgp = s.gp.prior
    end
    fgp = pgp(x)
    y = rand(rng, fgp)
    s.gp = posterior(fgp, y)
    return y
end

"""
    Independent
Generates a GP sample that samples function samples independent from previous samples.
"""
struct Independent{GPT<:AbstractGPs.AbstractGP}
    gp::GPT
    function Independent(gp)
        return new{typeof(gp)}(gp)
    end
end

function eval_at(s::Independent, rng, x::AbstractArray)
    gp = s.gp
    fgp = gp(x)
    y = rand(rng, fgp)
    return y
end

# ## WeightSpace

# ### Utils

get_weight_distribution(::AbstractGPs.GP, rff) = MvNormal(ones(rff.l))

function get_weight_distribution(pgp::AbstractGPs.PosteriorGP, rff)
    d = get_target_prior(pgp)

    P = rff.(pgp.data.x)
    Pt = reduce(hcat, P)
    Cp = Symmetric(Pt * (d.Σ \ Pt') + I)
    C = cholesky(Cp)

    μ = C \ (Pt * (d.Σ \ d.μ))
    Σ = C \ I
    return MvNormal(μ, Symmetric(Σ))
end

# ### Main

"""
    RFFSampling(l::Int, rff_type::Type{<:KernelSpectralDensities.AbstractRFF}=DoubleRFF)
Sampling by using Random Fourier Features.
Arguments:
- `l`: Number of random Fourier features to use.
- `rff_type`: Type of random Fourier features to use. Default is `DoubleRFF`.
"""
struct RFFSampling{RFF} <: AbstractGPSamplingMethod
    l::Int
    function RFFSampling(l, rff_type::Type{<:KernelSpectralDensities.AbstractRFF}=DoubleRFF)
        return new{rff_type}(l)
    end
end

function (rffs::RFFSampling{RFF})(gp) where {RFF}
    prior = _get_prior(gp)
    # for now, hardcoding "1", later expand for multi-input
    S = SpectralDensity(prior.kernel, 1)
    # ToDo: add rng to RFF
    rff = RFF(S, rffs.l)

    ws = get_weight_distribution(gp, rff)

    return rff, ws
end

function eval_at(rff::KernelSpectralDensities.AbstractRFF, w, x::AbstractArray)
    return dot.(rff.(x), Ref(w))
end

# ## PathwiseSampler

# ### utils
struct KernelBasis{K}
    ker::K
    x::AbstractArray
end

(kb::KernelBasis)(x) = kb.ker.(Ref(x), kb.x)

function update_basis(pgp, cs::CholeskySampling)
    ker = pgp.prior.kernel
    x = pgp.data.x
    return KernelBasis(ker, x)
end

function update_basis(pgp, rffs::RFFSampling)
    rff, _ = rffs(pgp)

    return rff
end

# ### Main

"""
    PathwiseSampling(l::Int)
Sampling by using pathwise sampling, which uses RFF sampling for the prior and an update rule
based on the kernel. Takes as an input the number of random Fourier features `l` to use.
"""
struct PathwiseSampling{P,U} <: AbstractGPSamplingMethod
    prior::P
    update::U
end

function PathwiseSampling(l::Int)
    return PathwiseSampling(RFFSampling(l), CholeskySampling())
end

struct PathwiseSampler{PS,TS,D}
    prior_sampler::PS
    target_sampler::TS
    data::D
end

function (ps::PathwiseSampling)(pgp::AbstractGPs.PosteriorGP)
    upd_fun = update_basis(pgp, ps.update)

    prior = pgp.prior
    prior_sampler = GPSampler(prior, ps.prior)

    target_dist = get_target_prior(pgp)

    data = (C=pgp.data.C, x=pgp.data.x)
    return upd_fun, PathwiseSampler(prior_sampler, target_dist, data)
end

function _rand(rng::AbstractRNG, ps::PathwiseSampler)
    prior = rand(rng, ps.prior_sampler)
    f = prior(ps.data.x)

    ts = rand(rng, ps.target_sampler)

    v = ps.data.C \ (ts - f)

    return (prior=prior, v=v)
end

function eval_at(basis::KernelBasis, s, x::AbstractArray)
    return s.prior(x) .+ dot.(basis.(x), Ref(s.v))
end