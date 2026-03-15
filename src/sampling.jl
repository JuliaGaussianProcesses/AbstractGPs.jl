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

(gs::GPSample)(x) = eval_at(gs.fun, gs.sample, x)

# This may become more challenging once we extend to multi-input GPS
(gs::GPSample)(x::Number) = only(gs([x]))
(gs::GPSample)(x::Tuple{T,Int}) where {T} = only(eval_at(gs.fun, gs.sample, x))

"""
    GPSampler(gp::AbstractGPs.AbstractGP, method::AbstractGPSamplingMethod)
Creates a sampler for the given `gp` using the specified `method`. 

```jldoctest
julia> f = GP(Matern32Kernel());

julia> gps = GPSampler(f, CholeskySampling());

julia> rand(gps);
```   
"""
struct GPSampler{F,S} <: Random.Sampler{GPSample}
    fun::F
    sampler::S

    # Specify input types here, since it is a "public" interface
    function GPSampler(
        gp::AbstractGPs.AbstractGP, method::AbstractGPSamplingMethod; dims=Val(:auto)
    )
        fun, sampler = method(gp, dims)
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
    x = pgp.data.x[1]
    σk = pgp.prior.kernel(x, x)
    v = diag(pgp.data.C.L * pgp.data.C.U) .- σk
    return max.(v, default_σ²)
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

function (cs::CholeskySampling{M,G})(gp, dims) where {M,G}
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

_rand(rng, t::Tuple{Normal,Int}) = rand(rng, t[1], t[2])
get_weight_distribution(::AbstractGPs.GP, rff) = (Normal(), length(rff))

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
struct RFFSampling{RFF,RNG} <: AbstractGPSamplingMethod
    l::Int
    rng::RNG
    function RFFSampling(
        rng, l; rff_type::Type{<:KernelSpectralDensities.AbstractRFF}=DoubleRFF
    )
        return new{rff_type,typeof(rng)}(l, rng)
    end
end

function RFFSampling(l; rff_type::Type{<:KernelSpectralDensities.AbstractRFF}=DoubleRFF)
    return RFFSampling(Random.default_rng(), l; rff_type)
end

_extract_dims(x::AbstractVector{<:Tuple}) = (length(x[1][1]), x.out_dim)
_extract_dims(x::AbstractVector) = (length(x[1]), 1)

function determine_dims(pgp::AbstractGPs.PosteriorGP, ::Val{:auto})
    d, p = _extract_dims(pgp.data.x)
    return (d, p)
end

function determine_dims(pgp::AbstractGPs.PosteriorGP, dims)
    det_dims = determine_dims(pgp, Val(:auto))
    if det_sims == dims
        return det_dims
    else
        throw(
            ArgumentError(
                "Specified dims $dims do not match dimensions inferred from data $(det_dims).",
            ),
        )
    end
end

function determine_dims(::AbstractGPs.AbstractGP, ::Val{:auto})
    throw(
        ArgumentError(
            "Cannot determine input/output dimensions for a non-posterior GP. Please specify dims explicitly.",
        ),
    )
end
determine_dims(::AbstractGPs.AbstractGP, dims) = dims

# currently no way to infer the input domain of a prior GP
# maybe additional optional arguments for the GPSampler? 
function (rffs::RFFSampling{RFF})(gp::AbstractGPs.AbstractGP, dims) where {RFF}
    prior = _get_prior(gp)
    dims = determine_dims(gp, dims)

    rff = sample_rff(rffs.rng, prior.kernel, rffs.l, dims...; rff_type=RFF)

    ws = get_weight_distribution(gp, rff)

    return rff, ws
end

function eval_at(rff::KernelSpectralDensities.AbstractRFF, w, x)
    # return dot.(rff.(x), Ref(w))
    return dot(rff(x), w)
end

function eval_at(rff::KernelSpectralDensities.AbstractMORFF, w, x)
    # return dot.(rff.(x), Ref(w))
    return rff(x) * w
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

function (ps::PathwiseSampling)(pgp::AbstractGPs.PosteriorGP, dims)
    upd_fun = update_basis(pgp, ps.update)

    dims = determine_dims(pgp, dims)
    prior = pgp.prior
    prior_sampler = GPSampler(prior, ps.prior; dims)

    target_dist = get_target_prior(pgp)

    data = (C=pgp.data.C, x=pgp.data.x)
    return upd_fun, PathwiseSampler(prior_sampler, target_dist, data)
end

function _rand(rng::AbstractRNG, ps::PathwiseSampler)
    prior = rand(rng, ps.prior_sampler)
    f = prior.(ps.data.x) # here
    # display(f)

    ts = rand(rng, ps.target_sampler)
    # display(ts)

    v = ps.data.C \ (ts - f)

    return (prior=prior, v=v)
end

function eval_at(basis::KernelBasis, s, x)
    # return s.prior(x) .+ dot.(basis.(x), Ref(s.v))
    return s.prior(x) + dot(basis(x), s.v)
end
