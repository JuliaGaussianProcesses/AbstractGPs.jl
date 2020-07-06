"""
    LatentGP(f<:GP, lik)

 - `fx` is a `FiniteGP`.
 - `lik` is the log likelihood function which maps sample from f to corresposing 
 conditional likelihood distributions.
    
"""
struct LatentGP{T<:AbstractGPs.FiniteGP, S}
    fx::T
    lik::S
end

function Distributions.rand(rng::AbstractRNG, lgp::LatentGP)
    f = rand(rng, lgp.fx)
    y = rand(rng, lgp.lik(f))
    return (f=f, y=y)
end

"""
    logpdf(lgp::LatentGP, y::NamedTuple{(:f, :y)})

```math
    log p(y, f; x)
```
Returns the joint log density of the gaussian process output `f` and real output `y`.
"""
function Distributions.logpdf(lgp::LatentGP, y::NamedTuple{(:f, :y)})
    return logpdf(lgp.fx, y.f) + logpdf(lgp.lik(y.f), y.y)
end
