"""
    GP{Tm<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) with known mean `m` and kernel `k`.

# Zero Mean
If only one argument is provided, assume the mean to be zero everywhere:
```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(5);

julia> mean(f(x)) == zeros(5)
true

julia> cov(f(x)) == pw(Matern32Kernel(), x)
true
```

### Constant Mean

If a `Real` is provided as the first argument, assume the mean function is constant with
that value
```jldoctest
julia> f = GP(5.0, RationalQuadraticKernel());

julia> x = randn(5);

julia> mean(f(x)) == 5.0 .* ones(5)
true

julia> cov(f(x)) == kernelmatrix(RationalQuadraticKernel(), x)
true
```

### Custom Mean

Provide an arbitrary function to compute the mean:
```jldoctest
julia> f = GP(x -> sin(x) + cos(x / 2), RationalQuadraticKernel(3.2));

julia> x = randn(5);

julia> mean(f(x)) == sin.(x) .+ cos.(x ./ 2)
true

julia> cov(f(x)) == Stheno.pw(RationalQuadraticKernel(3.2), x)
true
```
"""
struct GP{Tm<:MeanFunction, Tk<:Kernel} <: AbstractGP
    m::Tm
    k::Tk 
end

GP(m, k::Kernel) = GP(CustomMean(m), k)
GP(m::Real, k::Kernel) = GP(ConstMean(m), k)
GP(k::Kernel) = GP(ZeroMean(), k)

(f::GP)(x...) = FiniteGP(f, x...)


#
# Implementation of the AbstractGP API.
#

Statistics.mean(f::GP, x::AbstractVector) = map(f.m, x)

Statistics.cov(f::GP, x::AbstractVector) = kernelmatrix(f.k, x)

cov_diag(f::GP, x::AbstractVector) = kerneldiagmatrix(f.k, x)

Statistics.cov(f::GP, x::AbstractVector, x′::AbstractVector) = kernelmatrix(f.k, x, x′)

mean_and_cov(f::GP, x::AbstractVector) = (mean(f, x), cov(f, x))

mean_and_cov_diag(f::GP, x::AbstractVector) = (mean(f, x), cov_diag(f, x))
