"""
    GP{Tm<:MeanFunction, Tk<:Kernel}

A Gaussian Process (GP) with known `mean` and `kernel`. See e.g. [1] for an introduction.

# Zero Mean
If only one argument is provided, assume the mean to be zero everywhere:
```jldoctest
julia> f = GP(Matern32Kernel());

julia> x = randn(5);

julia> mean(f(x)) == zeros(5)
true

julia> cov(f(x)) == kernelmatrix(Matern32Kernel(), x)
true
```

### Constant Mean

If a `Real` is provided as the first argument, assume the mean function is constant with
that value
```jldoctest
julia> f = GP(5.0, Matern32Kernel());

julia> x = randn(5);

julia> mean(f(x)) == 5.0 .* ones(5)
true

julia> cov(f(x)) == kernelmatrix(Matern32Kernel(), x)
true
```

### Custom Mean

Provide an arbitrary function to compute the mean:
```jldoctest
julia> f = GP(x -> sin(x) + cos(x / 2), Matern32Kernel());

julia> x = randn(5);

julia> mean(f(x)) == sin.(x) .+ cos.(x ./ 2)
true

julia> cov(f(x)) == kernelmatrix(Matern32Kernel(), x)
true
```

[1] - C. E. Rasmussen and C. Williams. "Gaussian processes for machine learning". MIT Press.
2006.
"""
struct GP{Tm<:MeanFunction, Tk<:Kernel} <: AbstractGP
    mean::Tm
    kernel::Tk 
end

GP(mean, kernel::Kernel) = GP(CustomMean(mean), kernel)
GP(mean::Real, kernel::Kernel) = GP(ConstMean(mean), kernel)
GP(kernel::Kernel) = GP(ZeroMean(), kernel)



# AbstractGP interface implementation.

Statistics.mean(f::GP, x::AbstractVector) = map(f.mean, x)

Statistics.cov(f::GP, x::AbstractVector) = kernelmatrix(f.kernel, x)

cov_diag(f::GP, x::AbstractVector) = kerneldiagmatrix(f.kernel, x)

Statistics.cov(f::GP, x::AbstractVector, x′::AbstractVector) = kernelmatrix(f.kernel, x, x′)

mean_and_cov(f::GP, x::AbstractVector) = (mean(f, x), cov(f, x))

mean_and_cov_diag(f::GP, x::AbstractVector) = (mean(f, x), cov_diag(f, x))
