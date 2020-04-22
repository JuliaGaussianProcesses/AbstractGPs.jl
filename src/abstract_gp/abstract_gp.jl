# Define the AbstractGP type and its API.

abstract type AbstractGP end

# """


# """
# mean(f::AbstractGP, x::AbstractVector)

# cov(f::GP, x::AbstractVector) = kernelmatrix(f.k, x)
# cov_diag(f::GP, x::AbstractVector) = kerneldiagmatrix(f.k, x)

# cov(f::GP, x::AbstractVector, x′::AbstractVector) = kernelmatrix(f.k, x, x′)

# mean_and_cov(f::GP, x::AbstractVector) = (mean(f, x), cov(f, x))
# mean_and_cov_diag(f::GP, x::AbstractVector) = (mean(f, x), cov_diag(f, x))
