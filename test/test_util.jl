using LinearAlgebra, FiniteDifferences, Zygote, Random
using AbstractGPs: AbstractGP, MeanFunction
using LinearAlgebra: AbstractTriangular
using FiniteDifferences: j′vp
import FiniteDifferences: to_vec
using KernelFunctions: Kernel

const _rtol = 1e-10
const _atol = 1e-10

_to_psd(A::Matrix{<:Real}) = A * A' + I
_to_psd(a::Vector{<:Real}) = exp.(a) .+ 1
_to_psd(σ::Real) = exp(σ) + 1

Base.length(::Nothing) = 0

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

# # AbstractArrays.
# function to_vec(x::ColVecs{<:Real})
#     x_vec, back = to_vec(x.X)
#     return x_vec, x_vec -> ColVecs(back(x_vec))
# end

Base.zero(d::Dict) = Dict([(key, zero(val)) for (key, val) in d])
Base.zero(x::Array) = zero.(x)

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end

function adjoint_test(
    f, ȳ, x...;
    rtol=_rtol,
    atol=_atol,
    fdm=FiniteDifferences.Central(5, 1),
    print_results=false,
)
    # Compute forwards-pass and j′vp.
    y, back = Zygote.pullback(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # Check that forwards-pass agrees with plain forwards-pass.
    @test y ≈ f(x...)

    # Check that ad and fd adjoints (approximately) agree.
    print_results && print_adjoints(adj_ad, adj_fd, rtol, atol)
    @test fd_isapprox(adj_ad, adj_fd, rtol, atol)
end

"""
    mean_function_tests(m::MeanFunction, X::AbstractVector)

Test _very_ basic consistency properties of the mean function `m`.
"""
function mean_function_tests(m::MeanFunction, x::AbstractVector)
    @test map(m, x) isa AbstractVector
    @test length(ew(m, x)) == length(x)
end

"""
    differentiable_mean_function_tests(
        m::MeanFunction,
        ȳ::AbstractVector,
        x::AbstractVector,
    )

Ensure that the gradient w.r.t. the inputs of `MeanFunction` `m` are approximately correct.
"""
function differentiable_mean_function_tests(
    m::MeanFunction,
    ȳ::AbstractVector{<:Real},
    x::AbstractVector{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run forward tests.
    mean_function_tests(m, x)

    # Check adjoint.
    @assert length(ȳ) == length(x)
    adjoint_test(x->ew(m, x), ȳ, x; rtol=rtol, atol=atol)
end

# function differentiable_mean_function_tests(
#     m::MeanFunction,
#     ȳ::AbstractVector{<:Real},
#     x::ColVecs{<:Real};
#     rtol=_rtol,
#     atol=_atol,
# )
#     # Run forward tests.
#     mean_function_tests(m, x)

#     @assert length(ȳ) == length(x)
#     adjoint_test(X->ew(m, ColVecs(X)), ȳ, x.X; rtol=rtol, atol=atol)  
# end

function differentiable_mean_function_tests(
    rng::AbstractRNG,
    m::MeanFunction,
    x::AbstractVector;
    rtol=_rtol,
    atol=_atol,
)
    ȳ = randn(rng, length(x))
    return differentiable_mean_function_tests(m, ȳ, x; rtol=rtol, atol=atol)
end

"""
    abstractgp_interface_tests(
        f::AbstractGP,
        f′::AbstractGP,
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector,
        x3::AbstractVector;
        atol=1e-9, rtol=1e-9,
    )

Check that the `AbstractGP` interface is at least implemented for `f` and is
self-consistent. `x0` and `x1` must be valid inputs for `f`. `x2` and `x3` must be a valid
input for `f′`.
"""
function abstractgp_interface_tests(
    f::AbstractGP,
    f′::AbstractGP,
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector,
    x3::AbstractVector;
    atol=1e-9, rtol=1e-9,
)
    m = mean_vector(f, x0)
    @test m isa AbstractVector{<:Real}
    @test length(m) == length(x0)

    @assert length(x0) ≠ length(x1)
    @assert length(x0) ≠ length(x2)
    @assert length(x0) == length(x3)

    # Check that binary cov conforms to the API
    K_ff′_x0_x2 = cov(f, f′, x0, x2)
    @test K_ff′_x0_x2 isa AbstractMatrix{<:Real}
    @test size(K_ff′_x0_x2) == (length(x0), length(x2))
    @test K_ff′_x0_x2 ≈ cov(f′, f, x2, x0)'

    # Check that unary cov is consistent with binary cov and conforms to the API
    K_x0 = cov(f, x0)
    @test K_x0 isa AbstractMatrix{<:Real}
    @test size(K_x0) == (length(x0), length(x0))
    @test K_x0 ≈ cov(f, f, x0, x0) atol=atol rtol=rtol
    @test minimum(eigvals(K_x0)) > -atol
    @test K_x0 ≈ K_x0' atol=atol rtol=rtol

    # Check that single-process binary cov is consistent with binary-process binary-cov
    K_x0_x1 = cov(f, x0, x1)
    @test K_x0_x1 isa AbstractMatrix{<:Real}
    @test size(K_x0_x1) == (length(x0), length(x1))
    @test K_x0_x1 ≈ cov(f, f, x0, x1)

    # Check that binary cov_diag conforms to the API and is consistent with binary cov
    K_x0_x3_diag = cov_diag(f, f′, x0, x3)
    @test K_x0_x3_diag isa AbstractVector{<:Real}
    @test length(K_x0_x3_diag) == length(x0)
    @test K_x0_x3_diag ≈ diag(cov(f, f′, x0, x3)) atol=atol rtol=rtol
    @test K_x0_x3_diag ≈ cov_diag(f′, f, x3, x0) atol=atol rtol=rtol

    # Check that unary-binary cov_diag is consistent.
    K_x0_x0_diag = cov_diag(f, x0, x0)
    @test K_x0_x0_diag isa AbstractVector{<:Real}
    @test length(K_x0_x0_diag) == length(x0)
    @test K_x0_x0_diag ≈ diag(cov(f, x0, x0)) atol=atol rtol=rtol

    # Check that unary cov_diag conforms to the API and is consistent with unary cov
    K_x0_diag = cov_diag(f, x0)
    @test K_x0_diag isa AbstractVector{<:Real}
    @test length(K_x0_diag) == length(x0)
    @test K_x0_diag ≈ diag(cov(f, x0)) atol=atol rtol=rtol
end
