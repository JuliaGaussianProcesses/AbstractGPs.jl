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
        x::AbstractVector,
        z::AbstractVector;
        atol=1e-12,
    )

Check that the `AbstractGP` interface is at least implemented for `f` and is
self-consistent. `x` and `z` must be valid inputs for `f`. For tests to pass, the minimum
eigenvalue of `cov(f, x)` must be greater than `-eig_tol`.
"""
function abstractgp_interface_tests(
    f::AbstractGP,
    x::AbstractVector,
    z::AbstractVector;
    eig_tol::Real=1e-12,
    σ²::Real=1e-9,
)
    @assert length(x) ≠ length(z)

    # Verify that `mean` works and is the correct length and type.
    m = mean(f, x)
    @test m isa AbstractVector{<:Real}
    @test length(m) == length(x)

    # Verify that cov(f, x, z) works, is the correct size and type.
    C_xy = cov(f, x, z)
    @test C_xy isa AbstractMatrix{<:Real}
    @test size(C_xy) == (length(x), length(z))

    # Reversing arguments transposes the return.
    @test C_xy ≈ cov(f, z, x)'

    # Verify cov(f, x) works, is the correct size and type.
    C_xx = cov(f, x)
    @test size(C_xx) == (length(x), length(x))

    # Check that C_xx is positive definite.
    @test minimum(eigvals(Symmetric(C_xx))) > -eig_tol

    # Check that C_xx is consistent with cov(f, x, x).
    @test C_xx ≈ cov(f, x, x)

    # Check that cov_diag works, is the correct size and type.
    C_xx_diag = cov_diag(f, x)
    @test C_xx_diag isa AbstractVector{<:Real}
    @test length(C_xx_diag) == length(x)

    # Check C_xx_diag is consistent with cov(f, x).
    @test C_xx_diag ≈ diag(C_xx)

    # Check that mean_and_cov is consistent.
    let
        m, C = mean_and_cov(f, x)
        @test m ≈ mean(f, x)
        @test C ≈ cov(f, x)
    end

    # Check that mean_and_cov_diag is consistent.
    let
        m, c = mean_and_cov_diag(f, x)
        @test m ≈ mean(f, x)
        @test c ≈ cov_diag(f, x)
    end

    # Construct a FiniteGP, and check that all standard methods defined on it at least run.
    fx = f(x, σ²)
    fz = f(z, σ²)
    @test mean(fx) ≈ mean(f, x)
    @test cov(fx) ≈ cov(f, x) + fx.Σy
    @test cov(fx, fz) ≈ cov(f, x, z)
    @test first(mean_and_cov(fx)) ≈ mean(f, x)
    @test last(mean_and_cov(fx)) ≈ cov(f, x)
    @test mean.(marginals(fx)) ≈ mean(f, x)
    @test var.(marginals(fx)) ≈ cov_diag(f, x) .+ diag(fx.Σy)

    # Generate, compute logpdf, compare against VFE and DTC.
    y = rand(fx)
    @test length(y) == length(x)
    @test logpdf(fx, y) isa Real
    @test elbo(fx, y, f(x)) ≈ logpdf(fx, y) rtol=1e-5 atol=1e-5
    @test dtc(fx, y, f(x)) ≈ logpdf(fx, y) rtol=1e-5 atol=1e-5
end
