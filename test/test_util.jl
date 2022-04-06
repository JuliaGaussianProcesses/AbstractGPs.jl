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
    return println()
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
    f = (x_ad, x_fd) -> fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′) -> fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([
        fd_isapprox(get(() -> nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)
    ])
end

function adjoint_test(
    f, ȳ, x...; rtol=_rtol, atol=_atol, fdm=central_fdm(5, 1), print_results=false
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
    mean = AbstractGPs._map_meanfunction(m, x)
    @test mean isa AbstractVector
    @test length(mean) == length(x)
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
    ȳ::AbstractVector,
    x::AbstractVector;
    rtol=_rtol,
    atol=_atol,
)
    # Run forward tests.
    mean_function_tests(m, x)

    # Check adjoint.
    @assert length(ȳ) == length(x)
    adjoint_test(x -> AbstractGPs._map_meanfunction(m, x), ȳ, x; rtol=rtol, atol=atol)
    return nothing
end

function differentiable_mean_function_tests(
    rng::AbstractRNG, m::MeanFunction, x::AbstractVector; rtol=_rtol, atol=_atol
)
    ȳ = randn(rng, length(x))
    return differentiable_mean_function_tests(m, ȳ, x; rtol=rtol, atol=atol)
end
