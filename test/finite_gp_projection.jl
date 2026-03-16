_rng() = MersenneTwister(123456)

function generate_noise_matrix(rng::AbstractRNG, N::Int)
    A = randn(rng, N, N)
    return Symmetric(A * A' + I)
end

@testset "finite_gp" begin
    @testset "convert" begin
        f = GP(sin, SqExponentialKernel())
        x = randn(10)
        Σy = generate_noise_matrix(Random.GLOBAL_RNG, 10)
        fx = f(x, Σy)
        @test fx isa AbstractGPs.FiniteGP

        dist = @inferred(convert(MvNormal, fx))
        @test dist isa MvNormal{Float64}
        @test mean(dist) ≈ mean(fx)
        @test cov(dist) ≈ cov(fx)

        dist = @inferred(convert(MvNormal{Float32}, fx))
        @test dist isa MvNormal{Float32}
        @test mean(dist) ≈ mean(fx)
        @test cov(dist) ≈ cov(fx)
    end
    @testset "statistics" begin
        rng, N, N′ = MersenneTwister(123456), 1, 9
        x, x′, Σy, Σy′ = randn(rng, N), randn(rng, N′), zeros(N, N), zeros(N′, N′)
        σ² = 1e-3
        Xmat = randn(rng, N, N′)
        f = GP(sin, SqExponentialKernel())
        fx, fx′ = f(x, Σy), f(x′, Σy′)
        @test fx isa AbstractGPs.FiniteGP
        @test fx′ isa AbstractGPs.FiniteGP

        for (x, obsdim) in ((RowVecs(Xmat), 1), (ColVecs(Xmat), 2))
            @test f(x) isa AbstractGPs.FiniteGP
            @test f(x, σ²) isa AbstractGPs.FiniteGP
            @test f(Xmat; obsdim=obsdim) == f(x)
            @test f(Xmat, σ²; obsdim=obsdim) == f(x, σ²)
        end
        @test @test_deprecated(f(Xmat)) == f(ColVecs(Xmat))

        @test mean(fx) == mean(f, x)
        @test cov(fx) == cov(f, x)
        @test var(fx) == diag(cov(fx))
        @test cov(fx, fx′) == cov(f, x, x′)
        @test mean.(marginals(fx)) == mean(f(x))
        @test var.(marginals(fx)) == var(f, x)
        @test std.(marginals(fx)) == sqrt.(var(f, x))
        let
            m, C = mean_and_cov(fx)
            @test m == mean(fx)
            @test C == cov(fx)
        end
        let
            m, c = mean_and_var(fx)
            @test m == mean(fx)
            @test c == var(fx)
        end
    end
    @testset "rand (deterministic)" begin
        rng = MersenneTwister(123456)
        N = 10
        x = randn(rng, N)
        Σy = generate_noise_matrix(rng, N)
        fx = GP(1, SqExponentialKernel())(x, Σy)
        @test fx isa AbstractGPs.FiniteGP

        # Check that samples are the correct size.
        @test length(rand(rng, fx)) == length(x)
        @test size(rand(rng, fx, 10)) == (length(x), 10)
        @test length(rand(fx)) == length(x)
        @test size(rand(fx, 10)) == (length(x), 10)

        # Check that `rand!` calls do not error
        y = similar(x)
        rand!(rng, fx, y)
        rand!(fx, y)
        ys = similar(x, length(x), 10)
        rand!(rng, fx, ys)
        rand!(fx, ys)
    end
    @testset "rand (statistical)" begin
        rng = MersenneTwister(123456)
        N = 10
        m0 = 1
        S = 100_000
        x = range(-3.0, 3.0; length=N)
        f = GP(m0, SqExponentialKernel())(x, 1e-12)
        @test f isa AbstractGPs.FiniteGP

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂1 = rand(rng, f, S)
        f̂2 = similar(f̂1)
        rand!(rng, f, f̂2)

        for f̂ in (f̂1, f̂2)
            @test maximum(abs.(mean(f̂; dims=2) - mean(f))) < 1e-2

            Σ′ = (f̂ .- mean(f)) * (f̂ .- mean(f))' ./ S
            @test mean(abs.(Σ′ - cov(f))) < 1e-2
        end
    end
    @testset "rand (gradients)" begin
        rng, N, S = MersenneTwister(123456), 10, 3
        x = collect(range(-3.0; stop=3.0, length=N))
        Σy = 1e-12

        # Check that the gradient w.r.t. the samples is correct (single-sample).
        adjoint_test(
            x -> rand(MersenneTwister(123456), GP(sin, SqExponentialKernel())(x, Σy)),
            randn(rng, N),
            x;
            atol=1e-9,
            rtol=1e-9,
        )

        # Check that the gradient w.r.t. the samples is correct (multisample).
        adjoint_test(
            x -> rand(MersenneTwister(123456), GP(sin, SqExponentialKernel())(x, Σy), S),
            randn(rng, N, S),
            x;
            atol=1e-9,
            rtol=1e-9,
        )
    end
    @testset "logpdf / loglikelihood" begin
        rng = MersenneTwister(123456)
        N = 10
        S = 11
        σ = 1e-1
        x = collect(range(-3.0; stop=3.0, length=N))
        f = GP(1, SqExponentialKernel())
        fx = f(x, 0)
        @test fx isa AbstractGPs.FiniteGP
        y = f(x, σ^2)
        @test y isa AbstractGPs.FiniteGP
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean(y)), cov(y)), ŷ)
        @test loglikelihood(y, ŷ) == logpdf(y, ŷ)
        # Check that multi-sample logpdf returns the correct type and is consistent with
        # single-sample logpdf
        Ŷ = rand(rng, y, S)
        @test logpdf(y, Ŷ) isa Vector{Float64}
        @test logpdf(y, Ŷ) ≈ [logpdf(y, Ŷ[:, n]) for n in 1:S]
        @test loglikelihood(y, Ŷ) == sum(logpdf(y, Ŷ))

        # Check gradient of logpdf at mean is zero for `f`.
        adjoint_test(ŷ -> logpdf(fx, ŷ), 1, ones(size(ŷ)))

        # Check that gradient of logpdf at mean is zero for `y`.
        adjoint_test(ŷ -> logpdf(y, ŷ), 1, ones(size(ŷ)))

        # Check that gradient w.r.t. inputs is approximately correct for `f`.
        x, l̄ = randn(rng, N), randn(rng)
        adjoint_test(
            x -> logpdf(f(x, 1e-3), ones(size(x))), l̄, collect(x); atol=1e-8, rtol=1e-8
        )
        adjoint_test(
            x -> sum(logpdf(f(x, 1e-3), ones(size(Ŷ)))),
            l̄,
            collect(x);
            atol=1e-8,
            rtol=1e-8,
        )

        # Check that the gradient w.r.t. the noise is approximately correct for `f`.
        σ_ = randn(rng)
        adjoint_test((σ_, ŷ) -> logpdf(f(x, exp(σ_)), ŷ), l̄, σ_, ŷ)
        adjoint_test((σ_, Ŷ) -> sum(logpdf(f(x, exp(σ_)), Ŷ)), l̄, σ_, Ŷ)
    end
    @testset "Type Stability - $T" for T in [Float64, Float32]
        rng = MersenneTwister(123456)
        x = randn(rng, T, 123)
        z = randn(rng, T, 13)
        f = GP(T(0), SqExponentialKernel())

        fx = f(x, T(0.1))

        y = rand(rng, fx)
        @test y isa Vector{T}
        @test logpdf(fx, y) isa T
    end
    @testset "AbstractMvNormal API" begin
        rng = MersenneTwister(424242)
        N = 5
        x = randn(rng, N)
        f = GP(SqExponentialKernel())
        fx = f(x, 0.1)
        y = rand(rng, N)
        Y = rand(rng, N, 10)
        r = rand(rng, 10)

        Distributions.TestUtils.test_mvnormal(fx, 10^6, rng)
        @test Distributions.invcov(fx) ≈ inv(cov(fx))
        @test Distributions.gradlogpdf(fx, y) ≈
            first(FiniteDifferences.grad(central_fdm(3, 1), Base.Fix1(logpdf, fx), y))
        @test Distributions.sqmahal!(r, fx, Y) ≈ Distributions.sqmahal(fx, Y)
    end
end

@testset "Docs" begin
    docstring = string(Docs.doc(logpdf, Tuple{AbstractGPs.FiniteGP,Vector{Float64}}))
    @test occursin("logpdf(f::FiniteGP, y::AbstractVecOrMat{<:Real})", docstring)
end
