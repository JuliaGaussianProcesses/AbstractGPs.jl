@testset "mean_functions" begin
    @testset "CustomMean" begin
        rng, N, D = MersenneTwister(123456), 11, 2
        x = randn(rng, N)
        foo_mean = x->sum(abs2, x)
        f = CustomMean(foo_mean)

        @test map(f, x) == map(foo_mean, x)
        # differentiable_mean_function_tests(f, randn(rng, N), x)
    end
    @testset "ZeroMean" begin
        rng, P, Q, D = MersenneTwister(123456), 3, 2, 4
        # X = ColVecs(randn(rng, D, P))
        x = randn(rng, P)
        f = ZeroMean{Float64}()

        for x in [x]
            @test map(f, x) == zeros(size(x))
            # differentiable_mean_function_tests(f, randn(rng, P), x)
        end
    end
    @testset "ConstMean" begin
        rng, D, N = MersenneTwister(123456), 5, 3
        # X = ColVecs(randn(rng, D, N))
        x = randn(rng, N)
        c = randn(rng)
        m = ConstMean(c)

        for x in [x]
            @test map(m, x) == fill(c, N)
            # differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
end
