@testset "mean_functions" begin
    @testset "ZeroMean" begin
        rng, D, N = MersenneTwister(123456), 5, 3
        # X = ColVecs(randn(rng, D, N))
        x = randn(rng, N)
        x̄ = randn(rng, N)
        f = ZeroMean{Float64}()

        for x in [x]
            @test AbstractGPs._map_meanfunction(f, x) == zeros(size(x))
            differentiable_mean_function_tests(f, randn(rng, N), x)
        end

        # Manually verify the ChainRule. Really, this should employ FiniteDifferences, but
        # currently ChainRulesTestUtils isn't up to handling this, so this will have to do
        # for now.
        y, pb = rrule(AbstractGPs._map_meanfunction, f, x)
        @test y == AbstractGPs._map_meanfunction(f, x)
        Δmap, Δf, Δx = pb(randn(rng, N))
        @test iszero(Δmap)
        @test iszero(Δf)
        @test iszero(Δx)
    end
    @testset "ConstMean" begin
        rng, N, D = MersenneTwister(123456), 5, 3
        x1 = randn(rng, N)
        xD = ColVecs(randn(rng, D, N))
        xD′ = RowVecs(randn(rng, N, D))

        c = randn(rng)
        m = ConstMean(c)

        for x in [x1, xD, xD′]
            @test AbstractGPs._map_meanfunction(m, x) == fill(c, N)
            differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
    @testset "CustomMean" begin
        rng, N, D = MersenneTwister(123456), 5, 3
        x1 = randn(rng, N)
        xD = ColVecs(randn(rng, D, N))
        xD′ = RowVecs(randn(rng, N, D))

        foo_mean = x -> sum(abs2, x)
        m = CustomMean(foo_mean)

        for x in [x1, xD, xD′]
            @test AbstractGPs._map_meanfunction(m, x) == map(foo_mean, x)
            differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
end
