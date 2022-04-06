@testset "mean_functions" begin
    @testset "ZeroMean" begin
        rng, N, D = MersenneTwister(123456), 5, 3
        x1 = randn(rng, N)
        xD = ColVecs(randn(rng, D, N))
        xD′ = RowVecs(randn(rng, N, D))

        m = ZeroMean{Float64}()

        for x in [x1, xD, xD′]
            @test AbstractGPs._map_meanfunction(m, x) == zeros(size(x))
            differentiable_mean_function_tests(m, randn(rng, N), x)

            # Manually verify the ChainRule. Really, this should employ FiniteDifferences, but
            # currently ChainRulesTestUtils isn't up to handling this, so this will have to do
            # for now.
            y, pb = rrule(AbstractGPs._map_meanfunction, m, x)
            @test y == AbstractGPs._map_meanfunction(m, x)
            Δmap, Δf, Δx = pb(randn(rng, N))
            @test iszero(Δmap)
            @test iszero(Δf)
            @test iszero(Δx)
        end
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
