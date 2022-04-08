@testset "mean_functions" begin
    rng = MersenneTwister(123456)
    N, D = 5, 3
    x1 = randn(rng, N)
    xD_colvecs = ColVecs(randn(rng, D, N))
    xD_rowvecs = RowVecs(randn(rng, N, D))

    @testset "ZeroMean" begin
        m = ZeroMean{Float64}()

        for x in [x1, xD_colvecs, xD_rowvecs]
            @test AbstractGPs._map_meanfunction(m, x) == zeros(N)
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
        c = randn(rng)
        m = ConstMean(c)

        for x in [x1, xD_colvecs, xD_rowvecs]
            @test AbstractGPs._map_meanfunction(m, x) == fill(c, N)
            differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end

    @testset "CustomMean" begin
        foo_mean = x -> sum(abs2, x)
        m = CustomMean(foo_mean)

        for x in [x1, xD_colvecs, xD_rowvecs]
            @test AbstractGPs._map_meanfunction(m, x) == map(foo_mean, x)
            differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
end
