@testset "gp" begin

    # Ensure that GP implements the AbstractGP API consistently.
    @testset "GP" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        m, k = CustomMean(sin), Matern32Kernel()
        f = GP(m, k)
        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean(f, x) == map(m, x)
        @test cov(f, x) == kernelmatrix(k, x)
        abstractgp_interface_tests(f, x, x′)
    end

    # Check that mean-function specialisations work as expected.
    @testset "sugar" begin
        @test GP(5, Matern32Kernel()).mean isa ConstMean
        @test GP(Matern32Kernel()).mean isa ZeroMean
    end
end
