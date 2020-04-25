using AbstractGPs: ZeroMean, ConstMean, CustomMean
using KernelFunctions

@testset "gp" begin

    # Ensure that GP roughly implements the AbstractGP interface.
    @testset "GP" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        m, k = CustomMean(sin), Matern32Kernel()
        f = GP(m, k)
        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean(f, x) == map(m, x)
        @test cov(f, x) == kernelmatrix(k, x)
        @test cov_diag(f, x) == diag(cov(f, x))
        @test cov(f, x, x) == kernelmatrix(k, x, x)
        @test cov(f, x, x′) == kernelmatrix(k, x, x′)
        @test cov(f, x, x′) ≈ cov(f, x′, x)'

        let
            m, C = mean_and_cov(f, x)
            @test m ≈ mean(f, x)
            @test C ≈ cov(f, x)
        end
        let
            m, c = mean_and_cov_diag(f, x)
            @test m ≈ mean(f, x)
            @test c ≈ cov_diag(f, x)
        end
    end

    # Check that mean-function specialisations work as expected.
    @testset "sugar" begin
        @test GP(5, Matern32Kernel()).mean isa ConstMean
        @test GP(Matern32Kernel()).mean isa ZeroMean
    end
end
