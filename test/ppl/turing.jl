@testset "Turing" begin
    @testset "GP regression" begin
        k = SqExponentialKernel()
        y = randn(3)
        X = randn(3, 1)
        x = [rand(1) for _ in 1:3]

        Turing.@model function GPRegression(y, X)
            # Priors.
            α ~ LogNormal(0.0, 0.1)
            ρ ~ LogNormal(0.0, 1.0)
            σ² ~ LogNormal(0.0, 1.0)

            # Realized covariance function
            kernel = α * (SqExponentialKernel() ∘ ScaleTransform(1 / ρ))
            f = GP(kernel)

            # Sampling Distribution.
            return y ~ f(X, σ²)
        end
        # Test for matrices
        m = GPRegression(y, RowVecs(X))
        @test length(Turing.sample(m, Turing.HMC(0.5, 1), 5)) == 5

        # Test for vectors of vector
        m = GPRegression(y, x)
        @test length(Turing.sample(m, Turing.HMC(0.5, 1), 5)) == 5
    end
    @testset "latent GP regression" begin
        X = randn(3, 1)
        x = [rand(1) for _ in 1:3]
        y = rand.(Poisson.(exp.(randn(3))))

        Turing.@model function latent_gp_regression(y, X)
            f = GP(Matern32Kernel())
            u ~ f(X)
            λ = exp.(u)
            return y ~ product_distribution([Poisson(λ_i) for λ_i in λ])
        end
        m = latent_gp_regression(y, RowVecs(X))
        @test length(Turing.sample(m, Turing.NUTS(), 5)) == 5

        # Test for vectors of vector
        m = latent_gp_regression(y, x)
        @test length(Turing.sample(m, Turing.NUTS(), 5)) == 5
    end
end
