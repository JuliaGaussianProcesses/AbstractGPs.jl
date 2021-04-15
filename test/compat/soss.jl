@testset "soss compat" begin
    @testset "GP regression" begin
        k = SqExponentialKernel()
        y = randn(3)
        X = randn(3, 1)
        x = [rand(1) for _ in 1:3]

        gp_regression = Soss.@model X begin
            # Priors.
            α ~ LogNormal(0.0, 0.1)
            ρ ~ LogNormal(0.0, 1.0)
            σ² ~ LogNormal(0.0, 1.0)

            # Realized covariance function
            kernel = α * (SqExponentialKernel() ∘ ScaleTransform(1 / ρ))
            f = GP(kernel)

            # Sampling Distribution.
            y ~ f(X, σ² + 1e-9)
        end

        # Test for matrices
        m = gp_regression(; X=RowVecs(X))
        @test length(Soss.sample(DynamicHMCChain, (m | (y=y,)), 5, 1)) == 5

        # Test for vectors of vector
        m = gp_regression(; X=x)
        @test length(Soss.sample(DynamicHMCChain, (m | (y=y,)), 5, 1)) == 5
    end
    @testset "latent GP regression" begin
        X = randn(3, 1)
        x = [rand(1) for _ in 1:3]
        y = rand.(Poisson.(exp.(randn(3))))

        latent_gp_regression = Soss.@model X begin
            f = GP(Matern32Kernel())
            u ~ f(X)
            λ = exp.(u)
            y ~ For(eachindex(λ)) do i
                Poisson(λ[i])
            end
        end

        m = latent_gp_regression(; X=RowVecs(X))
        @test length(Soss.sample(DynamicHMCChain, (m | (y=y,)), 5, 1)) == 5

        # Test for vectors of vector
        m = latent_gp_regression(; X=x)
        @test length(Soss.sample(DynamicHMCChain, (m | (y=y,)), 5, 1)) == 5
    end
end
