@testset "turing compat" begin
    k = SqExponentialKernel()
    y = randn(3)
    X = randn(3, 1)
    x = [rand(1) for _ in 1:3]
    @model GPRegression(y, X) = begin
        # Priors.
        α ~ LogNormal(0.0, 0.1)
        ρ ~ LogNormal(0.0, 1.0)
        σ² ~ LogNormal(0.0, 1.0)

        # Realized covariance function
        kernel = α * transform(SqExponentialKernel(), 1/ρ)
        f = GP(kernel)

        # Sampling Distribution.
        y ~ f(X, σ²)
    end
    # Test for matrices
    m = GPRegression(y, RowVecs(X))
    @test_nowarn sample(m, HMC(0.5, 1), 5)
    # Test for vectors of vector
    m = GPRegression(y, x)
    @test_nowarn sample(m, HMC(0.5, 1), 5)
end
