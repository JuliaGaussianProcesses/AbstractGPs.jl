@testset "turing compat" begin
    k = SqExponentialKernel()
    y = randn(3)
    X = randn(3, 1)
    x = [rand(1) for _ in 1:3]
    @model GPRegression(y, X) = begin
        # Priors.
        alpha ~ LogNormal(0.0, 0.1)
        rho ~ LogNormal(0.0, 1.0)
        sigma ~ LogNormal(0.0, 1.0)

        # Realized covariance function
        kernel = α * transform(SqExponentialKernel(), 1/ρ)
        f ~ GP(kernel)

        # Sampling Distribution.
        y ~ f(X, sigma)
    end
    # Test for matrices
    m = GPRegression(y, X)
    chain = sample(m, HMC(5, 0.5), 5)
    # Test for vectors of vector
    m = GPRegression(y, x)
    chain = sample(m, HMC(5, 0.5), 5)
end
