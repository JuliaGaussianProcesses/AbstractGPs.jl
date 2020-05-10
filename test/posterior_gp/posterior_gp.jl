@testset "PosteriorGP" begin
    rng = MersenneTwister(123456)
    N_cond = 3
    N_a = 5
    N_b = 6

    # Specify prior.
    m = CustomMean(sin)
    k = Matern32Kernel()
    f = GP(m, k)

    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    fx = f(x, 1e-15)
    y = rand(rng, fx)

    # Construct posterior.
    f_post = posterior(fx, y)

    # Verify that posterior collapses around observations.
    @test mean(f_post, x) ≈ y
    @test cov_diag(f_post, x) ≈ zeros(N_cond) rtol=1e-14 atol=1e-14

    # Check interface is implemented fully and consistently.
    a = collect(range(-1.0, 1.0; length=N_a))
    b = randn(rng, N_b)
    abstractgp_interface_tests(f_post, a, b)

    #Check sequential conditioning posterior
    X = rand(5)
    y = rand(5)

    f1 = GP(SqExponentialKernel())
    p_fx1 = posterior(f1(X[1:3], 0.1), y[1:3])
    p_p_fx1 = posterior(p_fx1(X[4:5], 0.1), y[4:5])

    f2 = GP(SqExponentialKernel())
    p_fx2 = posterior(f2(X, 0.1), y)

    @test p_p_fx1.data.C.U ≈ p_fx2.data.C.U atol=1e-5
    @test p_p_fx1.data.α ≈ p_fx2.data.α atol=1e-5
    @test p_p_fx1.data.x ≈ p_fx2.data.x atol=1e-5
    @test p_p_fx1.data.δ ≈ p_fx2.data.δ atol=1e-5

end
