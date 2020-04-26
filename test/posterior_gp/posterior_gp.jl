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
end
