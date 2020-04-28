@testset "approx_posterior_gp" begin
    rng = MersenneTwister(123456)
    N_cond = 3
    N_a = 5
    N_b = 6

    @test Symmetric(Diagonal(randn(rng, 5))) isa Diagonal

    # Specify prior.
    f = GP(sin, Matern32Kernel())

    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    fx = f(x, 1e-15)
    y = rand(rng, fx)

    # Construct posterior.
    f_post = posterior(fx, y)

    # Construct optimal approximate posterior.
    f_approx_post = approx_posterior(VFE(), fx, y, fx)

    # Verify that approximate posterior ≈ posterior at various inputs.
    x_test = randn(rng, 100)
    @test mean(f_post, x_test) ≈ mean(f_approx_post, x_test)
    @test cov(f_post, x_test) ≈ cov(f_approx_post, x_test)

    # Verify that AbstractGP interface is implemented fully and consistently.
    a = collect(range(-1.0, 1.0; length=N_a))
    b = randn(rng, N_b)
    abstractgp_interface_tests(f_approx_post, a, b)
end
