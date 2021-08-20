@testset "approx_posterior_gp" begin
    rng = MersenneTwister(123456)
    N_cond = 3
    N_a = 5
    N_b = 6

    # Specify prior.
    f = GP(sin, Matern32Kernel())

    # Sample from prior.
    x = collect(range(-1.0, 1.0; length=N_cond))
    fx = f(x, 1e-15)
    y = rand(rng, fx)

    # Construct posterior.
    f_post = posterior(fx, y)

    # Construct optimal approximate posterior.
    f_approx_post = posterior(VFE(f(x, 1e-12)), fx, y)

    # Verify that approximate posterior ≈ posterior at various inputs.
    x_test = randn(rng, 100)
    @test mean(f_post, x_test) ≈ mean(f_approx_post, x_test)
    @test cov(f_post, x_test) ≈ cov(f_approx_post, x_test)

    # Verify that AbstractGP interface is implemented fully and consistently.
    a = collect(range(-1.0, 1.0; length=N_a))
    b = randn(rng, N_b)
    TestUtils.test_internal_abstractgps_interface(rng, f_approx_post, a, b)

    @testset "update_posterior (new observation)" begin
        rng = MersenneTwister(1)
        X = rand(rng, 10)
        y = rand(rng, 10)
        Z = rand(rng, 4)

        f = GP(SqExponentialKernel())

        # online learning
        p_fx1 = posterior(VFE(f(Z)), f(X[1:7], 0.1), y[1:7])
        u_p_fx1 = update_posterior(p_fx1, f(X[8:10], 0.1), y[8:10])

        # batch learning
        p_fx2 = posterior(VFE(f(Z)), f(X, 0.1), y)

        @test inducing_points(u_p_fx1) ≈ inducing_points(p_fx2) atol = 1e-5
        @test u_p_fx1.data.m_ε ≈ p_fx2.data.m_ε atol = 1e-5
        @test u_p_fx1.data.Λ_ε.U ≈ p_fx2.data.Λ_ε.U atol = 1e-5
        @test u_p_fx1.data.U ≈ p_fx2.data.U atol = 1e-5
        @test u_p_fx1.data.α ≈ p_fx2.data.α atol = 1e-5
        @test u_p_fx1.data.b_y ≈ p_fx2.data.b_y atol = 1e-5
        @test u_p_fx1.data.B_εf ≈ p_fx2.data.B_εf atol = 1e-5
        @test u_p_fx1.data.x ≈ p_fx2.data.x atol = 1e-5
        @test u_p_fx1.data.Σy ≈ p_fx2.data.Σy atol = 1e-5
    end

    @testset "update_posterior (new pseudo-points)" begin
        rng = MersenneTwister(1)
        X = rand(rng, 10)
        y = rand(rng, 10)
        Z1 = rand(rng, 4)
        Z2 = rand(rng, 3)
        Z = vcat(Z1, Z2)

        f = GP(SqExponentialKernel())

        # Sequentially adding pseudo points
        p_fx1 = posterior(VFE(f(Z1)), f(X, 0.1), y)
        u_p_fx1 = update_posterior(p_fx1, f(Z2))

        # Adding all pseudo-points at once
        p_fx2 = posterior(VFE(f(Z)), f(X, 0.1), y)

        @test inducing_points(u_p_fx1) ≈ inducing_points(p_fx2) atol = 1e-5
        @test u_p_fx1.data.m_ε ≈ p_fx2.data.m_ε atol = 1e-5
        @test u_p_fx1.data.Λ_ε.U ≈ p_fx2.data.Λ_ε.U atol = 1e-5
        @test u_p_fx1.data.U ≈ p_fx2.data.U atol = 1e-5
        @test u_p_fx1.data.α ≈ p_fx2.data.α atol = 1e-2 rtol = 1e-2
        @test u_p_fx1.data.b_y ≈ p_fx2.data.b_y atol = 1e-5
        @test u_p_fx1.data.B_εf ≈ p_fx2.data.B_εf atol = 1e-5
        @test u_p_fx1.data.x ≈ p_fx2.data.x atol = 1e-5
        @test u_p_fx1.data.Σy ≈ p_fx2.data.Σy atol = 1e-5
    end

    @testset "elbo / dtc" begin
        x = collect(Float64, range(-1.0, 1.0; length=N_cond))
        f = GP(SqExponentialKernel())
        fx = f(x, 0.1)
        y = rand(rng, fx)

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(VFE(f(x)), fx, y) isa Real
        @test elbo(VFE(f(x)), fx, y) ≈ logpdf(fx, y)
        @test elbo(VFE(f(x .+ randn(rng, N_cond))), fx, y) < logpdf(fx, y)

        # Ensure that the dtc is close to the logpdf when appropriate.
        @test dtc(VFE(f(x)), fx, y) isa Real
        @test dtc(VFE(f(x)), fx, y) ≈ logpdf(fx, y)
    end

    @testset "Type Stability - $T" for T in [Float64, Float32]
        x = collect(T, range(-1.0, 1.0; length=N_cond))
        f = GP(T(0), SqExponentialKernel())
        fx = f(x, T(0.1))
        y = rand(rng, fx)
        jitter = T(1e-12)

        @test elbo(VFE(f(x, jitter)), fx, y) isa T
        @test dtc(VFE(f(x, jitter)), fx, y) isa T

        post = posterior(VFE(f(x, jitter)), fx, y)
        p_fx = post(x, jitter)

        @test rand(rng, p_fx) isa Vector{T}
        @test logpdf(p_fx, y) isa T
    end
end
