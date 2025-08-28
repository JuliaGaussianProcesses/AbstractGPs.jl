@testset "mean_functions" begin
    rng = MersenneTwister(123456)
    N, D = 5, 3
    x1 = randn(rng, N)
    xD_colvecs = ColVecs(randn(rng, D, N))
    xD_rowvecs = RowVecs(randn(rng, N, D))

    zero_mean_testcase = (; mean_function=ZeroMean(), calc_expected=_ -> zeros(N))

    c = randn(rng)
    const_mean_testcase = (; mean_function=ConstMean(c), calc_expected=_ -> fill(c, N))

    foo_mean = x -> sum(abs2, x)
    custom_mean_testcase = (;
        mean_function=CustomMean(foo_mean), calc_expected=x -> map(foo_mean, x)
    )

    @testset "$(typeof(testcase.mean_function))" for testcase in [
        zero_mean_testcase, const_mean_testcase, custom_mean_testcase
    ]
        for x in [x1, xD_colvecs, xD_rowvecs]
            m = testcase.mean_function
            @test mean_vector(m, x) == testcase.calc_expected(x)
            differentiable_mean_function_tests(rng, m, x)
        end
    end

    @testset "ColVecs & RowVecs" begin
        m = custom_mean_testcase.mean_function

        @test mean_vector(m, xD_colvecs) == map(foo_mean, eachcol(xD_colvecs.X))
        @test mean_vector(m, xD_rowvecs) == map(foo_mean, eachrow(xD_rowvecs.X))
    end

    # This test fails without the specialized methods
    #   `mean_vector(m::CustomMean, x::ColVecs)`
    #   `mean_vector(m::CustomMean, x::RowVecs)`
    @testset "DifferentiationInterface gradients" begin
        X = [1.;; 2.;; 3.;;]
        y = [1., 2., 3.]
        foo_mean = x -> sum(abs2, x)

        function construct_finite_gp(X, lengthscale, noise)
            mean = CustomMean(foo_mean)
            kernel = with_lengthscale(Matern52Kernel(), lengthscale)
            return GP(mean, kernel)(X, noise)
        end

        function loglike(lengthscale, noise)
            gp = construct_finite_gp(X, lengthscale, noise)
            return logpdf(gp, y)
        end

        backend = AutoMooncake()
        @test only(gradient(n -> loglike(1., n), backend, 1.)) isa Real
        @test only(gradient(l -> loglike(l, 1.), backend, 1.)) isa Real    
    end
end
