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
            @test AbstractGPs._map_meanfunction(m, x) == testcase.calc_expected(x)
            differentiable_mean_function_tests(rng, m, x)
        end
    end
end
