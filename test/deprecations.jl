@testset "deprecations" begin
    x = rand(10)
    f = GP(SqExponentialKernel())
    gp = f(x, 0.1)

    plt = @test_deprecated sampleplot(gp, 10)
    @test plt.n == 1

    @test_deprecated sampleplot!(gp, 4)
    @test plt.n == 2

    @test_deprecated sampleplot!(Plots.current(), gp, 3)
    @test plt.n == 3
end
