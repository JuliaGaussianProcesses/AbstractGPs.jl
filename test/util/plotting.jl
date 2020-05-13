@testset "plotting" begin
    using Plots
    f = GP(SqExponentialKernel())
    plt1 = sampleplot(f(rand(10)), samples=10)
    @test plt1.n == 10

end
