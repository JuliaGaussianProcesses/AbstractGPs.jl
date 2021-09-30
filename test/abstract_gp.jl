@testset "abstract_gp.jl" begin
    k = SqExponentialKernel()
    f = GP(k)
    @test_throws ErrorException mean(f)
    @test_throws ErrorException var(f)
    @test_throws ErrorException cov(f)
    @test_throws ErrorException mean_and_var(f)
    @test_throws ErrorException mean_and_cov(f)
end
