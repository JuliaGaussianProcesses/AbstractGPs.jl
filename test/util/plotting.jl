@testset "plotting" begin
    
    x = rand(10)
    gp = GP(SqExponentialKernel())(x)
    plt1 = sampleplot(gp, samples=10)
    @test plt1.n == 10

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), gp)
    @test rec[1].args[1] ≈ x atol=1e-5
    @test rec[1].args[2] ≈ zero(x) atol=1e-5
end
