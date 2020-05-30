@testset "plotting" begin
    
    x = rand(10)
    f = GP(SqExponentialKernel())
    gp = f(x)
    plt1 = sampleplot(gp, 10)
    @test plt1.n == 10

    rec1 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), gp)
    @test rec1[1].args[1] ≈ x atol=1e-5
    @test rec1[1].args[2] ≈ zero(x) atol=1e-5

    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), f, rand(10))
    @test typeof(rec2[1].args[1]) <: AbstractGPs.FiniteGP

    rec3 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), f, 0:0.01:1)
    @test typeof(rec3[1].args[1]) <: AbstractGPs.FiniteGP

    rec4 = RecipesBase.apply_recipe(Dict{Symbol, Any}(), f, 0, 1)
    @test typeof(rec4[1].args[1]) <: AbstractGPs.FiniteGP
end
