@testset "deprecations" begin
    x = rand(10)
    f = GP(SqExponentialKernel())
    gp = f(x, 0.1)

    # with `AbstractVector` and `AbstractRange`:
    for x in (rand(10), 0:0.01:1)
        rec = @test_deprecated RecipesBase.apply_recipe(Dict{Symbol,Any}(), f, x)
        @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
        @test rec[1].args[1] == x
        @test rec[1].args[2] == f
        @test isempty(rec[1].plotattributes) # no default attributes

        z = 1 .+ x
        rec = @test_deprecated RecipesBase.apply_recipe(Dict{Symbol,Any}(), z, f, x)
        @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
        @test rec[1].args[1] == z
        @test rec[1].args[2] isa AbstractGPs.FiniteGP
        @test rec[1].args[2].x == x
        @test rec[1].args[2].f == f
        @test isempty(rec[1].plotattributes) # no default attributes
    end

    # with minimum and maximum:
    xmin = rand()
    xmax = 4 + rand()
    rec = @test_deprecated RecipesBase.apply_recipe(Dict{Symbol,Any}(), f, xmin, xmax)
    @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
    @test rec[1].args[1] == range(xmin, xmax; length=1_000)
    @test rec[1].args[2] == f
    @test isempty(rec[1].plotattributes) # no default attributes

    z = range(0, 1; length=1_000)
    rec = @test_deprecated RecipesBase.apply_recipe(Dict{Symbol,Any}(), z, f, xmin, xmax)
    @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
    @test rec[1].args[1] == z
    @test rec[1].args[2] isa AbstractGPs.FiniteGP
    @test rec[1].args[2].x == range(xmin, xmax; length=1_000)
    @test rec[1].args[2].f == f
    @test isempty(rec[1].plotattributes) # no default attributes
end
