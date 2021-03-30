@testset "plotting" begin
    x = rand(10)
    f = GP(SqExponentialKernel())
    gp = f(x, 0.1)

    plt1 = sampleplot(gp, 10)
    @test plt1.n == 10

    # Check recipe dispatches for `FiniteGP`s
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), gp)
    @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
    @test rec[1].args[1] == x
    @test rec[1].args[2] == gp
    @test isempty(rec[1].plotattributes) # no default attributes

    z = 1 .+ x
    for kwargs in (Dict{Symbol, Any}(), Dict{Symbol, Any}(:ribbon_scale => rand()))
        scale = get(kwargs, :ribbon_scale, 1.0)::Float64
        rec = RecipesBase.apply_recipe(kwargs, z, gp)
        @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
        @test rec[1].args[1] == z
        @test rec[1].args[2] == zero(x)
        # 3 default attributes
        attributes = rec[1].plotattributes
        @test sort!(collect(keys(attributes))) == [:fillalpha, :linewidth, :ribbon]
        @test attributes[:fillalpha] == 0.3
        @test attributes[:linewidth] == 2
        @test attributes[:ribbon] == scale .* sqrt.(cov_diag(gp))
    end

    # Check recipe dispatches for `AbstractGP`s
    # with `AbstractVector` and `AbstractRange`:
    for x in (rand(10), 0:0.01:1)
        rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), x, f)
        @test length(rec) == 1 && length(rec[1].args) == 1 # one series with one argument
        @test rec[1].args[1] isa AbstractGPs.FiniteGP
        @test rec[1].args[1].x == x
        @test rec[1].args[1].f == f
        @test isempty(rec[1].plotattributes) # no default attributes
    end

    # Checks
    @test_throws DimensionMismatch plot(rand(5), gp)
    @test_throws ErrorException plot(rand(10), gp; ribbon_scale=-0.5)
end
