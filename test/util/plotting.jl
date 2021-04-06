@testset "plotting" begin
    x = rand(10)
    f = GP(SqExponentialKernel())
    gp = f(x, 0.1)

    z = rand(10)
    plt1 = sampleplot(z, gp)
    @test plt1.n == 1
    @test plt1.series_list[1].plotattributes[:x] == sort(z)

    plt2 = sampleplot(gp; samples=10)
    @test plt2.n == 10
    sort_x = sort(x)
    @test all(series.plotattributes[:x] == sort_x for series in plt2.series_list)

    z = rand(7)
    plt3 = sampleplot(z, f; samples=8)
    @test plt3.n == 8
    sort_z = sort(z)
    @test all(series.plotattributes[:x] == sort_z for series in plt3.series_list)

    # Check recipe dispatches for `FiniteGP`s
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), gp)
    @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
    @test rec[1].args[1] == x
    @test rec[1].args[2] == gp
    @test isempty(rec[1].plotattributes) # no default attributes

    z = 1 .+ x
    for kwargs in (
        Dict{Symbol,Any}(),
        Dict{Symbol,Any}(:ribbon_scale => 3),
        Dict{Symbol,Any}(:ribbon_scale => rand()),
    )
        scale = get(kwargs, :ribbon_scale, 1.0)
        rec = RecipesBase.apply_recipe(kwargs, z, gp)
        @test length(rec) == 1 && length(rec[1].args) == 2 # one series with two arguments
        @test rec[1].args[1] == z
        @test rec[1].args[2] == zero(x)
        # 3 default attributes
        attributes = rec[1].plotattributes
        @test sort!(collect(keys(attributes))) == [:fillalpha, :linewidth, :ribbon]
        @test attributes[:fillalpha] == 0.3
        @test attributes[:linewidth] == 2
        @test attributes[:ribbon] == scale .* sqrt.(var(gp))
    end

    # Check recipe dispatches for `AbstractGP`s
    # with `AbstractVector` and `AbstractRange`:
    for x in (rand(10), 0:0.01:1)
        rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), x, f)
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
