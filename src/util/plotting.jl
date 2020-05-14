using RecipesBase

@recipe f(gp::AbstractGP, x::Array) = gp(x)
@recipe f(gp::AbstractGP, x::AbstractRange) = gp(x)
@recipe f(gp::AbstractGP, xmin::Number, xmax::Number) = (gp, range(xmin, xmax, length=1000))
@recipe function f(gp::FiniteGP)
    x = gp.x
    f = gp.f
    ms = marginals(gp)

    @series begin
        μ = mean.(ms)
        σ = std.(ms)
        ribbon := σ
        fillalpha --> 0.3
        linewidth --> 2
        x, μ
    end
end

@userplot SamplePlot
@recipe function f(sp::SamplePlot;
        samples = 1,
        sample_seriestype = :line,
        linealpha=0.2,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=0.5,
        markeralpha=0.3,
        seriescolor="red"
    )
    x = sp.args[1].x
    f = sp.args[1].f
    @series begin
        samples = rand(f(x, 1e-9), samples)

        seriestype --> sample_seriestype
        linealpha := linealpha
        markershape --> markershape
        markerstrokewidth --> markerstrokewidth
        markersize --> markersize
        markeralpha --> markeralpha
        seriescolor --> seriescolor
        label := ""

        x, samples
    end
end

