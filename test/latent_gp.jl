@testset "latent_gp" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)

    lgp = LatentGP(gp, x -> MvNormal(x, 0.1), 1e-5)
    @test lgp isa LatentGP
    @test lgp.f isa AbstractGPs.AbstractGP
    @test lgp.Σy isa Real

    lfgp = lgp(x)
    @test lfgp isa AbstractGPs.LatentFiniteGP
    @test lfgp.fx isa AbstractGPs.FiniteGP
    @test length(lfgp) == length(x)

    f = rand(10)
    @test logpdf(lfgp, (f=f, y=y)) isa Real
    @test rand(lfgp) isa NamedTuple{(:f, :y)}
end
