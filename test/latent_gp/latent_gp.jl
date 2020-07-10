@testset "latent_gp" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    
    lgp = LatentGP(gp, x -> MvNormal(x, 0.1), 1e-5)
    @test typeof(lgp) <: LatentGP
    @test typeof(lgp.f) <: AbstractGPs.AbstractGP
    @test typeof(lgp.Σy) <: Real

    lfgp = lgp(x)
    @test typeof(lfgp) <: AbstractGPs.LatentFiniteGP
    @test typeof(lfgp.fx) <: AbstractGPs.FiniteGP
    f = rand(10)
    @test typeof(logpdf(lfgp, (f=f, y=y))) <: Real
    @test typeof(rand(lfgp)) <: NamedTuple{(:f, :y)}
end
