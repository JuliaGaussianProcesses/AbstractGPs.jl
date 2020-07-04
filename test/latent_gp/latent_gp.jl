@testset "latent_gp" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    fx = gp(x, 1e-5)

    lik = GaussianLikelihood(1e-5)
    
    lgp = LatentGP(fx, lik)
    @test typeof(lgp) <: LatentGP
    @test typeof(lgp.fx) <: AbstractGPs.FiniteGP
    f = rand(10)
    @test typeof(logpdf(lgp, (f=f, y=y))) <: Real
    @test typeof(rand(lgp)) <: NamedTuple{(:f, :y)}
end
