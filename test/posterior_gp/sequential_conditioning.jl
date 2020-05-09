@testset "sequential_conditioning" begin

    # Tests on posterior
    X = rand(5)
    y = rand(5)

    f1 = GP(SqExponentialKernel())
    fx1 = f1(X[1:3])
    p_fx1 = posterior(fx1, y[1:3])
    p_p_fx1 = posterior(p_fx1(X[4:5]), y[4:5])

    f2 = GP(SqExponentialKernel())
    fx2 = f2(X)
    p_fx2 = posterior(fx2, y)

    @test p_p_fx1.data.C.U ≈ p_fx2.data.C.U atol=1e-5
    @test p_p_fx1.data.α ≈ p_fx2.data.α atol=1e-5
    @test p_p_fx1.data.x ≈ p_fx2.data.x atol=1e-5

    # Tests on update_chol
    k = SqExponentialKernel()

    C = kernelmatrix(k, X, X)
    U = cholesky(C).U
    C11 = kernelmatrix(k, X[1:3], X[1:3])
    U11 = cholesky(C11).U
    C22 = kernelmatrix(k, X[4:5], X[4:5])
    C12 = kernelmatrix(k, X[1:3], X[4:5])
    U_cal = AbstractGPs.update_chol(U11, C12, C22)

    @test U ≈ U_cal atol=1e-5
    @test U11 ≈ U_cal[1:3, 1:3] atol=1e-5

end

