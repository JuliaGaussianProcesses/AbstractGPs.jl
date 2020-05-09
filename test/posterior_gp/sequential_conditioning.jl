@testset "sequential_conditioning" begin
    X = rand(5)
    y = rand(5)

    f1 = GP(SqExponentialKernel())
    fx1 = f1(X[1:3], 0.1)
    p_fx1 = posterior(fx1, y[1:3])
    p_p_fx1 = posterior(p_fx1(X[4:5]), y[4:5])

    Base.show(stdout, "text/plain", p_p_fx1.data.C.U)
    println()
    Base.show(stdout, "text/plain", p_p_fx1.data.α)
    println()
    Base.show(stdout, "text/plain", p_p_fx1.data.x)
    println()


    f2 = GP(SqExponentialKernel())
    fx2 = f2(X, 0.1)
    p_fx2 = posterior(fx2, y)
    Base.show(stdout, "text/plain", p_fx2.data.C.U)
    println()
    Base.show(stdout, "text/plain", p_fx2.data.α)
    println()
    Base.show(stdout, "text/plain", p_fx2.data.x)
    println()

    # Tests on update_chol
    k = SqExponentialKernel()
    v1 = X

    C = kernelmatrix(k, v1, v1)
    U = cholesky(C).U
    Base.show(stdout, "text/plain", U)
    println()

    C11 = kernelmatrix(k, v1[1:3], v1[1:3])
    U11 = cholesky(C11).U

    C22 = kernelmatrix(k, v1[4:5], v1[4:5])

    C12 = kernelmatrix(k, v1[1:3], v1[4:5])

    U_cal = AbstractGPs.update_chol(U11, C12, C22)
    Base.show(stdout, "text/plain", U_cal)
    println()

    @test U ≈ U_cal atol=1e-5
    @test U11 ≈ U_cal[1:3, 1:3] atol=1e-5

end

