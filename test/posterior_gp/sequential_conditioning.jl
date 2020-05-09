@testset "sequential_conditioning" begin
    X = rand(5)
    y = rand(5)

    @testset "posterior" begin
        # Tests on posterior

        f1 = GP(SqExponentialKernel())
        p_fx1 = posterior(f1(X[1:3], 0.1), y[1:3])
        p_p_fx1 = posterior(p_fx1(X[4:5], 0.1), y[4:5])

        f2 = GP(SqExponentialKernel())
        p_fx2 = posterior(f2(X, 0.1), y)

        @test p_p_fx1.data.C.U ≈ p_fx2.data.C.U atol=1e-5
        @test p_p_fx1.data.α ≈ p_fx2.data.α atol=1e-5
        @test p_p_fx1.data.x ≈ p_fx2.data.x atol=1e-5
    end

    @testset "update_chol" begin
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
end

