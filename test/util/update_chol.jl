@testset "update_chol" begin
    X = rand(5)
    y = rand(5)

    k = SqExponentialKernel()

    C = kernelmatrix(k, X, X)
    U = cholesky(C).U
    C11 = kernelmatrix(k, X[1:3], X[1:3])
    chol1 = cholesky(C11)
    C22 = kernelmatrix(k, X[4:5], X[4:5])
    C12 = kernelmatrix(k, X[1:3], X[4:5])
    chol = AbstractGPs.update_chol(chol1, C12, C22)

    @test U ≈ chol.U atol=1e-5
    @test chol1.U ≈ chol.U[1:3, 1:3] atol=1e-5
end


