@testset "common_covmat_ops" begin
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


    # Set up some matrices and factorisations.
    rng, N, N′, P, Q = MersenneTwister(123456), 5, 3, 6, 2
    B = randn(rng, N, N)
    A_ = B' * B + UniformScaling(1e-6)
    A = cholesky(A_)
    x, y, z = randn(rng, N), randn(rng, N′), randn(rng, N)
    X, Y = randn(rng, N, P), randn(rng, N, Q)
    Z = randn(rng, N, P)
    A_1_ = exp(randn(rng))
    A_1 = cholesky(A_1_)

    # Specialised matrix operations.
    @test Xt_A_X(A, x) isa Real
    @test Xt_A_X(A, x) ≈ x' * A_ * x

    @test Xt_A_X(A_1, x') isa Symmetric
    @test Xt_A_X(A_1, x') ≈ x * A_1_ * x'

    @test Xt_A_X(A, X) isa Symmetric
    @test Xt_A_X(A, X) ≈ X' * A_ * X

    @test Xt_A_Y(x, A, z) isa Real
    @test Xt_A_Y(x, A, z) ≈ x' * A_ * z

    @test Xt_A_Y(x', A_1, y') isa Matrix
    @test Xt_A_Y(x', A_1, y') ≈ x * A_1_ * y'

    @test Xt_A_Y(X, A, Y) isa Matrix
    @test Xt_A_Y(X, A, Y) ≈ X' * A_ * Y

    @test Xt_invA_X(A, x) isa Real
    @test Xt_invA_X(A, x) ≈ x' * (A \ x)

    @test Xt_invA_X(A, X) isa Symmetric
    @test Xt_invA_X(A, X) ≈ X' * (A \ X)

    @test Xt_invA_Y(X, A, Y) isa Matrix
    @test Xt_invA_Y(X, A, Y) ≈ X' * (A \ Y)

    @test diag_At_A(x) ≈ [x'x]
    @test diag_At_A(X) ≈ diag(X'X)

    @test diag_At_B(x, z) ≈ [x'z]
    @test diag_At_B(X, Z) ≈ diag(X'Z)

    @test diag_Xt_A_X(A, x) ≈ [Xt_A_X(A, x)]
    @test diag_Xt_A_X(A, X) ≈ diag(Xt_A_X(A, X))

    @test diag_Xt_A_Y(x, A, z) ≈ [x' * A_ * z]
    @test diag_Xt_A_Y(X, A, Z) ≈ diag(Xt_A_Y(X, A, Z))

    @test diag_Xt_invA_X(A, X) ≈ diag(Xt_invA_X(A, X))

    @test diag_Xt_invA_Y(X, A, Z) ≈ diag(Xt_invA_Y(X, A, Z))

    @test Xtinv_A_Xinv(A, A) isa Symmetric
    @test Xtinv_A_Xinv(A, A) ≈ A \ (A \ A_)'

    @testset "tr_At_A" begin
        rng, P = MersenneTwister(123456), 11
        X = randn(rng, P, P)

        @test tr_At_A(X) ≈ tr(X'X)
        adjoint_test(tr_At_A, randn(rng), X)
    end
end
