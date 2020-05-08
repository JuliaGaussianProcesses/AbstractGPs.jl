@testset "sequential_conditioning" begin
    X = rand(5)
    y = rand(5)

    f1 = GP(SqExponentialKernel())
    fx1 = f(X[1:3], 0.1)
    p_fx1 = posterior(fx1, y[1:3])
    p_p_fx1 = posterior(p_fx1(X[4:5]), y[4:5])

    # Base.show(stdout, "text/plain", p_p_fx1.data.C.U)
    # println()
    # Base.show(stdout, "text/plain", p_p_fx1.data.α)
    # println()


    f2 = GP(SqExponentialKernel())
    fx2 = f(X, 0.1)
    p_fx2 = posterior(fx2, y)
    # Base.show(stdout, "text/plain", p_fx2.data.C.U)
    # println()
    # Base.show(stdout, "text/plain", p_fx2.data.α)
    # println()
end

