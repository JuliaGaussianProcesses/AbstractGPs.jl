@testset "Sampling" begin
    rng = Xoshiro(123456)

    nx = 8
    x1 = collect(range(0, 2; length=nx))
    y1 = rand(nx)

    k = GaussianKernel()
    g1 = GP(k)
    g1x1 = g1(x1, 0.1)
    pg1 = posterior(g1x1, y1)

    # # FunctionSpace
    @testset "Constructors" begin
        @testset "Cholesky" begin
            @test CholeskySampling() isa CholeskySampling{Conditional,Xoshiro}
            @test CholeskySampling(Conditional) isa CholeskySampling{Conditional,Xoshiro}
            @test CholeskySampling(Conditional, Xoshiro) isa
                CholeskySampling{Conditional,Xoshiro}
            @test CholeskySampling(Conditional, MersenneTwister) isa
                CholeskySampling{Conditional,MersenneTwister}
            @test CholeskySampling(Independent) isa CholeskySampling{Independent,Xoshiro}
        end
        @testset "RFF" begin
            @test RFFSampling(10) isa RFFSampling
            # Testing other RFFs? Needs re-export from KernelSpectralDensities
        end
        @testset "Pathwise" begin
            @test PathwiseSampling(RFFSampling(10), CholeskySampling()) isa
                PathwiseSampling{<:RFFSampling,<:CholeskySampling}
            @test PathwiseSampling(10) isa PathwiseSampling
        end
    end

    @testset "Basic Functional" begin
        function test_basic_fun(gp, method)
            gps = GPSampler(gp, method)
            gps1 = rand(gps)
            @test gps1(0.4) isa Float64
            @test gps1([0.6, 0.7]) isa Vector{Float64}
        end
        @testset "Cholesky" begin
            test_basic_fun(g1, CholeskySampling())
            test_basic_fun(pg1, CholeskySampling())
        end

        @testset "RFF" begin
            test_basic_fun(g1, RFFSampling(20))
            test_basic_fun(pg1, RFFSampling(20))
        end

        @testset "Pathwise" begin
            method = PathwiseSampling(RFFSampling(20), CholeskySampling())
            test_basic_fun(pg1, method)
        end
    end

    # ## Accuracy test

    # compute error between empirical and analytical version
    function eval_res(x, gp, resv)
        empres = cov(resv)
        trueres = cov(gp, x)
        return norm(empres .- trueres)
    end

    # Evaluate sample all at once
    function oneshot_error(x, gp, gps, n)
        resv = [rand(gps)(x) for _ in 1:n]
        return eval_res(x, gp, resv)
    end

    # Evaluate samples one by one
    function onebyone(gpsampler, x)
        gps = rand(gpsampler)
        y = [gps(xi) for xi in x]
        return y
    end
    function onebyone_error(x, gp, gps, n)
        resv = [onebyone(gps, x) for _ in 1:n]
        return eval_res(x, gp, resv)
    end

    function grid_test(gp, x, nv, method, evalfun)
        gps = GPSampler(gp, method)

        res = [evalfun(x, gp, gps, n) for n in nv]
        return all(diff(res) .< 0)
    end

    x = collect(range(0, 2; length=9))
    nv = [10, 100, 1000]

    @testset "Correctness" begin
        @testset "FunctionSpace" begin
            @testset "Prior, FullMemory" begin
                @test grid_test(g1, x, nv, CholeskySampling(Conditional), oneshot_error)
                @test grid_test(g1, x, nv, CholeskySampling(Conditional), onebyone_error)
            end

            @testset "Posterior, FullMemory" begin
                @test grid_test(pg1, x, nv, CholeskySampling(Conditional), oneshot_error)
                @test grid_test(pg1, x, nv, CholeskySampling(Conditional), onebyone_error)
            end

            @testset "Prior, NoMemory" begin
                @test grid_test(g1, x, nv, CholeskySampling(Independent), oneshot_error)
                # @test grid_test(g1, x, nv, FunctionSpace(NoMemory), onebyone_error)
            end

            @testset "Posterior, NoMemory" begin
                @test grid_test(pg1, x, nv, CholeskySampling(Independent), oneshot_error)
                # @test grid_test(g1, x, nv, FunctionSpace(NoMemory), onebyone_error)
            end
        end

        @testset "WeightSpace" begin
            l = 80
            @testset "Prior, DoubleRFF" begin
                wsp = RFFSampling(l)
                @test grid_test(g1, x, nv, wsp, oneshot_error)
                @test grid_test(g1, x, nv, wsp, onebyone_error)
            end
            @testset "Posterior, DoubleRFF" begin
                wsp = RFFSampling(l)
                @test grid_test(pg1, x, nv, wsp, oneshot_error)
                @test grid_test(pg1, x, nv, wsp, onebyone_error)
            end
        end

        l = 80
        @testset "Posterior, DoubleRFF" begin
            method = PathwiseSampling(RFFSampling(l), CholeskySampling())
            @test grid_test(pg1, x, nv, method, oneshot_error)
            @test grid_test(pg1, x, nv, method, onebyone_error)
        end
    end
end