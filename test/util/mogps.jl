@testset "mogps" begin
    N = 10
    in_dim = 3
    out_dim = 5
    x = [rand(in_dim) for _ in 1:N]
    y = [rand(out_dim) for _ in 1:N]

    # Matrix inputs
    xm = hcat(x...)
    ym = hcat(y...)

    @testset "MOutput" begin
        mout = MOutput(y)
        @test length(mout) == out_dim * N
        
        @test size(mout) == (out_dim * N,)
        @test size(mout, 1) == out_dim * N
        @test size(mout, 2) == 1
        
        @test lastindex(mout) == out_dim * N
        @test firstindex(mout) == 1
        
        @test mout[1] == y[1][1]
        @test mout[12] == y[2][2]
        @test mout[out_dim * N] == y[N][out_dim]

        @test mout isa MOutput{<:Real, <:AbstractVector}
        @test mout isa AbstractVector{<:Real}
        @test mout ≈ vcat(([yi[i] for yi in y] for i in 1:out_dim)...)
        
        # Matrix Input
        mout2 = MOutput(ym)
        @test mout2 isa MOutput{<:Real, <:ColVecs}
        @test mout2 isa AbstractVector{<:Real}
        @test mout2 ≈ vcat(([yi[i] for yi in y] for i in 1:out_dim)...)
    end

    X, Y =  mo_transform(x, y, out_dim)
    @test length(X) == length(Y)
    @test (x, y) == mo_inverse_transform(X, Y)
    @test (x, y) == mo_inverse_transform(collect(X), collect(Y), out_dim)

    
    Xm, Ym = mo_transform(xm, ym)
    @test length(Xm) == length(Ym)
    @test (X, Y) == (Xm, Ym)
    @test size(ym) == (out_dim, N)
end