@testset "mogps" begin
    N = 10
    in_dim = 3
    out_dim = 5
    x = [rand(in_dim) for _ in 1:N]
    y = [rand(out_dim) for _ in 1:N]

    X, Y =  mo_transform(x, y, out_dim)
    @test length(X) == length(Y)
    @test (x, y) == mo_inverse_transform(X, Y)
    @test (x, y) == mo_inverse_transform(collect(X), Y, out_dim)
    @test y == mo_inverse_transform(Y, out_dim)
end