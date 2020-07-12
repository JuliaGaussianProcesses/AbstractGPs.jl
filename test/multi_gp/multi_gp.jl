@testset "multi_gp" begin
    @testset "MultiGPInput" begin
        x =  rand(5)
        mgpi = MultiGPInput(x, 3)
        
        @test length(mgpi) == 15
        @test size(mgpi) == (15,)
        @test size(mgpi, 1) == 15
        @test size(mgpi, 2) == 1
        @test mgpi[2] == (x[2], 1)
        @test mgpi[5] == (x[5], 1)
        @test mgpi[7] == (x[2], 2)
        @test all([(x_, i) for i in 1:3 for x_ in x ] .== mgpi)
    end
end