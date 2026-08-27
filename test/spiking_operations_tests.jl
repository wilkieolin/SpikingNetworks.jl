using Test, SpikingNetworks

function spiking_operations_tests()
    @testset "Spiking Operations" begin
        train = SpikeTrain([CartesianIndex(1,1)], Float32[0.1], (2,2), 0.0f0)
        @test length(train) == 1
        @test size(train) == (2,2)
    end
end
