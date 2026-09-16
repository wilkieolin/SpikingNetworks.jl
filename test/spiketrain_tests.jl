using Test, SpikingNetworks

function spiketrain_tests()
    @testset "SpikeTrain" begin
        train = random_spike_train((4, 5); n_spikes=3, t_max=0.5f0)
        @test train.shape == (4, 5)
        @test length(train.times) == 3
        @test train.offset == 0.0f0
    end
end
