using Test, SpikingNetworks

function types_tests()
    @testset "Types" begin
        sc = SpikingCall(random_spike_train((2,2); n_spikes=1), (0.0f0, 1.0f0))
        @test sc isa SpikingCall
    end
end
