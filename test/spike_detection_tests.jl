using Test, SpikingNetworks

function spike_detection_tests()
    @testset "Spike Detection" begin
        t = Float32.(collect(0.0:0.001:3.0))

        @testset "find_spikes_ref finds local maxima" begin
            # Two channels, 3 full cycles => 3 peaks each. Channel 2 is scaled below
            # the threshold, so only channel 1 should be reported.
            u = stack([Float32.(sin.(2pi .* t)), Float32.(0.1 .* sin.(2pi .* t))], dims=1)
            channels, times = find_spikes_ref(u, t, 0.5f0; dim=2)
            @test length(times) == 3
            @test all(c -> c == CartesianIndex(1), channels)
            # peaks of sin(2*pi*t) sit at t = 0.25, 1.25, 2.25
            @test all(abs.(sort(times) .- Float32[0.25, 1.25, 2.25]) .< 0.01f0)
        end

        @testset "peaks below threshold are rejected" begin
            u = reshape(Float32.(0.1 .* sin.(2pi .* t)), 1, :)
            _, times = find_spikes_ref(u, t, 0.5f0; dim=2)
            @test isempty(times)
        end

        @testset "regression: zero crossings are not spikes" begin
            # The old detector looked for sign changes of u itself, so a signal that
            # crosses zero but peaks below threshold could never be distinguished
            # from one that peaks above it. Both cases must now behave correctly.
            big = reshape(Float32.(sin.(2pi .* t)), 1, :)
            small = reshape(Float32.(0.2 .* sin.(2pi .* t)), 1, :)
            @test length(find_spikes_ref(big, t, 0.5f0; dim=2)[2]) == 3
            @test isempty(find_spikes_ref(small, t, 0.5f0; dim=2)[2])
        end

        @testset "monotonic signal has no peaks" begin
            u = reshape(Float32.(t), 1, :)
            _, times = find_spikes_ref(u, t, 0.5f0; dim=2)
            @test isempty(times)
        end
    end
end
