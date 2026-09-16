using Test, SpikingNetworks

# Trapezoid integral of a kernel over its support, accumulated in Float64 so the
# quadrature error stays well below the tolerance we are actually testing.
function kernel_area(k; n=100_001)
    r = Float64(support(k))
    lo = k isa AlphaKernel ? 0.0 : -r
    grid = range(lo, r; length=n)
    step = grid[2] - grid[1]
    vals = [Float64(k(Float32(x))) for x in grid]
    return step * (sum(vals) - 0.5 * (vals[1] + vals[end]))
end

function kernel_tests()
    @testset "Spike Kernels" begin
        @testset "unit area" begin
            # Each kernel is normalised so widening smears charge rather than adding it.
            # Tolerance is set by truncation, not normalisation: AlphaKernel's 8*tau
            # cut discards (1 + 8)*exp(-8) ~ 0.3% of the tail, the largest of the four.
            for k in (DeltaKernel(0.02f0), GaussianKernel(0.25f0),
                      RaisedCosineKernel(0.1f0), AlphaKernel(0.05f0))
                @test kernel_area(k) ≈ 1.0 atol = 5.0e-3
            end
            # Widening does not change the area: that is the whole point
            @test kernel_area(GaussianKernel(1.0f0)) ≈ kernel_area(GaussianKernel(0.05f0)) atol = 1.0e-3
        end

        @testset "support gates" begin
            g = GaussianKernel(0.1f0)
            @test support(g) == 0.4f0
            @test g(0.0f0) > g(0.05f0) > g(0.2f0)
            @test RaisedCosineKernel(0.1f0)(0.11f0) == 0.0f0
            @test DeltaKernel(0.02f0)(0.011f0) == 0.0f0
            @test DeltaKernel(0.02f0)(0.009f0) == 50.0f0   # unit area => 1/width
            @test AlphaKernel(0.05f0)(-0.01f0) == 0.0f0   # causal
        end

        @testset "resolve_kernel" begin
            args = SpikingArgs(t_window=0.05f0)
            @test resolve_kernel(args) isa GaussianKernel          # :gaussian default
            @test resolve_kernel(SpikingArgs(spike_kernel=:delta)) isa DeltaKernel
            @test resolve_kernel(SpikingArgs(spike_kernel=:raised_cosine)) isa RaisedCosineKernel
            @test resolve_kernel(SpikingArgs(spike_kernel=GaussianKernel(1.0f0))) isa GaussianKernel
            @test resolve_kernel(SpikingArgs(spike_kernel=dt -> 1.0f0)) isa FunctionKernel
            @test_throws ErrorException resolve_kernel(SpikingArgs(spike_kernel=:nope))
        end

        @testset "spike_current accumulates overlaps" begin
            # Regression: `current[inds] .+= v` silently dropped coincident spikes
            k = GaussianKernel(0.1f0)
            one_spike = SpikeTrain([CartesianIndex(1)], Float32[0.5], (3,), 0.0f0)
            two_spikes = SpikeTrain([CartesianIndex(1), CartesianIndex(1)],
                                    Float32[0.5, 0.5], (3,), 0.0f0)
            single = spike_current(one_spike, 0.5f0, k)
            double = spike_current(two_spikes, 0.5f0, k)
            @test double[1] ≈ 2.0f0 * single[1]
            @test single[2] == 0.0f0   # other channels untouched
        end

        @testset "spike_current respects support and scale" begin
            k = DeltaKernel(0.02f0)
            train = SpikeTrain([CartesianIndex(2)], Float32[1.0], (3,), 0.0f0)
            @test spike_current(train, 1.0f0, k)[2] == 50.0f0
            @test spike_current(train, 1.5f0, k)[2] == 0.0f0   # outside support
            @test spike_current(train, 1.0f0, k; scale=0.5f0)[2] == 25.0f0
            # SpikingArgs path applies spk_scale
            args = SpikingArgs(spike_kernel=k, spk_scale=2.0f0)
            @test spike_current(train, 1.0f0, args)[2] == 100.0f0
        end
    end
end
