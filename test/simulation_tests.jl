using Test, SpikingNetworks
using Random: Xoshiro

function simulation_tests()
    @testset "Simulation" begin
        n = 8
        tspan = (0.0f0, 2.0f0)
        model = IntegrateAndFire(n, 0.2f0; threshold=0.15f0, seed=Xoshiro(1))
        train = poisson_spike_train(CurrentGenerator(5.0f0, 0.001f0, n, Xoshiro(2)), tspan)
        z0 = zeros(Float32, n)

        function run(; callback=nothing)
            args = SpikingArgs(spike_kernel=DeltaKernel(0.02f0), spk_scale=0.5f0,
                               solver_args=Dict(:dt => 0.005f0, :adaptive => false))
            I_fn = t -> spike_current(train, t, args)
            return simulate_network(model, z0, I_fn, tspan; spike_args=args, callback=callback)
        end

        @test length(train) > 0

        cb, rec = reset_callback(model)
        sol = run(callback=cb)
        out = SpikeTrain(rec, (n,))

        # The headline regression: the demo used to report zero spikes.
        @test length(out) > 0
        @test length(out) == length(rec.times)
        @test all(t -> tspan[1] <= t <= tspan[2], out.times)
        @test all(i -> 1 <= i[1] <= n, out.indices)

        # The callback actually holds the potential at threshold. The old in-RHS
        # mutation could not guarantee this because it wrote to stage vectors.
        u = stack(sol.u, dims=2)
        @test maximum(u) <= model.threshold + 1.0f-3

        # Deterministic RHS => identical results on a rerun
        cb2, rec2 = reset_callback(model)
        run(callback=cb2)
        @test rec2.times == rec.times
        @test rec2.indices == rec.indices

        # Without the reset, the potential free-runs and the fixed peak detector
        # finds real local maxima above threshold.
        sol_free = run()
        _, ref_times = find_spikes_ref(stack(sol_free.u, dims=2), sol_free.t,
                                       model.threshold; dim=2)
        @test length(ref_times) > 0
        @test maximum(stack(sol_free.u, dims=2)) > model.threshold
    end
end
