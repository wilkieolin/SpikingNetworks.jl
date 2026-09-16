# Tutorial: Integrate-and-Fire Network

This tutorial demonstrates a real-domain integrate-and-fire network with Erdos-Renyi connections. Two things are shown:

1. **Getting a spike train out.** Threshold reset and spike events are handled by a solver callback, so spike times are root-found (exact) rather than read off the saved trajectory.
2. **Swapping the input spike kernel.** The same input spike train is convolved with a narrow delta-like impulse and with a very wide Gaussian. Both kernels carry unit area, so widening smears the same total charge over more time rather than also turning up the drive.

## Setup

```julia
using SpikingNetworks, Plots
using Random: Xoshiro

n_neurons = 20
t_span = (0.0f0, 5.0f0)
dt = 0.005f0
```

## Define the Neuron Model

Integrate-and-fire neurons wired into an Erdos-Renyi random graph.

```julia
model = IntegrateAndFire(n_neurons, 0.15f0; threshold=0.15f0, seed=Xoshiro(42))
println("Connection density: ", sum(model.W .!= 0) / length(model.W))
println("Leakage lambda: ", model.lambda)
println("Threshold: ", model.threshold, ", rest: ", model.rest)
```

## Define Input Current

Sample a Poisson source **once** into a `SpikeTrain`. The train is then a fixed, known object that every kernel below convolves; the ODE right-hand side stays a deterministic function of `t`, which is what the solver (and root finding) needs.

```julia
input_gen = CurrentGenerator(5.0f0, 0.001f0, n_neurons, Xoshiro(7))
input_train = poisson_spike_train(input_gen, t_span)
println("Input ", input_train)

z0 = zeros(Float32, n_neurons)
```

## Run with Different Kernels

```julia
"""
Run the network once. The kernel is the only thing that varies between runs.
Returns the solution, the callback-recorded output train, and the injected current.
"""
function run_kernel(kernel; scale=0.5f0, reset=true)
    args = SpikingArgs(spike_kernel=kernel, spk_scale=scale,
                       solver_args=Dict(:dt => dt, :adaptive => false))
    I_fn = t -> spike_current(input_train, t, args)
    if reset
        cb, rec = reset_callback(model)
        sol = simulate_network(model, z0, I_fn, t_span; spike_args=args, callback=cb)
        return sol, SpikeTrain(rec, (n_neurons,)), I_fn
    else
        sol = simulate_network(model, z0, I_fn, t_span; spike_args=args)
        return sol, nothing, I_fn
    end
end
```

A kernel narrower than the solver step can be stepped straight over, so keep the delta-like kernel a few `dt` wide. Both kernels carry unit area, so widening smears the same total charge over more time instead of also strengthening the drive.

```julia
kernels = ["delta    (width 0.02)" => DeltaKernel(0.02f0),
           "gaussian (sigma 0.25)" => GaussianKernel(0.25f0)]

for (name, kernel) in kernels
    sol, out_train, I_fn = run_kernel(kernel)
    u = stack(sol.u, dims=2)
    I_grid = stack([I_fn(t) for t in t_span[1]:dt:t_span[2]], dims=2)

    # Same input, no reset: the potential free-runs, so find_spikes_ref has real
    # local maxima to find. (With the reset active, every peak is clipped exactly
    # at threshold and a `> threshold` test cannot fire — the callback is the
    # spike source there, not the peak detector.)
    sol_free, _, _ = run_kernel(kernel; reset=false)
    _, ref_times = find_spikes_ref(stack(sol_free.u, dims=2), sol_free.t,
                                   model.threshold; dim=2)

    isis = let ts = sort(out_train.times)
        length(ts) > 1 ? diff(ts) : Float32[]
    end

    println("\n--- ", name, " ---")
    println("  injected current  : mean ", round(sum(I_grid) / length(I_grid), digits=3),
            ", peak ", round(maximum(I_grid), digits=2))
    println("  saved time points : ", length(sol.t))
    println("  peak potential    : ", round(maximum(u), digits=4),
            "  (threshold ", model.threshold, ")")
    println("  output ", out_train)
    println("  mean inter-spike interval: ", round(sum(isis) / length(isis), digits=4), " s")
    println("  free-running run, find_spikes_ref: ", length(ref_times), " peaks above threshold")
end
```

Both runs share the same input spike train. Equal mean current, very different peak current: the wide Gaussian spreads each spike's charge over ~1 s instead of 20 ms.

## Cross-check: `find_spikes_ref`

`find_spikes_ref` is the reference detector — local maxima of the trajectory above threshold. It is the right tool for a model with **no** reset. With the reset active every peak is clipped exactly *at* threshold, so a strict `> threshold` test cannot fire; there the callback is the spike source, not the detector.

Re-run without the callback so the potentials free-run, and the detector has real peaks to find.

```julia
for (name, kernel) in [("delta", DeltaKernel(0.02f0)), ("gaussian", GaussianKernel(0.25f0))]
    sol_free, _ = run_kernel(kernel; reset=false)
    u_free = stack(sol_free.u, dims=2)
    channels, times = find_spikes_ref(u_free, sol_free.t, model.threshold; dim=2)
    println(name, ": free-running peak ", round(maximum(u_free), digits=2),
            ", find_spikes_ref found ", length(times), " peaks above threshold")
end
```

## Summary

- The input train, the network and the solver are all deterministic — rerun this notebook and you get identical spike times.
- With unit-area kernels, widening the Gaussian leaves total charge and overall firing rate roughly unchanged, but smooths the potential trajectory and spreads each input spike's influence over ~1 s.
- Any callable with a `support` can be dropped in: `DeltaKernel`, `GaussianKernel`, `RaisedCosineKernel`, `AlphaKernel`, or `FunctionKernel(f, radius)` for something custom.