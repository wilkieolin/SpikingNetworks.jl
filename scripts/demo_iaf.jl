using Pkg
# Use parent project environment that has dependencies installed
Pkg.activate(dirname(dirname(@__FILE__)))

using SpikingNetworks
using Random: Xoshiro

n_neurons = 20
t_span = (0.0f0, 5.0f0)
dt = 0.005f0

# --- network ------------------------------------------------------------------
# 20 neurons, Erdos-Renyi random connections (p = 0.15)
model = IntegrateAndFire(n_neurons, 0.15f0; threshold=0.15f0, seed=Xoshiro(42))

println("Neuron model: ", typeof(model))
println("Connection matrix W size: ", size(model.W))
println("Connection density: ", sum(model.W .!= 0) / length(model.W))
println("Leakage lambda: ", model.lambda, ", threshold: ", model.threshold)

# --- input --------------------------------------------------------------------
# Sample a Poisson source ONCE into a SpikeTrain. The train is then a fixed, known
# object that every kernel below convolves; the ODE right-hand side stays a
# deterministic function of t, which is what the solver (and root finding) needs.
input_gen = CurrentGenerator(5.0f0, 0.001f0, n_neurons, Xoshiro(7))
input_train = poisson_spike_train(input_gen, t_span)
println("Input ", input_train)

z0 = zeros(Float32, n_neurons)

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

# A kernel narrower than the solver step can be stepped straight over, so keep the
# delta-like kernel a few dt wide. Both kernels carry unit area, so widening smears
# the same total charge over more time instead of also strengthening the drive.
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

println("\nBoth runs share the same ", length(input_train), "-spike input train.")
println("Equal mean current, very different peak current: the wide Gaussian spreads")
println("each spike's charge over ~1 s instead of 20 ms.")
