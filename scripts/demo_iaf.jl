using Pkg
# Use local project environment
Pkg.activate("/app")
push!(LOAD_PATH, "/app/src")

using SpikingNetworks
using Random

# Create 20-neuron network with Erdos-Renyi random connections (p=0.15)
model = IntegrateAndFire(20, 0.15f0)

println("Neuron model: ", typeof(model))
println("Connection matrix W size: ", size(model.W))
println("Connection density: ", sum(model.W .!= 0) / length(model.W))
println("Leakage lambda: ", model.lambda)

# Random initial real potentials
z0 = randn(Float32, 20)

# Constant current input for demonstration
I_fn(t) = Float32.(0.1 .* randn(20))

# Simulate over 5 seconds
sol = simulate_network(model, z0, I_fn, (0.0f0, 5.0f0))
println("Simulation completed. Solution saved at ", length(sol.t), " time points.")

# Detect spikes at final time using user-defined detector
spikes = detect_spike(sol.u[end], sol.t[end], 0.8f0; model=model)
println("Detected ", length(spikes.times), " spikes at t=", sol.t[end])
