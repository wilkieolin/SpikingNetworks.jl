"""
Demonstration: generic integrate-and-fire neuron in real domain.
No reset mechanism, Erdos-Renyi random connections, leakage dynamics.
"""

using Random

struct IntegrateAndFire <: AbstractNeuronModel
    dims::Int           # number of neurons
    lambda::Float32     # leakage rate (same as old R&F real part)
    W::Matrix{Float32}  # Erdos-Renyi connection matrix
end

function IntegrateAndFire(n_neurons::Int, p::Float32=0.1)
    # Erdos-Renyi random connection matrix (antidiagonal allowed by general matrix)
    W = Float32.(rand(n_neurons, n_neurons) .< p) .* randn(Float32, n_neurons, n_neurons)
    model = IntegrateAndFire(n_neurons, -0.2f0, W)
    return model
end

# Real-domain update: dz/dt = lambda*z + W*I(t)  (leakage + arbitrary connection drive)
function update_equation!(dz, z, I, t, params, model::IntegrateAndFire; connections=nothing)
    W_use = connections !== nothing ? connections : model.W
    # Self-dependency (leakage) + nonlinear spike-transformed input via connection matrix
    # I is the continuous current from spike inputs; W transforms potentials across network
    dz .= params.lambda .* z .+ W_use * I
end

function detect_spike(z, t, threshold; model::IntegrateAndFire=IntegrateAndFire(1))
    # Simple threshold crossing: detect when potential crosses threshold upward
    above = z .> threshold
    indices = findall(above)
    times = fill(Float32(t), length(indices))
    # Construct minimal SpikeTrain shaped (n_neurons,) for simplicity
    shape = (length(z),)
    cart = [CartesianIndex(i) for i in indices]
    return SpikeTrain(cart, times, shape, 0.0f0)
end
