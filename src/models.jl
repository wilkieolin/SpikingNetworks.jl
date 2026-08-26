"""
Arbitrary neuron model interface — replaces the hard-coded R&F bridge.
User defines:
  - update equation f(z, I, t, params; connections=W)
  - detect_spike(z, t) -> SpikeTrain
"""

abstract type AbstractNeuronModel end

# Protocol: user must implement for their model
function update_equation!(dz, z, I, t, params, model::AbstractNeuronModel; connections=nothing)
    error("update_equation! not implemented for $(typeof(model)). " *
          "Expected (dz, z, I, t, params; connections=W) -> nothing.")
end

# User must implement detect_spike for their concrete model.
# Example for IntegrateAndFire is shown in demo_iaf.jl.

struct GenericNeuronModel <: AbstractNeuronModel
    dims::Tuple{Int,Vararg{Int}}
    params::NamedTuple
end

# Default generic update: dz/dt = self_term(z) + spike_drive(I, connections)
function update_equation!(dz, z, I, t, params, model::GenericNeuronModel; connections=nothing)
    # Self-dependency (must be provided by user params or overridden)
    # This is a placeholder — real use requires user-defined method.
    # We expose connections matrix for arbitrary connectivity.
    if connections !== nothing
        # connections is an antidiagonal / arbitrary matrix linking neuron potentials
        # to current drive. User embeds dynamics here.
        I_connected = connections * I
    else
        I_connected = I
    end
    # Placeholder dynamics: user should overload this.
    dz .= params.self_decay .* z .+ I_connected
end

# Network simulation over arbitrary neuron models
function simulate_network(model::AbstractNeuronModel,
                           z0::AbstractArray,
                           input_current::Function,
                           tspan::Tuple{Float32, Float32};
                           connections=nothing,
                           spike_args::SpikingArgs=SpikingArgs())
    function dzdt(z, p, t)
        I = input_current(t)
        dz = similar(z)
        update_equation!(dz, z, I, t, p, model; connections=connections)
        return dz
    end
    return oscillator_bank(z0, dzdt, model; tspan=tspan, spk_args=spike_args)
end
