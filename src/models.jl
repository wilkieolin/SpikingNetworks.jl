"""
Arbitrary neuron model interface — replaces the hard-coded R&F bridge.
User defines:
  - update equation f(z, I, t, params; connections=W)
  - detect_spike(z, t) -> SpikeTrain
"""

abstract type AbstractNeuronModel end

struct SpikingArgs
    leakage::Float32
    t_period::Float32
    t_window::Float32
    spk_scale::Float32
    steepness::Float32
    threshold::Float32
    spike_kernel::Union{Symbol, Function, SpikeKernel}
    solver
    solver_args::Dict
    warmup_periods::Int
end

function SpikingArgs(; leakage::Real=-0.2f0, t_period::Real=1.0f0, t_window::Real=0.01f0,
                      spk_scale::Real=1.0f0, steepness::Real=0.05f0, threshold::Real=0.001f0,
                      spike_kernel=:gaussian, solver=Tsit5(),
                      solver_args=Dict(:dt=>0.01f0, :adaptive=>false),
                      warmup_periods::Integer=0)
    SpikingArgs(Float32(leakage), Float32(t_period), Float32(t_window), Float32(spk_scale),
                Float32(steepness), Float32(threshold), spike_kernel, solver, solver_args, Int(warmup_periods))
end

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
                           callback=nothing,
                           spike_args::SpikingArgs=SpikingArgs())
    function dzdt(z, p, t)
        I = input_current(t)
        dz = similar(z)
        update_equation!(dz, z, I, t, p, model; connections=connections)
        return dz
    end
    return neuron_bank(z0, dzdt, model; tspan=tspan, spk_args=spike_args, callback=callback)
end
