# SpikingNetworks.jl

Standalone Julia package for simulating networks of arbitrary spiking neurons.

Forked from `PhasorNetworks.jl` — breaks off spiking dynamics from the Resonate-and-Fire / phase-SSM bridge. No Lux, no VSA, no phase-locked SSM.

## Design

- `SpikeTrain` / `SpikeTrainGPU`: spike event representation.
- `SpikingArgs`: solver parameters, kernel settings.
- `oscillator_bank`: generic ODE solver (`dz/dt = f(z, I, t, params)`).
- `update_equation!`: user-overridable protocol for arbitrary neuron dynamics, with optional connection matrix `W`.
- `detect_spike`: user-provided spike detection on potentials.
- Supports continuous current injection (`CurrentCall`) and discrete spike events (`SpikingCall`).

## Usage

```julia
using SpikingNetworks

# Define custom neuron model
model = GenericNeuronModel(dims=(10, 10), params=(self_decay=-0.2,))

# Provide update equation and spike detector
function update_equation!(dz, z, I, t, params, model; connections=W)
    dz .= params.self_decay .* z .+ W * I + nonlinear_spike_term(z)
end

detect_spike(z, t, threshold) = ... # user-defined

# Simulate
sol = simulate_network(model, z0, I_fn, (0f0, 10f0); connections=W)
```
