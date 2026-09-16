# SpikingNetworks.jl

Standalone Julia package for simulating networks of arbitrary spiking neurons.

Forked from `PhasorNetworks.jl` — breaks off spiking dynamics from the Resonate-and-Fire / phase-SSM bridge. No Lux, no VSA, no phase-locked SSM.

## Design

- `SpikeTrain` / `SpikeTrainGPU`: spike event representation.
- `SpikingArgs`: solver parameters, kernel settings.
- `neuron_bank`: generic ODE solver (`dz/dt = f(z, I, t, params)`).
- `update_equation!`: user-overridable protocol for arbitrary neuron dynamics, with optional connection matrix `W`.
- `detect_spike`: user-provided spike detection on potentials.
- Supports continuous current injection (`CurrentCall`) and discrete spike events (`SpikingCall`).

## Quick Start

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

## Documentation

- [Getting Started](@ref) — Environment setup, Julia, Jupyter, VS Code
- [Tutorial: Integrate-and-Fire Network](@ref) — Interactive demo with kernel comparison
- [Custom Neuron Models](@ref) — Protocol walkthrough for defining new models
- [Callbacks & Recording](@ref) — Spike recording, reset callbacks, current generators
- [API Reference](@ref) — Complete function/type reference

## Demos

| Method | Command |
|--------|---------|
| Script (headless) | `julia --project=. scripts/demo_iaf.jl` |
| Notebook (interactive) | `jupyter notebook notebooks/` → open `demo_iaf_interactive.ipynb` |
| VS Code | Open notebook → Run All (kernel: **SpikingNetworks**) |

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```