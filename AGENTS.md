# Repository Overview

## Project Description

**SpikingNetworks.jl** is a standalone Julia package for arbitrary spiking neuron dynamics. It breaks off generic ODE-mode spiking from phase-SSM systems and replaces hard-coded Resonate-and-Fire dynamics with a user-defined neuron model protocol.

- Main purpose: provide `update_equation!`, `detect_spike`, and `simulate_network` for any real-domain neuron model using DifferentialEquations.jl.
- Key goals: arbitrary differential updates (self-dependency + parameter/nonlinear spike drive + connection matrix `W`), custom spike detection on potentials, and efficient simulation via `oscillator_bank`.
- No Lux, VSA, phase scalar (`Phase`), SSM kernels, or phase-locked carrier dependencies.

### Key Technologies

- Julia 1.11+
- DifferentialEquations.jl (ODE solving)
- LinearAlgebra (connection matrices)
- Random (Erdos-Renyi initialization in demo)

## Architecture Overview

### Generic Neuron Protocol

Every concrete model extends `AbstractNeuronModel` and provides:

- `update_equation!(dz, z, I, t, params, model; connections=W)` — user-defined dynamics with arbitrary `W`.
- `detect_spike(z, t, threshold; model=...)` — user-defined threshold/crossing logic.

`simulate_network` wraps the dynamics in `oscillator_bank` and passes the model instance as params (`p`).

### Demonstration Model

`IntegrateAndFire` (in `demo_iaf.jl`): real domain, leakage `λ`, Erdos-Renyi random `W`, no reset mechanism.

## Directory Structure

```
SpikingNetworks.jl/
├── src/
│   ├── SpikingNetworks.jl   # module, exports
│   ├── types.jl             # SpikeTrain, SpikingArgs
│   ├── spiking.jl           # oscillator_bank, spike_current, find_spikes_ref
│   ├── models.jl            # AbstractNeuronModel, GenericNeuronModel, simulate_network
│   └── demo_iaf.jl          # IntegrateAndFire + Erdos-Renyi demo
├── scripts/
│   └── demo_iaf.jl          # executable demo script
├── notebooks/
│   └── demo_iaf_interactive.ipynb  # interactive Jupyter demo
└── AGENTS.md
```

## Main Components

| File | Responsibility |
|---|---|
| `types.jl` | `SpikeTrain`, `SpikingArgs`, `SpikingCall`, `CurrentCall` |
| `spiking.jl` | Generic `oscillator_bank`, `spike_current`, `find_spikes_ref` |
| `models.jl` | `AbstractNeuronModel`, protocol definitions, `simulate_network` |
| `demo_iaf.jl` | `IntegrateAndFire` concrete model and custom detectors |

## Key Entry Points

- Module: `src/SpikingNetworks.jl`
- Interactive: `notebooks/demo_iaf_interactive.ipynb`
- Script: `scripts/demo_iaf.jl`

## Development Workflow

```bash
julia --project=/home/wilkie/code/PhasorNetworks.jl
```
Load package via `push!(LOAD_PATH, "src")`, then `using SpikingNetworks`.

Run interactive demo with Jupyter (`jupyter notebook notebooks/`) or script (`include("scripts/demo_iaf.jl")`).

## Design / Code Style

- 4-space indentation, snake_case functions, PascalCase types.
- `Float32` for GPU/ODE compatibility.
- No formal linter; follow existing file patterns.
- All public exports listed in `src/SpikingNetworks.jl`.
- User dynamics are fully arbitrary; no hidden Lux or phase-SSM assumptions.
