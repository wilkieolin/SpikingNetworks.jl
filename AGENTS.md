# Repository Overview

## Project Description

**SpikingNetworks.jl** is a standalone Julia package for arbitrary spiking neuron dynamics. It breaks off generic ODE-mode spiking from phase-SSM systems and replaces hard-coded Resonate-and-Fire dynamics with a user-defined neuron model protocol.

- Main purpose: provide `update_equation!`, `detect_spike`, and `simulate_network` for any real-domain neuron model using DifferentialEquations.jl.
- Key goals: arbitrary differential updates (self-dependency + parameter/nonlinear spike drive + connection matrix `W`), custom spike detection on potentials, pluggable temporal spike kernels, and efficient simulation via `neuron_bank`.
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
- `detect_spike(z, t, threshold, model)` — user-defined threshold/crossing logic.

`simulate_network` wraps the dynamics in `neuron_bank` and passes the model instance as params (`p`).
It accepts a `callback` kwarg that is forwarded to `solve`.

`update_equation!` must be a **pure** function of `(z, t)`. Never write to `z`: it is the
solver's state, or one of Tsit5's internal stage vectors, and mutating it is undefined
behaviour. Anything discontinuous (threshold reset, spike emission) belongs in a callback.

### Spike Kernels

`kernels.jl` defines `SpikeKernel`: `DeltaKernel`, `GaussianKernel`, `RaisedCosineKernel`,
`AlphaKernel`, and `FunctionKernel` for anything custom. Each is callable on a time offset
`dt = t - t_spike` and declares its truncation radius via `support(k)`, which is what
`spike_current` gates on — there is no kernel-independent window parameter.

Outer constructors normalise every kernel to **unit area**, so widening a kernel smears the
same total charge over more time rather than also increasing total drive. Use
`SpikingArgs.spk_scale` for strength.

`spike_current(train, t, spk_args)` convolves a `SpikeTrain` with the resolved kernel;
overlapping contributions on a channel accumulate. Sample stochastic sources into a
`SpikeTrain` **once** (see `poisson_spike_train`) — drawing random numbers inside an ODE
right-hand side gives the solver a different vector field at every stage and destroys
reproducibility.

A kernel narrower than the solver's `dt` can be stepped straight over; keep delta-like
kernels a few steps wide.

### Demonstration Model

`IntegrateAndFire` (in `demo_iaf.jl`): real domain, leakage `λ`, Erdos-Renyi random `W`.
Threshold reset and spike emission are handled by `reset_callback`, which returns a
`VectorContinuousCallback` plus a `SpikeRecorder`. Crossings are root-found, so spike times
are exact rather than quantised to save points, and `SpikeTrain(recorder, shape)` turns the
recorded events into a train.

`find_spikes_ref` is the reference detector (local maxima above threshold) for models with
**no** reset. With a reset active every peak is clipped exactly at threshold, so a strict
`> threshold` test cannot fire — the callback is the spike source there, not the detector.

## Directory Structure

```
SpikingNetworks.jl/
├── src/
│   ├── SpikingNetworks.jl   # module, exports
│   ├── kernels.jl           # SpikeKernel hierarchy, support()
│   ├── models.jl            # AbstractNeuronModel, SpikingArgs, simulate_network
│   ├── spiketrain_types.jl  # SpikeTrain, SpikingCall, CurrentCall
│   ├── spiking.jl           # neuron_bank, spike_current, find_spikes_ref
│   └── demo_iaf.jl          # IntegrateAndFire + reset callback + Erdos-Renyi demo
├── scripts/
│   └── demo_iaf.jl          # executable demo script
├── notebooks/
│   └── demo_iaf_interactive.ipynb  # interactive Jupyter demo
├── test/
│   └── runtests.jl          # entry point; one file per area
└── AGENTS.md
```

## Main Components

| File | Responsibility |
|---|---|
| `kernels.jl` | `SpikeKernel` types, unit-area constructors, `support` |
| `spiketrain_types.jl` | `SpikeTrain`, `SpikeTrainGPU`, `SpikingCall`, `CurrentCall` |
| `spiking.jl` | `neuron_bank`, `resolve_kernel`, `spike_current`, `find_spikes_ref` |
| `models.jl` | `AbstractNeuronModel`, `SpikingArgs`, protocol, `simulate_network` |
| `demo_iaf.jl` | `IntegrateAndFire`, `reset_callback`, `SpikeRecorder`, `poisson_spike_train` |

Include order matters: `kernels.jl` must precede `models.jl`, because the
`SpikingArgs.spike_kernel` field type references `SpikeKernel`.

## Key Entry Points

- Module: `src/SpikingNetworks.jl`
- Interactive: `notebooks/demo_iaf_interactive.ipynb`
- Script: `scripts/demo_iaf.jl`

## Development Workflow

```bash
julia --project=.                       # REPL in this package's environment
julia --project=. scripts/demo_iaf.jl   # run the demo
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run the interactive demo with Jupyter (`jupyter notebook notebooks/`). Note the repo has no
`nbconvert` available, so notebook cells are validated by extracting them to a script and
running headlessly (`GKSwstype=100`) rather than by executing the notebook in place.

## Design / Code Style

- 4-space indentation, snake_case functions, PascalCase types.
- `Float32` for GPU/ODE compatibility.
- No formal linter; follow existing file patterns.
- All public exports listed in `src/SpikingNetworks.jl`.
- User dynamics are fully arbitrary; no hidden Lux or phase-SSM assumptions.
