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

## Environment Setup (Linux)

### 1. Install Julia

Use **[juliaup](https://github.com/JuliaLang/juliaup)** (recommended, manages versions):

```bash
curl -fsSL https://install.julialang.org | sh
# restart shell, then:
juliaup add 1.11   # Project.toml compat; 1.12+ also works
juliaup default 1.11
```

Verify: `julia --version` → `1.11.x` (or `1.12.x`)

### 2. Bootstrap the Project Environment

```bash
cd SpikingNetworks.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Installs `DifferentialEquations`, `Plots`, `Random`, `Test` per `Project.toml`.

### 3. Jupyter / IJulia Kernel (for VS Code notebooks)

```bash
julia --project=. -e 'using Pkg; Pkg.add("IJulia")'
julia --project=. -e 'using IJulia; installkernel("SpikingNetworks", env=Dict("JULIA_PROJECT"=>pwd()))'
```

- Registers a kernel named **"SpikingNetworks"** tied to this project's environment.
- In VS Code: open `notebooks/demo_iaf_interactive.ipynb` → Select Kernel → **SpikingNetworks**.

> **Plots backend**: VS Code with display works out of the box with GR (default). For interactive plots, `Pkg.add("PlotlyJS")` and `using PlotlyJS; plotlyjs()` in the notebook.

### 4. VS Code Setup

1. Install **Julia** extension (`julialang.language-julia`)
2. Open folder `SpikingNetworks.jl`
3. Command Palette → **Julia: Select Environment** → Choose this folder (auto-detects `Project.toml`)
4. Open `.ipynb` files directly — they run via the registered kernel.

### 5. Run the Demos

| Method | Command |
|--------|---------|
| Script (headless) | `julia --project=. scripts/demo_iaf.jl` |
| Notebook (interactive) | `jupyter notebook notebooks/` → open `demo_iaf_interactive.ipynb` |
| VS Code | Open notebook → Run All (kernel: **SpikingNetworks**) |

### 6. Run Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

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
