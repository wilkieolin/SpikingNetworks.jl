# Superconducting Transponder Neuron Models

`SpikingNetworks.jl` provides three superconducting transponder neuron models, each implementing the `AbstractNeuronModel` interface with distinct physics and use cases:

| Model | Physics | Best For |
|-------|---------|----------|
| `SimplifiedTransponder` | Integrate-and-Fire with variable-resistor nanowire | Large networks, fast simulation |
| `FullPhysicsTransponder` | Adaptive/Resonate-and-Fire (nTron hotspot dynamics) | Accurate physics, small networks |
| `ResonateAndFireTransponder` | Explicit R&F (LC tank oscillator) | Resonant dynamics, frequency-domain analysis |

All three support:
- Network coupling via weight matrix `W` and delay matrix `D`
- Custom spike detection via `detect_spike`
- Integration with `simulate_network` and `SpikeRecorder`
- Unit-area spike kernels through `SpikingArgs`

---

## SimplifiedTransponder

### Overview
An Integrate-and-Fire model where the nanowire resistance switches dynamically during a refractory period. Each neuron has 5 state variables:
- `i_nw` — nanowire current
- `i_loop` — loop current (spike detection variable)
- `ref_timer` — refractory timer
- `output_firing` — output pulse flag
- `output_pulse_start` — pulse start time

### Parameters (`SimplifiedParams`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ib1`, `Ib2` | 10µA, 20µA | Bias currents |
| `Ls1`, `Rs1` | 5nH, 3Ω | Source inductance/resistance |
| `LT`, `RT` | 140nH, 4Ω | Tank inductance/resistance |
| `Lnw`, `Rnw_max` | 10nH, 1kΩ | Nanowire inductance/max resistance |
| `Ithresh1`, `Ithresh2` | 1µA, 2µA | Detection thresholds |
| `ref_period` | 5ns | Refractory period |
| `t_rise`, `t_fall`, `t_width` | 1ns each | Nanowire switching times |
| `output_tau`, `output_x0`, `output_mu0` | 1ns, 1ns, 0.2ns | Output pulse shape |

### Construction
```julia
using SpikingNetworks

# Single neuron with defaults
model = SimplifiedTransponder(1)

# Network of 100 neurons with custom coupling
n = 100
W = rand(Float32, n, n) .* 0.01f0
D = rand(Float32, n, n) .* 1e-9
model = SimplifiedTransponder(n; W=W, D=D)

# Custom parameters per neuron
params = [SimplifiedParams(Ithresh2=Float32(3.0e-6)) for _ in 1:n]
model = SimplifiedTransponder(n; params=params, W=W, D=D)
```

### Simulation
```julia
using SpikingNetworks
using DifferentialEquations

# External current (n neurons × timesteps)
tspan = (0.0f0, 100.0f0)
I_ext = zeros(Float32, model.n, 1000)
I_ext[1, 200:300] .= 5.0f-6  # 5µA pulse to neuron 1

# Spiking arguments with alpha kernel
spk_args = SpikingArgs(;
    spike_kernel = AlphaKernel(Float32(2.0e-9)),
    spk_scale = Float32(1.0),
    spk_thr = Float32(1.5e-6),
    dt = Float32(0.1e-9)
)

sol = simulate_network(model, tspan, I_ext; spiking_args=spk_args, dt=Float32(0.1e-9))
```

### Spike Detection
Spikes are detected when `i_loop > threshold` (default uses `Ithresh2`). The `detect_spike` function returns a `SpikeTrain` compatible with `spike_current`.

---

## FullPhysicsTransponder

### Overview
A biophysically detailed nTron (nanowire transistor) model with hotspot dynamics. Based on the physics of superconducting nanowires with thermal hotspot formation. Each nTron has 5 internal state variables tracking hotspot resistance evolution.

### Network Topology
The default `build_snn2()` constructs a fixed 6-nTron circuit with:
- 4 input nTrons (U1, U3, U4, U6) with `SQ_G_IN` gate squares
- 2 output nTrons (U2, U5) with `SQ_G_OUT` gate squares
- Coupling inductors, bias resistors, and current sources
- Probe branches (default: `cpA` — loop current of U2)

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bv` | 1.5 | Bias voltage scale |
| `dt_ns` | 0.0 | Inter-pulse delay (ns) |
| `tstop` | 30ns | Simulation stop time |
| `h` | 1ps | Fixed time step |
| `probe` | `("cpA",)` | Probe branch names |

### Construction
```julia
# Default 6-nTron network
model = FullPhysicsTransponder()

# Custom probe branches
model = FullPhysicsTransponder(probe=("cpA", "cpB"))

# Adjust time step
model = FullPhysicsTransponder(h=Float32(0.5e-12))
```

### Simulation
```julia
tspan = (0.0f0, model.tstop)
I_ext = zeros(Float32, length(model), 1)  # No external current by default

sol = simulate_network(model, tspan, I_ext; dt=model.h)
```

### Probe Currents
Access internal branch currents during/after simulation:
```julia
# During simulation via callback
function probe_callback(integrator)
    z = integrator.u
    currents = get_probe_currents(model, z)
    # Log currents...
end

# After simulation
for (i, u) in enumerate(sol.u)
    currents = get_probe_currents(model, u)
    println("t=$(sol.t[i]): ", currents)
end
```

### Spike Detection
Spikes detected on probe branches `cpA` (U2) and `cpB` (U5) when loop current exceeds threshold.

---

## ResonateAndFireTransponder

### Overview
An explicit Resonate-and-Fire model with an LC tank circuit. Each neuron has 5 state variables:
- `i_tank` — tank current (oscillates at resonance)
- `q_cap` — capacitor charge
- `ref_timer` — refractory timer
- `output_firing` — output pulse flag
- `output_pulse_start` — pulse start time

### Parameters (`RnFParams`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `L_tank` | 10nH | Tank inductance |
| `C_tank` | 0.5pF | Tank capacitance |
| `R_damp` | 10Ω | Damping resistance |
| `I_thresh` | 2µA | Spike threshold (|i_tank|) |
| `ref_period` | 5ns | Refractory period |
| `t_rise`, `t_fall`, `t_width` | 0.5ns, 0.5ns, 1ns | Output pulse timing |
| `output_tau`, `output_x0`, `output_mu0` | 1ns, 1ns, 0.2ns | Output pulse shape |

### Derived Properties
```julia
p = RnFParams()
f_res = resonance_frequency(p)  # Hz
Q = quality_factor(p)           # Quality factor
```

### Construction
```julia
# Single neuron
model = ResonateAndFireTransponder(1)

# Network with resonant coupling
n = 50
W = rand(Float32, n, n) .* 0.005f0
D = rand(Float32, n, n) .* 0.5e-9
model = ResonateAndFireTransponder(n; W=W, D=D)

# Custom resonance frequency (e.g., 5 GHz)
params = [RnFParams(L_tank=Float32(5.0e-9), C_tank=Float32(0.2e-12)) for _ in 1:n]
model = ResonateAndFireTransponder(n; params=params, W=W, D=D)
```

### Simulation
```julia
tspan = (0.0f0, 200.0f0)
I_ext = zeros(Float32, model.n, 2000)

# Inject at resonance frequency
f_res = resonance_frequency(model.params[1])
for i in 1:model.n
    t = range(0, 200e-9, length=2000)
    I_ext[i, :] .= 1.0e-6 .* sin.(2π * f_res * t)
end

spk_args = SpikingArgs(;
    spike_kernel = AlphaKernel(Float32(1.0e-9)),
    spk_scale = Float32(1.0),
    spk_thr = Float32(2.0e-6),
    dt = Float32(0.1e-9)
)

sol = simulate_network(model, tspan, I_ext; spiking_args=spk_args, dt=Float32(0.1e-9))
```

---

## Common Patterns

### Recording Spikes
```julia
using SpikingNetworks

recorder = SpikeRecorder()
cb = reset_callback(model, spk_args; recorder=recorder)

sol = simulate_network(model, tspan, I_ext; 
    spiking_args=spk_args, 
    callback=cb,
    dt=Float32(0.1e-9)
)

spike_train = SpikeTrain(recorder, (model.n,))
```

### Poisson Input
```julia
rates = fill(10e6, model.n)  # 10 MHz per neuron
train = poisson_spike_train(rates, tspan[2]; dt=Float32(0.1e-9))
```

### Custom Kernels
```julia
# Gaussian kernel
spk_args = SpikingArgs(spike_kernel=GaussianKernel(Float32(2.0e-9)), ...)

# Raised cosine
spk_args = SpikingArgs(spike_kernel=RaisedCosineKernel(Float32(5.0e-9)), ...)

# Custom function
my_kernel = FunctionKernel(dt -> exp(-dt/1e-9), Float32(10.0e-9))
spk_args = SpikingArgs(spike_kernel=my_kernel, ...)
```

---

## Model Comparison

| Feature | Simplified | FullPhysics | ResonateAndFire |
|---------|------------|-------------|-----------------|
| State variables/neuron | 5 | ~30 (circuit-wide) | 5 |
| Physics fidelity | Phenomenological | Biophysical (hotspot) | Circuit-level (LC) |
| Network size | 100s–1000s | ~6 (fixed topology) | 10s–100s |
| Time step | Adaptive (Tsit5) | Fixed (1ps) | Adaptive (Tsit5) |
| Coupling | W, D matrices | Fixed netlist | W, D matrices |
| Spike output | Pulse + refractory | Probe current | Pulse + refractory |
| Frequency tuning | No | Via bias | Via L/C |

---

## Choosing a Model

- **SimplifiedTransponder**: Large-scale spiking networks, rate-coded inputs, ML-style tasks
- **FullPhysicsTransponder**: Device-level accuracy, nTron circuit design, thermal dynamics
- **ResonateAndFireTransponder**: Oscillatory networks, frequency-domain processing, resonance-based computing