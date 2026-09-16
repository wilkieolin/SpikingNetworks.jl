# Custom Neuron Models

SpikingNetworks.jl uses a **protocol-based design**: you define a concrete model by extending `AbstractNeuronModel` and implementing two functions. No inheritance, no macros — just plain Julia multiple dispatch.

## The Protocol

Every neuron model must provide:

```julia
# 1. Dynamics: dz/dt = f(z, I, t, params; connections=W)
function update_equation!(dz, z, I, t, params, model::YourModel; connections=nothing)
    # dz    — output: derivative vector (MUST write to this, not z)
    # z     — current state (potentials); DO NOT MUTATE
    # I     — input current at time t
    # t     — current time
    # params — model parameters (NamedTuple or whatever you stored)
    # connections — optional N×N matrix W for recurrent coupling
end

# 2. Spike detection: returns a SpikeTrain
function detect_spike(z, t, threshold, model::YourModel)
    # z        — state vector at time t
    # t        — current time
    # threshold — spike threshold (from SpikingArgs or your model)
    # returns  — SpikeTrain with spike times/indices
end
```

## Minimal Example: Leaky Integrate-and-Fire

```julia
using SpikingNetworks
using LinearAlgebra

struct MyLIF <: AbstractNeuronModel
    N::Int
    lambda::Float32      # leakage (negative = leak)
    threshold::Float32
    W::Matrix{Float32}   # recurrent connections
end

# Dynamics: dz/dt = λ*z + W*I
function update_equation!(dz, z, I, t, params, model::MyLIF; connections=nothing)
    W = connections !== nothing ? connections : model.W
    dz .= model.lambda .* z .+ W * I
end

# Detect spikes: simple threshold crossing
function detect_spike(z, t, threshold, model::MyLIF)
    spikes = findall(>(threshold), z)
    return SpikeTrain(spikes, fill(t, length(spikes)), length(z))
end
```

## Using the Model

```julia
model = MyLIF(20, -0.2f0, 0.8f0, rand(Float32, 20, 20))
z0 = zeros(Float32, 20)
I_fn(t) = 0.1f0 .* randn(Float32, 20)

sol = simulate_network(model, z0, I_fn, (0.0f0, 5.0f0))
```

## Key Rules

| Rule | Reason |
|------|--------|
| **Never mutate `z`** | `z` is the solver's state or an internal stage vector; mutating it is undefined behavior |
| **Write to `dz` only** | `dz` is the derivative output; the solver owns `z` |
| **Discontinuities in callbacks** | Threshold reset, spike emission → use `ContinuousCallback` (see `reset_callback` in `demo_iaf.jl`) |
| **`connections` is optional** | Pass `W` at call site: `simulate_network(model, z0, I_fn, tspan; connections=W)` |

## Advanced: Spike-Driven Input

For discrete spike inputs, implement a `SpikingCall` kernel or use the built-in `SpikingArgs`:

```julia
args = SpikingArgs(spike_kernel=GaussianKernel(0.1f0), spk_scale=0.5f0)
I_fn(t) = spike_current(input_train, t, args)
```

The `SpikingArgs` struct holds:
- `leakage`, `t_period`, `t_window` — solver hints
- `spk_scale` — global kernel amplitude
- `steepness`, `threshold` — detection parameters
- `spike_kernel` — `DeltaKernel`, `GaussianKernel`, `RaisedCosineKernel`, `AlphaKernel`, or `FunctionKernel`
- `solver`, `solver_args` — passed to `DifferentialEquations.solve`
- `warmup_periods` — pre-simulation steps

## Built-in Reference: `IntegrateAndFire`

See `src/demo_iaf.jl` for a complete implementation:
- `IntegrateAndFire` struct with `lambda`, `threshold`, `rest`, `W`
- `reset_callback` — `VectorContinuousCallback` + `SpikeRecorder` for exact spike times
- `CurrentGenerator` + `poisson_spike_train` — stochastic input generation

## Testing Your Model

Add tests following the pattern in `test/model_tests.jl`:
```julia
@testset "MyLIF" begin
    model = MyLIF(10, -0.2f0, 1.0f0, zeros(10,10))
    z = rand(Float32, 10)
    dz = similar(z)
    update_equation!(dz, z, zeros(10), 0.0f0, (), model)
    @test dz ≈ -0.2f0 .* z
end
```