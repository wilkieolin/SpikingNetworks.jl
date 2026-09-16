# Callbacks & Recording

SpikingNetworks.jl uses `DifferentialEquations.jl` callbacks for exact spike detection and recording. This avoids quantization errors from fixed save points.

## SpikeRecorder

Records spike times and neuron indices during simulation.

```julia
using SpikingNetworks

rec = SpikeRecorder()
# After simulation:
train = SpikeTrain(rec, (n_neurons,))  # Convert to SpikeTrain
```

## reset_callback (IntegrateAndFire)

Built-in callback for `IntegrateAndFire` that:
1. Detects threshold crossing via root-finding (exact spike times)
2. Resets voltage to `rest` value
3. Records spike in `SpikeRecorder`

```julia
model = IntegrateAndFire(20, 0.15f0; threshold=0.15f0, rest=0.0f0)
cb, rec = reset_callback(model)

sol = simulate_network(model, z0, I_fn, tspan; callback=cb)
spikes = SpikeTrain(rec, (20,))
```

The callback returns a `VectorContinuousCallback` — see `DifferentialEquations.jl` docs for chaining multiple callbacks.

## CurrentGenerator & poisson_spike_train

Generate stochastic input currents once (outside the ODE) for reproducibility:

```julia
# Poisson spike generator: rate=5Hz, dt=1ms, 20 neurons
gen = CurrentGenerator(5.0f0, 0.001f0, 20, Xoshiro(7))
train = poisson_spike_train(gen, (0.0f0, 5.0f0))

# Convolve with kernel during simulation
args = SpikingArgs(spike_kernel=GaussianKernel(0.1f0), spk_scale=0.5f0)
I_fn(t) = spike_current(train, t, args)
```

**Why sample once?** Drawing random numbers inside the ODE right-hand side gives the solver a different vector field at every stage, destroying reproducibility and breaking root-finding.

## Custom Callbacks

Any `DifferentialEquations.jl` callback works. Pass via `simulate_network`:

```julia
using DifferentialEquations

cb = DiscreteCallback(condition, affect!)
sol = simulate_network(model, z0, I_fn, tspan; callback=cb)
```

For multiple callbacks, use `CallbackSet(cb1, cb2, ...)`.

## SpikingArgs Callback Options

`SpikingArgs` accepts `solver_args` passed to `solve`:

```julia
args = SpikingArgs(
    solver=Tsit5(),
    solver_args=Dict(:dt => 0.01f0, :adaptive => false, :saveat => 0.01f0)
)
```

Common options:
- `dt` — fixed step (if `adaptive=false`)
- `adaptive` — `true`/`false`
- `saveat` — save interval
- `reltol`, `abstol` — tolerances
- `maxiters` — max iterations

See `DifferentialEquations.jl` [solve options](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/) for full list.

## Example: Custom Recording Callback

```julia
using DifferentialEquations

# Record membrane potential every 1ms
saved_u = Float32[]
saved_t = Float32[]

condition(u, t, integrator) = true
affect!(integrator) = begin
    push!(saved_u, copy(integrator.u))
    push!(saved_t, integrator.t)
end

cb = DiscreteCallback(condition, affect!; save_positions=(false,false))
sol = simulate_network(model, z0, I_fn, tspan; callback=cb)

# saved_u now has trajectory at every solver step
```