"""
Demonstration: generic integrate-and-fire neuron in real domain.
Erdos-Renyi random connections, leakage dynamics, threshold reset applied through
a solver callback (see `reset_callback`) rather than inside the update equation.
"""

using Random

struct IntegrateAndFire <: AbstractNeuronModel
    dims::Int
    lambda::Float32
    W::Matrix{Float32}
    rest::Float32
    filter::Matrix{Float32}
    threshold::Float32
end

function IntegrateAndFire(n_neurons::Int, p::Float32=0.1f0; threshold::Float32=0.15f0, seed::AbstractRNG=Random.default_rng())
    W = Float32.(rand(seed, n_neurons, n_neurons) .< p) .* randn(seed, Float32, n_neurons, n_neurons)
    rest = 0.0f0
    filter = Matrix{Float32}(I, n_neurons, n_neurons)
    model = IntegrateAndFire(n_neurons, -0.2f0, W, rest, filter, threshold)
    return model
end

# Real-domain update: dz/dt = lambda*z + W*(filter*I(t))  (leakage + arbitrary connection drive).
# Pure function of (z, t): the reset lives in `reset_callback`, never in here — writing to `z`
# would corrupt the solver's Runge-Kutta stage vectors.
function update_equation!(dz, z, I, t, params, model::IntegrateAndFire; connections=nothing)
    W_use = connections !== nothing ? connections : model.W
    # `filter` is a spatial mixing matrix over channels, not a temporal spike kernel
    I_filt = model.filter * I
    dz .= params.lambda .* z .+ W_use * I_filt
    return nothing
end

"""
Accumulates (neuron, time) pairs emitted by `reset_callback` during a solve.
"""
mutable struct SpikeRecorder
    indices::Vector{CartesianIndex{1}}
    times::Vector{Float32}
end

SpikeRecorder() = SpikeRecorder(CartesianIndex{1}[], Float32[])

Base.length(rec::SpikeRecorder) = length(rec.times)

function Base.show(io::IO, rec::SpikeRecorder)
    print(io, "SpikeRecorder: ", length(rec.times), " events.")
end

function SpikeTrain(rec::SpikeRecorder, shape::Tuple)
    return SpikeTrain(rec.indices, rec.times, shape, 0.0f0)
end

"""
    reset_callback(model) -> (callback, recorder)

Root-find upward crossings of `model.threshold` on every neuron. On a crossing the
potential is driven back to `model.rest` through `integrator.u` and the event is
recorded, so spike times are exact rather than quantised to save points.
"""
function reset_callback(model::IntegrateAndFire)
    rec = SpikeRecorder()
    condition(out, u, t, integrator) = (out .= u .- model.threshold)

    function fire!(integrator, i::Integer)
        push!(rec.indices, CartesianIndex(i))
        push!(rec.times, Float32(integrator.t))
        integrator.u[i] = model.rest
    end

    # VectorContinuousCallback has no affect_neg!. Recent DiffEqBase hands `event` a
    # length-`len` Int8 mask (+1 upcrossing, -1 downcrossing, 0 quiet); older versions
    # pass a scalar index. Handle both, and fire on upcrossings only.
    function affect!(integrator, event)
        if event isa Integer
            fire!(integrator, event)
        else
            for i in eachindex(event)
                event[i] > 0 && fire!(integrator, i)
            end
        end
        u_modified!(integrator, true)
        return nothing
    end

    cb = VectorContinuousCallback(condition, affect!, model.dims;
                                  save_positions=(true, true))
    return cb, rec
end

"""
Poisson-style spike source. Each call draws one Bernoulli sample per neuron.
Use it to *build* a SpikeTrain once — do not call it from inside an ODE right-hand
side, where the solver would see a different vector field at every stage.
"""
mutable struct CurrentGenerator
    rate::Float32
    dt::Float32
    n_neurons::Int
    rng::AbstractRNG
end

function (cg::CurrentGenerator)(t; p=nothing)
    p_use = p === nothing ? cg.rate * cg.dt : p
    Float32.(rand(cg.rng, cg.n_neurons) .< p_use) .+ 0.0f0
end

"""
    poisson_spike_train(cg, tspan) -> SpikeTrain

Sample `cg` on a `cg.dt` grid across `tspan` and collect the draws into a
deterministic SpikeTrain that can be convolved with any `SpikeKernel`.
"""
function poisson_spike_train(cg::CurrentGenerator, tspan::Tuple{<:Real,<:Real})
    indices = CartesianIndex{1}[]
    times = Float32[]
    for t in Float32(tspan[1]):cg.dt:Float32(tspan[2])
        draw = cg(t)
        for i in eachindex(draw)
            if draw[i] > 0.0f0
                push!(indices, CartesianIndex(i))
                push!(times, t)
            end
        end
    end
    return SpikeTrain(indices, times, (cg.n_neurons,), 0.0f0)
end

# Call the spatial filter on potentials (post-simulation or during dynamics)
function (model::IntegrateAndFire)(z)
    return model.filter * z
end

function detect_spike(z, t, threshold, model::IntegrateAndFire)
    # Simple threshold crossing: detect which potentials currently sit above threshold
    indices = findall(z .> threshold)
    times = fill(Float32(t), length(indices))
    cart = [CartesianIndex(i) for i in indices]
    return SpikeTrain(cart, times, (length(z),), 0.0f0)
end
