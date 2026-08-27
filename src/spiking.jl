"""
Generic spiking dynamics — no R&F bridge, no phase-SSM link.
User provides dzdt; we solve and detect.
"""

function neuron_bank(u0, dzdt; tspan=(0.0f0, 10.0f0), spk_args=SpikingArgs(), callback=nothing)
    prob = ODEProblem(dzdt, u0, tspan)
    kwargs = callback === nothing ? spk_args.solver_args : merge(spk_args.solver_args, Dict(:callback=>callback))
    sol = solve(prob, spk_args.solver; kwargs...)
    return sol
end

function neuron_bank(u0, dzdt, params; tspan=(0.0f0, 10.0f0), spk_args=SpikingArgs(), callback=nothing)
    prob = ODEProblem(dzdt, u0, tspan, params)
    kwargs = callback === nothing ? spk_args.solver_args : merge(spk_args.solver_args, Dict(:callback=>callback))
    sol = solve(prob, spk_args.solver, p=params; kwargs...)
    return sol
end

"""
    resolve_kernel(spk_args::SpikingArgs) -> SpikeKernel

Map the `spike_kernel` field onto a concrete kernel. Symbols are interpreted
relative to `spk_args.t_window`; a bare `Function` is wrapped untruncated.
"""
function resolve_kernel(spk_args::SpikingArgs)
    k = spk_args.spike_kernel
    k isa SpikeKernel && return k
    k isa Function && return FunctionKernel(k)
    if k === :gaussian
        return GaussianKernel(spk_args.t_window)
    elseif k === :raised_cosine
        return RaisedCosineKernel(2.0f0 * spk_args.t_window)
    elseif k === :delta
        return DeltaKernel(spk_args.t_window)
    elseif k === :alpha
        return AlphaKernel(spk_args.t_window)
    end
    error("Unknown spike kernel $(repr(k)). Expected a SpikeKernel, a Function, " *
          "or one of :gaussian, :raised_cosine, :delta, :alpha.")
end

"""
    spike_current(train::SpikeTrain, t, spk_args::SpikingArgs) -> Array{Float32}

Current injected at time `t` by convolving `train` with the kernel named in
`spk_args`. Overlapping contributions on the same channel accumulate, so wide
kernels sum correctly.
"""
function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs)
    kernel = resolve_kernel(spk_args)
    return spike_current(train, t, kernel; scale=spk_args.spk_scale)
end

function spike_current(train::SpikeTrain, t::Real, kernel::SpikeKernel; scale::Real=1.0f0)
    current = zeros(Float32, train.shape)
    radius = support(kernel)
    s = Float32(scale)
    tf = Float32(t)
    @inbounds for n in eachindex(train.times)
        dt = tf - train.times[n]
        abs(dt) <= radius || continue
        current[train.indices[n]] += s * kernel(dt)
    end
    return current
end

"""
    find_spikes_ref(u, t, threshold; dim=ndims(u)) -> (channels, times)

Reference spike detector: local maxima of `u` along `dim` whose peak value
exceeds `threshold`. A local maximum is a downward sign change of the discrete
derivative, so this needs two `diff`s — one for the derivative, one for its
sign change.
"""
function find_spikes_ref(u, t, threshold; dim=-1)
    if dim == -1
        dim = ndims(u)
    end
    d = diff(u, dims=dim)              # size N-1 along dim
    dd = diff(sign.(d), dims=dim)      # size N-2 along dim; < 0 marks a local maximum
    found = findall(<(0.0f0), dd)
    # index j along `dim` in dd corresponds to the peak at index j+1 in u
    peaks = [shift_index(ci, dim, 1) for ci in found]
    selected = [ci for ci in peaks if u[ci] > threshold]
    times = Float32[t[ci[dim]] for ci in selected]
    # Build spatial-only CartesianIndex objects
    spatial_dims = filter(i -> i != dim, 1:ndims(u))
    channels = [CartesianIndex(ntuple(k -> ci[spatial_dims[k]], length(spatial_dims)))
                for ci in selected]
    return channels, times
end

# Offset one component of a CartesianIndex, leaving the rest untouched.
function shift_index(ci::CartesianIndex{N}, dim::Int, by::Int) where {N}
    return CartesianIndex(ntuple(i -> i == dim ? ci[i] + by : ci[i], N))
end
