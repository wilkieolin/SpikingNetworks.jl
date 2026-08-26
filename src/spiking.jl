"""
Generic spiking dynamics — no R&F bridge, no phase-SSM link.
User provides dzdt; we solve and detect.
"""

function oscillator_bank(u0, dzdt; tspan=(0.0f0, 10.0f0), spk_args=SpikingArgs())
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    return sol
end

function oscillator_bank(u0, dzdt, params; tspan=(0.0f0, 10.0f0), spk_args=SpikingArgs())
    prob = ODEProblem(dzdt, u0, tspan, params)
    sol = solve(prob, spk_args.solver, p=params; spk_args.solver_args...)
    return sol
end

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs; sigma=9.0)
    current = zeros(Float32, train.shape)
    scale = spk_args.spk_scale
    t = Float32(t)
    times = train.times
    active = (times .> (t - sigma * spk_args.t_window)) .* (times .< (t + sigma * spk_args.t_window))
    active_inds = train.indices[active]
    active_tms = train.times[active]
    impulses = if spk_args.spike_kernel == :gaussian
        raised_cosine_kernel(active_tms, t, spk_args.t_window)
    elseif spk_args.spike_kernel isa Function
        spk_args.spike_kernel(active_tms, t)
    else
        error("Unknown spike kernel")
    end
    current[active_inds] .+= (scale .* impulses)
    return current
end

function raised_cosine_kernel(x::AbstractArray, t::Real, t_sigma::Real)
    dt = t .- x
    hw = 2.0f0 .* t_sigma
    abs_dt = abs.(dt)
    return ifelse.(abs_dt .<= hw,
                   0.5f0 .* (1.0f0 .+ cos.(pi_f32 .* dt ./ hw)),
                   0.0f0)
end

function find_spikes_ref(u, t, threshold; dim=-1)
    # Generic reference spike detector: peaks above threshold along time axis
    if dim == -1
        dim = ndims(u)
    end
    op = x -> x .< 0.0f0
    maxima = findall(op.(diff.(sign.(u), dims=dim)))
    peak_vals = u[maxima]
    above = peak_vals .> threshold
    spikes = maxima[above]
    ax = collect(1:ndims(u))
    spatial_ax = filter(i -> i != dim, ax)
    spatial_idx = [getindex.(spikes, i) for i in spatial_ax]
    channels = CartesianIndex.(spatial_idx...)
    times = t[getindex.(spikes, dim)]
    return channels, times
end
