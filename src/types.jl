"""
Core data structures for SpikingNetworks — stripped from PhasorNetworks.
No Lux, no VSA, no phase-locked SSM assumptions.
"""

struct SpikeTrain{N}
    indices::Vector{CartesianIndex{N}}
    times::Vector{Float32}
    shape::Tuple
    offset::Float32
end

Base.size(st::SpikeTrain) = st.shape
Base.length(st::SpikeTrain) = length(st.times)

struct SpikeTrainGPU{N}
    indices::AbstractArray  # GPU array of CartesianIndex
    times::AbstractArray{Float32}
    shape::Tuple
    linear_shape::Int
    offset::Float32
end

Base.size(st::SpikeTrainGPU) = st.shape

struct SpikingArgs
    leakage::Float32
    t_period::Float32
    t_window::Float32
    spk_scale::Float32
    threshold::Float32
    spike_kernel::Union{Symbol, Function}
    solver
    solver_args::Dict
    warmup_periods::Int
end

function SpikingArgs(; leakage=-0.2f0, t_period=1.0f0, t_window=0.01f0,
                      spk_scale=1.0f0, threshold=0.001f0, spike_kernel=:gaussian,
                      solver=Tsit5(), solver_args=Dict(:dt=>0.01f0, :adaptive=>false),
                      warmup_periods=0)
    SpikingArgs(Float32(leakage), Float32(t_period), Float32(t_window),
                Float32(spk_scale), Float32(threshold), spike_kernel,
                solver, solver_args, Int(warmup_periods))
end

struct SpikingCall
    train::Union{SpikeTrain, SpikeTrainGPU}
    spk_args::SpikingArgs
    t_span::Tuple{Float32, Float32}
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Float32
end

function LocalCurrent(st::Union{SpikeTrain, SpikeTrainGPU}, spk_args::SpikingArgs)
    LocalCurrent(t -> spike_current(st, t, spk_args), st.shape, st.offset)
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{Float32, Float32}
end

function CurrentCall(sc::SpikingCall)
    CurrentCall(LocalCurrent(sc.train, sc.spk_args), sc.spk_args, sc.t_span)
end
