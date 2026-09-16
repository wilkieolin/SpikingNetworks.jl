using LinearAlgebra

struct SpikeTrain{N}
    indices::Vector{CartesianIndex{N}}
    times::Vector{Float32}
    shape::Tuple
    offset::Float32
end

function SpikeTrain(indices::AbstractArray, times::AbstractArray, shape::Tuple, offset::Real)
    times_f32 = eltype(times) == Float32 ? Float32.(times) : Float32.(times)
    return SpikeTrain{length(shape)}(convert(Vector{CartesianIndex{length(shape)}}, indices), times_f32, shape, Float32(offset))
end

struct SpikeTrainGPU{N}
    indices::AbstractArray
    linear_indices::AbstractArray
    times::AbstractArray{Float32}
    shape::Tuple
    linear_shape::Int
    offset::Float32
end

function SpikeTrainGPU(indices::AbstractArray, times::AbstractArray, shape::Tuple, offset::Real)
    times_f32 = eltype(times) == Float32 ? Float32.(times) : Float32.(times)
    linear_indices = LinearIndices(shape)[indices]
    return SpikeTrainGPU{length(shape)}(indices, linear_indices, times_f32, shape, prod(shape), Float32(offset))
end

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

function Base.size(x::SpikeTrain)
    return x.shape
end
Base.length(st::SpikeTrain) = length(st.times)

function random_spike_train(shape::Tuple; n_spikes::Int=10, t_max::Float32=1.0f0, offset::Float32=0.0f0)
    indices = [CartesianIndex(ntuple(i -> rand(1:shape[i]), length(shape))...) for _ in 1:n_spikes]
    times = sort(Float32.(rand(Float32, n_spikes) .* t_max))
    return SpikeTrain(indices, times, shape, offset)
end

struct SpikingCall
    train::Union{SpikeTrain, SpikeTrainGPU}
    t_span::Tuple{Float32, Float32}
end

function Base.size(x::SpikingCall)
    return x.train.shape
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Float32
end

function Base.size(x::LocalCurrent)
    return x.shape
end

struct CurrentCall
    current::LocalCurrent
    t_span::Tuple{Float32, Float32}
end

function Base.size(x::CurrentCall)
    return x.current.shape
end
