module SpikingNetworks

using DifferentialEquations
using LinearAlgebra
using Random

export SpikeTrain, SpikeTrainGPU, SpikingArgs,
       SpikingCall, CurrentCall, LocalCurrent,
       oscillator_bank, spike_current,
       detect_spike_protocol, NeuronModel,
       phase_to_train, solution_to_train,
       find_spikes_ref, AbstractNeuronModel,
       GenericNeuronModel, simulate_network,
       update_equation!, detect_spike,
       IntegrateAndFire

include("types.jl")
include("spiking.jl")
include("models.jl")
include("demo_iaf.jl")

end
