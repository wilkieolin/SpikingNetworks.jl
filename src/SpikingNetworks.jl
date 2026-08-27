module SpikingNetworks

using DifferentialEquations
using LinearAlgebra
using Random

export SpikeTrain, SpikeTrainGPU,
       SpikingCall, CurrentCall, LocalCurrent,
       random_spike_train,
       SpikingArgs, neuron_bank, spike_current,
       SpikeKernel, DeltaKernel, GaussianKernel,
       RaisedCosineKernel, AlphaKernel, FunctionKernel,
       support, resolve_kernel,
       find_spikes_ref, AbstractNeuronModel,
       GenericNeuronModel, simulate_network,
       update_equation!, detect_spike,
       IntegrateAndFire, CurrentGenerator,
       SpikeRecorder, reset_callback, poisson_spike_train

include("kernels.jl")
include("models.jl")
include("spiketrain_types.jl")
include("spiking.jl")
include("demo_iaf.jl")

end
