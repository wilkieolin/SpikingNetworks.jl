module SpikingNetworks

using DifferentialEquations
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using Distributions
using NPZ

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
       initial_state,
       IntegrateAndFire, CurrentGenerator,
       SpikeRecorder, reset_callback, poisson_spike_train,
       # Transponder models
       SimplifiedTransponder, SimplifiedParams, SimplifiedState,
       FullPhysicsTransponder,
       ResonateAndFireTransponder, RnFParams, RnFState,
       resonance_frequency, quality_factor,
       build_snn2, dc_operating_point, NTron, NTronParams, NTronState

include("kernels.jl")
include("models.jl")
include("spiketrain_types.jl")
include("spiking.jl")
include("demo_iaf.jl")
include("transponder_models.jl")

end
