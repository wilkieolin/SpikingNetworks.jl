using Test
using SpikingNetworks

include("spiking_operations_tests.jl")
include("model_tests.jl")
include("types_tests.jl")
include("spiketrain_tests.jl")
include("kernel_tests.jl")
include("spike_detection_tests.jl")
include("simulation_tests.jl")

@testset "SpikingNetworks.jl" begin
    spiking_operations_tests()
    model_tests()
    types_tests()
    spiketrain_tests()
    kernel_tests()
    spike_detection_tests()
    simulation_tests()
end
