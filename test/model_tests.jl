using Test, SpikingNetworks

function model_tests()
    @testset "Neuron Models" begin
        # Protocol contract
        # second positional arg is the Erdos-Renyi connection *probability*
        model = IntegrateAndFire(4, 0.25f0)
        @test model isa AbstractNeuronModel
        @test size(model.W) == (4, 4)
        # GenericNeuronModel placeholder
        gm = GenericNeuronModel((4,), (self_decay=-0.2f0,))
        @test gm isa AbstractNeuronModel

        # Reset happens in the callback, never in the update equation: the RHS must
        # leave the solver's state untouched.
        z = Float32[0.0, 0.5, 1.0, 0.0]
        z_before = copy(z)
        dz = similar(z)
        update_equation!(dz, z, zeros(Float32, 4), 0.0f0, model, model)
        @test z == z_before
    end
end
