using Documenter
using Pkg

# Develop the local SpikingNetworks package
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
using SpikingNetworks

makedocs(;
    sitename="SpikingNetworks.jl",
    format=Documenter.HTML(; prettyurls=false),
    modules=[SpikingNetworks],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorial: Integrate-and-Fire Network" => "tutorial_iaf.md",
        "Custom Neuron Models" => "custom_models.md",
        "Callbacks & Recording" => "callbacks.md",
        "API Reference" => "api_reference.md",
    ],
    warnonly=:cross_references,
    doctest=false,
)

# Local build only - no deploydocs