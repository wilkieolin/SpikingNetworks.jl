# Getting Started

## Environment Setup (Linux)

### 1. Install Julia

Use **[juliaup](https://github.com/JuliaLang/juliaup)** (recommended, manages versions):

```bash
curl -fsSL https://install.julialang.org | sh
# restart shell, then:
juliaup add 1.11   # Project.toml compat; 1.12+ also works
juliaup default 1.11
```

Verify: `julia --version` → `1.11.x` (or `1.12.x`)

### 2. Bootstrap the Project Environment

```bash
cd SpikingNetworks.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Installs `DifferentialEquations`, `Plots`, `Random`, `Test` per `Project.toml`.

### 3. Jupyter / IJulia Kernel (for VS Code notebooks)

```bash
julia --project=. -e 'using Pkg; Pkg.add("IJulia")'
julia --project=. -e 'using IJulia; installkernel("SpikingNetworks", env=Dict("JULIA_PROJECT"=>pwd()))'
```

- Registers a kernel named **"SpikingNetworks"** tied to this project's environment.
- In VS Code: open `notebooks/demo_iaf_interactive.ipynb` → Select Kernel → **SpikingNetworks**.

> **Plots backend**: VS Code with display works out of the box with GR (default). For interactive plots, `Pkg.add("PlotlyJS")` and `using PlotlyJS; plotlyjs()` in the notebook.

### 4. VS Code Setup

1. Install **Julia** extension (`julialang.language-julia`)
2. Open folder `SpikingNetworks.jl`
3. Command Palette → **Julia: Select Environment** → Choose this folder (auto-detects `Project.toml`)
4. Open `.ipynb` files directly — they run via the registered kernel.

### 5. Run the Demos

| Method | Command |
|--------|---------|
| Script (headless) | `julia --project=. scripts/demo_iaf.jl` |
| Notebook (interactive) | `jupyter notebook notebooks/` → open `demo_iaf_interactive.ipynb` |
| VS Code | Open notebook → Run All (kernel: **SpikingNetworks**) |

### 6. Run Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```