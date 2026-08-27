"""
Temporal spike kernels — the shape of the current a single input spike injects.

Every kernel is callable on a time offset `dt = t - t_spike` and reports its own
truncation radius via `support`, so `spike_current` never has to guess how wide a
kernel is. Outer constructors normalise each kernel to unit area, which means
widening a kernel smears the same total charge over more time instead of also
scaling the total drive; use `SpikingArgs.spk_scale` for strength.
"""

abstract type SpikeKernel end

"""
    support(k::SpikeKernel) -> Float32

Truncation radius: contributions with `abs(dt) > support(k)` are treated as zero.
"""
function support end

# --- box / delta-like impulse -------------------------------------------------

struct DeltaKernel <: SpikeKernel
    width::Float32
    amplitude::Float32
end

DeltaKernel(width::Real=0.005f0) = DeltaKernel(Float32(width), 1.0f0 / Float32(width))

(k::DeltaKernel)(dt::Real) = abs(Float32(dt)) <= 0.5f0 * k.width ? k.amplitude : 0.0f0
support(k::DeltaKernel) = 0.5f0 * k.width

# --- Gaussian -----------------------------------------------------------------

struct GaussianKernel <: SpikeKernel
    sigma::Float32
    amplitude::Float32
end

function GaussianKernel(sigma::Real=0.05f0)
    s = Float32(sigma)
    return GaussianKernel(s, 1.0f0 / (s * sqrt(2.0f0 * Float32(pi))))
end

(k::GaussianKernel)(dt::Real) = k.amplitude * exp(-0.5f0 * (Float32(dt) / k.sigma)^2)
# 4 sigma retains ~99.99% of the area
support(k::GaussianKernel) = 4.0f0 * k.sigma

# --- raised cosine ------------------------------------------------------------

struct RaisedCosineKernel <: SpikeKernel
    halfwidth::Float32
    amplitude::Float32
end

function RaisedCosineKernel(halfwidth::Real=0.02f0)
    hw = Float32(halfwidth)
    return RaisedCosineKernel(hw, 1.0f0 / hw)
end

function (k::RaisedCosineKernel)(dt::Real)
    d = Float32(dt)
    abs(d) <= k.halfwidth || return 0.0f0
    return k.amplitude * 0.5f0 * (1.0f0 + cos(Float32(pi) * d / k.halfwidth))
end
support(k::RaisedCosineKernel) = k.halfwidth

# --- alpha function (causal) --------------------------------------------------

struct AlphaKernel <: SpikeKernel
    tau::Float32
    amplitude::Float32
end

AlphaKernel(tau::Real=0.05f0) = AlphaKernel(Float32(tau), 1.0f0)

function (k::AlphaKernel)(dt::Real)
    d = Float32(dt)
    d >= 0.0f0 || return 0.0f0
    return k.amplitude * (d / k.tau^2) * exp(-d / k.tau)
end
# causal, but the gate in spike_current is symmetric; the dt < 0 half evaluates to 0
support(k::AlphaKernel) = 8.0f0 * k.tau

# --- escape hatch for ad-hoc kernels -----------------------------------------

struct FunctionKernel <: SpikeKernel
    f::Function
    radius::Float32
end

FunctionKernel(f::Function) = FunctionKernel(f, Inf32)

(k::FunctionKernel)(dt::Real) = Float32(k.f(Float32(dt)))
support(k::FunctionKernel) = k.radius
