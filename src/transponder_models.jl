"""
Superconducting transponder neuron models for SpikingNetworks.

Three variants:
1. SimplifiedTransponder — Integrate-and-Fire (variable-resistor nanowire)
2. FullPhysicsTransponder — Adaptive/Resonate-and-Fire (nTron hotspot dynamics)
3. ResonateAndFireTransponder — Explicit R&F (LC tank oscillator)
"""

using LinearAlgebra
using StaticArrays
using SparseArrays
using Random
using Distributions
using NPZ
using DifferentialEquations

# =====================================================================
# 1. SimplifiedTransponder — Integrate-and-Fire
# =====================================================================

struct SimplifiedParams
    Ib1::Float32
    Ib2::Float32
    Ls1::Float32
    Rs1::Float32
    LT::Float32
    RT::Float32
    Lnw::Float32
    Rnw_max::Float32
    Ithresh1::Float32
    Ithresh2::Float32
    ref_period::Float32
    t_rise::Float32
    t_fall::Float32
    t_width::Float32
    output_tau::Float32
    output_x0::Float32
    output_mu0::Float32
end

function SimplifiedParams(;
    Ib1 = Float32(10.0e-6),
    Ib2 = Float32(20.0e-6),
    Ls1 = Float32(5.0e-9),
    Rs1 = Float32(3.0),
    LT = Float32(140.0e-9),
    RT = Float32(4.0),
    Lnw = Float32(10.0e-9),
    Rnw_max = Float32(1000.0),
    Ithresh1 = Float32(1.0e-6),
    Ithresh2 = Float32(2.0e-6),
    ref_period = Float32(5.0e-9),
    t_rise = Float32(1.0e-9),
    t_fall = Float32(1.0e-9),
    t_width = Float32(1.0e-9),
    output_tau = Float32(1.0e-9),
    output_x0 = Float32(1.0e-9),
    output_mu0 = Float32(0.2e-9),
)
    SimplifiedParams(
        Float32(Ib1), Float32(Ib2), Float32(Ls1), Float32(Rs1),
        Float32(LT), Float32(RT), Float32(Lnw), Float32(Rnw_max),
        Float32(Ithresh1), Float32(Ithresh2), Float32(ref_period),
        Float32(t_rise), Float32(t_fall), Float32(t_width),
        Float32(output_tau), Float32(output_x0), Float32(output_mu0)
    )
end

struct SimplifiedState
    i_nw::Float32
    i_loop::Float32
    ref_timer::Float32
    output_firing::Float32
    output_pulse_start::Float32
end

SimplifiedState() = SimplifiedState(0.0f0, 0.0f0, 0.0f0, 0.0f0, -1.0f0)

mutable struct SimplifiedTransponder <: AbstractNeuronModel
    n::Int
    params::Vector{SimplifiedParams}
    state::Vector{SimplifiedState}
    W::Matrix{Float32}
    D::Matrix{Float32}
    L_coup::Matrix{Float32}
    pending_pulses::Vector{Tuple{Float32, Float32, Int}}
end

function SimplifiedTransponder(n::Int; params=nothing, W=nothing, D=nothing, L_coup=nothing)
    if params === nothing
        params = [SimplifiedParams() for _ in 1:n]
    end
    state = [SimplifiedState() for _ in 1:n]
    if W === nothing
        W = zeros(Float32, n, n)
    end
    if D === nothing
        D = zeros(Float32, n, n)
    end
    if L_coup === nothing
        L_coup = zeros(Float32, n, n)
    end
    SimplifiedTransponder(n, params, state, W, D, L_coup, Tuple{Float32, Float32, Int}[])
end

function Base.length(model::SimplifiedTransponder)
    model.n
end

function nanowire_resistance(t::Float32, t0::Float32, p::SimplifiedParams)
    dt = t - t0
    if dt < -p.t_rise || dt > p.t_width + p.t_fall
        return 0.0f0
    elseif dt < 0.0f0
        return p.Rnw_max * (dt + p.t_rise) / p.t_rise
    elseif dt < p.t_width
        return p.Rnw_max
    else
        return p.Rnw_max * (p.t_fall - (dt - p.t_width)) / p.t_fall
    end
end

function output_pulse_shape(t::Float32, t_start::Float32, p::SimplifiedParams)
    x = t - t_start
    if x < 0.0f0
        return 0.0f0
    end
    sigmoid = 1.0f0 - 1.0f0 / (exp((x - p.output_x0) / p.output_mu0) + 1.0f0)
    decay = exp(-x / p.output_tau)
    return sigmoid * decay
end

function compute_coupling_input(t::Float32, model::SimplifiedTransponder, i::Int)
    input = 0.0f0
    for (pulse_time, amplitude, src) in model.pending_pulses
        if t >= pulse_time
            delay = model.D[i, src]
            if t >= pulse_time + delay
                input += amplitude * model.W[i, src]
            end
        end
    end
    return input
end

function update_equation!(dz, z, I, t, params, model::SimplifiedTransponder; connections=nothing)
    n = model.n
    p_vec = model.params
    state_vec = model.state
    W = connections !== nothing ? connections : model.W

    dz_idx = 1
    for i in 1:n
        p = p_vec[i]
        s = state_vec[i]

        i_nw = z[dz_idx]
        i_loop = z[dz_idx + 1]
        ref_timer = z[dz_idx + 2]
        output_firing = z[dz_idx + 3]
        output_pulse_start = z[dz_idx + 4]

        input_coupling = compute_coupling_input(t, model, i)
        i1 = p.Ib1 + input_coupling

        Rnw = 0.0f0
        if ref_timer > 0.0f0
            Rnw = nanowire_resistance(t, t - ref_timer, p)
        end

        denom = p.Lnw * (p.Ls1 - p.LT) - p.Ls1 * p.LT
        if abs(denom) < Float32(1e-30)
            denom = Float32(1e-30) * sign(denom)
        end

        di_nw_dt = (-i_nw * p.Ls1 * Rnw - i1 * p.LT * p.Rs1 + i_loop * p.LT * p.Rs1 +
                    i_nw * p.LT * (Rnw + p.Rs1) + i_loop * p.Ls1 * p.RT) / denom

        di_loop_dt = (-i_nw * p.Ls1 * Rnw - i1 * p.Lnw * p.Rs1 + i_loop * p.Lnw * p.Rs1 +
                      i_nw * p.Lnw * p.Rs1 + i_loop * p.Lnw * p.RT + i_loop * p.Ls1 * p.RT) / denom

        dref_dt = ref_timer > 0.0f0 ? 1.0f0 : 0.0f0

        doutput_firing_dt = 0.0f0
        doutput_pulse_start_dt = 0.0f0

        if ref_timer >= p.ref_period
            if i_loop > p.Ithresh2 && output_firing == 0.0f0
                output_firing = 1.0f0
                output_pulse_start = t
                doutput_firing_dt = 1.0f0 / 1.0e-12
                doutput_pulse_start_dt = 1.0f0
            end
        elseif output_firing > 0.0f0 && t - output_pulse_start > p.t_width + p.t_fall
            output_firing = 0.0f0
            output_pulse_start = -1.0f0
        end

        dz[dz_idx] = di_nw_dt
        dz[dz_idx + 1] = di_loop_dt
        dz[dz_idx + 2] = dref_dt
        dz[dz_idx + 3] = doutput_firing_dt
        dz[dz_idx + 4] = doutput_pulse_start_dt

        dz_idx += 5
    end

    new_pending = Tuple{Float32, Float32, Int}[]
    for (pulse_time, amplitude, src) in model.pending_pulses
        if t < pulse_time + maximum(model.D[:, src]) + maximum(p_vec[1].t_width + p_vec[1].t_fall, Float32(1.0e-9))
            push!(new_pending, (pulse_time, amplitude, src))
        end
    end
    model.pending_pulses = new_pending

    dz_idx = 1
    for i in 1:n
        p = p_vec[i]
        output_firing = z[dz_idx + 3]
        output_pulse_start = z[dz_idx + 4]
        if output_firing > 0.0f0 && output_pulse_start > 0.0f0 && abs(t - output_pulse_start) < 1.0e-12
            amplitude = p.Ithresh2 * 10.0f0
            push!(model.pending_pulses, (t, amplitude, i))
        end
        dz_idx += 5
    end
end

function detect_spike(z, t, threshold, model::SimplifiedTransponder)
    n = model.n
    indices = CartesianIndex{1}[]
    times = Float32[]
    for i in 1:n
        i_loop = z[(i-1)*5 + 2]
        if i_loop > threshold
            push!(indices, CartesianIndex(i))
            push!(times, Float32(t))
        end
    end
    shape = (n,)
    return SpikeTrain(indices, times, shape, 0.0f0)
end

function initial_state(model::SimplifiedTransponder)
    n = model.n
    z0 = zeros(Float32, n * 5)
    for i in 1:n
        idx = (i-1)*5
        z0[idx + 1] = 0.0f0
        z0[idx + 2] = 0.0f0
        z0[idx + 3] = 0.0f0
        z0[idx + 4] = 0.0f0
        z0[idx + 5] = -1.0f0
    end
    return z0
end

# =====================================================================
# 2. FullPhysicsTransponder — Adaptive/Resonate-and-Fire (nTron hotspot dynamics)
# =====================================================================

# Physical constants
const S = Float32(1.0)
const Jc = Float32(40.0e9)
const TH = Float32(19.0e-9)
const SHEET = Float32(130.0)
const TC = Float32(8.5)
const TSUB = Float32(4.3)
const W_G = Float32(20.0e-9) / S
const W_S = Float32(200.0e-9) / S
const W_D = Float32(200.0e-9) / S
const W_C = Float32(250.0e-9) / S
const SQ_D = Float32(400.0) / S
const SQ_S = Float32(80.0) / S
const SQ_C = Float32(10.0) / S
const SQ_G_IN = Float32(10.0)
const SQ_G_OUT = Float32(5.0) / S
const HC = Float32(50.0e3)
const HEATCAP = Float32(4400.0)
const A1 = Float32(0.4)

const IND = SHEET * TH / HC
const L1P = IND * (SQ_G_IN + SQ_S + SQ_C + SQ_D + SQ_G_OUT)
const SQ_S2 = SQ_S / 2.0f0
const LB = L1P - (SQ_G_OUT + SQ_S2) * IND

const PSI = SHEET * (Jc * TH)^2 / (HC * (TC - TSUB))
const KAPPA = Float32(2.44e-8) * TC / (SHEET * TH)
const VO = sqrt(HC * KAPPA / TH) / HEATCAP
const ISW_G = Jc * W_G * TH
const ISW_C = Jc * W_C * TH
const ISW_S = Jc * W_S * TH
const ISW_D = Jc * W_D * TH
const VTH_G = sqrt(Float32(2.0) / PSI) * ISW_G
const VTH_C = sqrt(Float32(2.0) / PSI) * ISW_C
const RN_C2 = SHEET * SQ_C / Float32(2.0)
const RN_S = SHEET * SQ_S
const RN_D = SHEET * SQ_D
const C_G = W_G / (SHEET * VO)
const C_CHOKE = W_C / (SHEET * VO)
const C_SRC = W_S / (SHEET * VO)
const C_DRN = W_D / (SHEET * VO)
const R_SH = SHEET * SQ_S
const V_BIAS = Float32(1.0)
const R_BIAS = Float32(10.0e3) + Float32(25.0)
const R_LOOP = Float32(10.0)

function _g(i, Isw)
    u = min(0.6f0 * abs(i) / Isw, 0.999999f0)
    return sqrt(1.0f0 - u^2) + u * asin(u)
end

function dphi_di(i, L, Isw, N, Rn)
    if abs(i) <= Isw
        return L
    else
        return L * (1.0f0 - N / Rn) + Rn / (Jc * TH * _g(i, Isw))
    end
end

function dphi_dr(i, L, Isw, Rn)
    if abs(i) <= Isw
        return 0.0f0
    else
        return L * abs(i) / (Jc * TH * _g(i, Isw))
    end
end

function vel(i, Isw)
    return VO * (abs(i) / Isw - 1.0f0)
end

struct NTronParams
    name::String
    sq_g::Float32
    Lg::Float32
    Ld::Float32
    Ls::Float32
    Lc2::Float32
    Rn_g::Float32
end

function NTronParams(name::String, sq_g::Float32)
    Lg = sq_g * IND
    Ld = (SQ_D + SQ_C) * IND
    Ls = (SQ_S + SQ_C) * IND
    Lc2 = SQ_C * IND
    Rn_g = SHEET * sq_g
    NTronParams(name, sq_g, Lg, Ld, Ls, Lc2, Rn_g)
end

struct NTronState
    hs::SVector{5, Float32}
end

NTronState() = NTronState(SVector{5, Float32}(zeros(5)))

mutable struct NTron
    params::NTronParams
    state::NTronState
end

function NTron(name::String, sq_g::Float32)
    NTron(NTronParams(name, sq_g), NTronState())
end

function branch_LR(nt::NTron, ig, idr, isr)
    N3, N003, N004, N03, N04 = nt.state.hs
    p = nt.params
    Lg = dphi_di(ig, p.Lg, ISW_G, N3, p.Rn_g)
    Ld = dphi_di(idr, p.Ld, ISW_D, N004, RN_D) +
         dphi_di(idr, p.Lc2, ISW_C, N003, RN_C2)
    Ls = dphi_di(isr, p.Lc2, ISW_C, N03, RN_C2) +
         dphi_di(isr, p.Ls, ISW_S, N04, RN_S)
    return (Lg, N3), (Ld, N003 + N004), (Ls, N03 + N04)
end

function cross_EMF(nt::NTron, ig, idr, isr, d_rates)
    p = nt.params
    eg = -dphi_dr(ig, p.Lg, ISW_G, p.Rn_g) * d_rates[1]
    ed = -(dphi_dr(idr, p.Ld, ISW_D, RN_D) * d_rates[3] +
           dphi_dr(idr, p.Lc2, ISW_C, RN_C2) * d_rates[2])
    es = -(dphi_dr(isr, p.Lc2, ISW_C, RN_C2) * d_rates[4] +
           dphi_dr(isr, p.Ls, ISW_S, RN_S) * d_rates[5])
    return eg, ed, es
end

function rates(nt::NTron, ig, idr, isr)
    N3, N003, N004, N03, N04 = nt.state.hs
    gn = (abs(ig) > ISW_G) || (abs(ig) * N3 > VTH_G)
    dN3 = (gn && N3 < nt.params.Rn_g) ? vel(ig, ISW_G) / C_G : 0.0f0
    sup = A1 * exp(-(abs(ig) - ISW_G) / HEATCAP) * (abs(ig) > ISW_G ? 1.0f0 : 0.0f0) + (abs(ig) <= ISW_G ? 1.0f0 : 0.0f0)
    ic = ISW_C * sup
    dn = (abs(idr) > ic) || ((N003 + N004) * abs(idr) > VTH_C)
    dN003 = (dn && N003 <= RN_C2) ? vel(idr, ISW_C) / C_CHOKE : 0.0f0
    dN004 = (dn && N003 > RN_C2 && N004 < RN_D) ? vel(idr, ISW_D) / C_DRN : 0.0f0
    sn = (abs(isr) > ic) || ((N03 + N04) * abs(isr) > VTH_C)
    dN03 = (sn && N03 <= RN_C2) ? vel(isr, ISW_C) / C_CHOKE : 0.0f0
    dN04 = (sn && N03 > RN_C2 && N04 < RN_S) ? vel(isr, ISW_S) / C_SRC : 0.0f0
    return SVector{5, Float32}(dN3, dN003, dN004, dN03, dN04), (gn, dn, sn)
end

function step!(nt::NTron, d_rates, flags, h)
    gn, dn, sn = flags
    new_hs = nt.state.hs + h * d_rates
    caps = SVector{5, Float32}(nt.params.Rn_g, RN_C2, RN_D, RN_C2, RN_S)
    new_hs = min.(max.(new_hs, 0.0f0), caps)
    if !gn
        new_hs = setindex(new_hs, 0.0f0, 1)
    end
    if !dn
        new_hs = setindex(new_hs, 0.0f0, 2)
        new_hs = setindex(new_hs, 0.0f0, 3)
    end
    if !sn
        new_hs = setindex(new_hs, 0.0f0, 4)
        new_hs = setindex(new_hs, 0.0f0, 5)
    end
    nt.state = NTronState(new_hs)
end

struct Branch
    a::Int
    b::Int
    tag::Tuple
end

mutable struct Net
    br::Vector{Branch}
    nodes::Set{Int}
    M::Matrix{Float32}
    links::Vector{Int}
    tr::Dict{String, NTron}
    bmap::Dict{String, Int}
    dt_ns::Float32
    node_ids::Dict{String, Int}
end

Net() = Net(Branch[], Set{Int}(), Matrix{Float32}(undef,0,0), Int[], Dict{String, NTron}(), Dict{String, Int}(), Float32(0.0), Dict{String, Int}())

function add_branch!(net::Net, a::String, b::String, tag::Tuple)
    # Map string node names to integer IDs
    if !haskey(net.node_ids, a)
        net.node_ids[a] = length(net.node_ids) + 1
    end
    if !haskey(net.node_ids, b)
        net.node_ids[b] = length(net.node_ids) + 1
    end
    na = net.node_ids[a]
    nb = net.node_ids[b]
    push!(net.nodes, na)
    push!(net.nodes, nb)
    push!(net.br, Branch(na, nb, tag))
    length(net.br)
end

function build_mesh!(net::Net)
    # Map node IDs to consecutive indices 1..N
    node_list = collect(net.nodes)
    node_to_idx = Dict(node_list[i] => i for i in 1:length(node_list))
    N = length(node_list)
    B = length(net.br)
    adj = [Tuple{Int,Int,Int}[] for _ in 1:N]
    for (k, br) in enumerate(net.br)
        ia = node_to_idx[br.a]
        ib = node_to_idx[br.b]
        push!(adj[ia], (ib, k, 1))
        push!(adj[ib], (ia, k, -1))
    end
    # Find ground node index (node "0" should have node_ids["0"] = 1)
    ground_idx = node_to_idx[1]  # ground node has ID 1 in node_ids
    par = fill(-1, N)
    pbr = fill((0, 0), N)
    seen = falses(N)
    queue = Int[ground_idx]
    seen[ground_idx] = true
    tree = Set{Int}()
    while !isempty(queue)
        u = popfirst!(queue)
        for (v, k, s) in adj[u]
            if !seen[v]
                seen[v] = true
                par[v] = u
                pbr[v] = (k, s)
                push!(tree, k)
                push!(queue, v)
            end
        end
    end
    function path_to_ground(x)
        out = Tuple{Int, Int}[]
        while x != ground_idx
            k, s = pbr[x]
            push!(out, (k, s))
            x = par[x]
        end
        return out
    end
    links = [k for k in 1:B if k ∉ tree]
    M = zeros(Float32, length(links), B)
    for (r, k) in enumerate(links)
        br = net.br[k]
        ia = node_to_idx[br.a]
        ib = node_to_idx[br.b]
        M[r, k] = 1.0f0
        for (kk, s) in path_to_ground(ib)
            M[r, kk] += s
        end
        for (kk, s) in path_to_ground(ia)
            M[r, kk] -= s
        end
    end
    net.M = M
    net.links = links
end

function pulse(t, width)
    if t < 0.0f0 || t > width
        return 0.0f0
    elseif t < Float32(0.1e-9)
        return Float32(10.0e3) * t / Float32(0.1e-9)
    elseif t > width - Float32(0.1e-9)
        return Float32(10.0e3) * (width - t) / Float32(0.1e-9)
    else
        return Float32(10.0e3)
    end
end

function dc_operating_point(net::Net)
    nB = length(net.br)
    nL = size(net.M, 1)
    Lv = zeros(Float32, nB)
    Rv = zeros(Float32, nB)
    Ev = zeros(Float32, nB)
    for k in 1:nB
        br = net.br[k]
        tag = br.tag
        if tag[1] == "src"
            Lv[k], Rv[k] = tag[4], tag[3]
            Ev[k] = 0.0f0
        elseif tag[1] == "dc"
            Lv[k], Rv[k], Ev[k] = tag[4], tag[3], tag[2]
        elseif tag[1] == "lr"
            Lv[k], Rv[k], Ev[k] = tag[2], tag[3], 0.0f0
        else
            nm, term = tag[2], tag[3]
            dev = net.tr[nm]
            Lg, Ld, Ls = dev.params.Lg, dev.params.Ld, dev.params.Ls
            if term == "g"
                Lv[k], Rv[k] = Lg, 0.0f0
            elseif term == "d"
                Lv[k], Rv[k] = Ld, 0.0f0
            else
                Lv[k], Rv[k] = Ls, 0.0f0
            end
            Ev[k] = 0.0f0
        end
    end
    Rv .+= Float32(1e-9)
    return net.M * Diagonal(Rv) * net.M' \ (net.M * Ev)
end

function build_snn2(; bv=Float32(1.5), dt_ns=Float32(0.0))
    net = Net()
    tr = Dict{String, NTron}()
    for (nm, sq) in [("U1", SQ_G_IN), ("U4", SQ_G_IN), ("U3", SQ_G_IN),
                     ("U6", SQ_G_IN), ("U2", SQ_G_OUT), ("U5", SQ_G_OUT)]
        tr[nm] = NTron(nm, sq)
    end
    net.tr = tr
    b = Dict{String, Int}()
    b["p1"] = add_branch!(net, "0", "N048", ("src", "V1", Float32(10.0e3), Float32(5.0e-9)))
    b["p2"] = add_branch!(net, "0", "N037", ("src", "V4", Float32(10.0e3), Float32(5.0e-9)))
    for (nm, gnode, dnode, c) in [
        ("U1", "N048", "N024", "N005"),
        ("U4", "N037", "N024", "N006"),
        ("U3", "N005", "N024", "N007"),
        ("U6", "N006", "N024", "N008"),
        ("U2", "N007", "N044", "bias11"),
        ("U5", "N008", "N044", "bias1"),
    ]
        b[nm * "g"] = add_branch!(net, gnode, c, ("nt", nm, "g"))
        b[nm * "d"] = add_branch!(net, dnode, c, ("nt", nm, "d"))
        b[nm * "s"] = add_branch!(net, "0", c, ("nt", nm, "s"))
    end
    for nd in ["bias11", "bias1", "bias2", "bias3", "bias4", "pulser2"]
        b["sh" * nd] = add_branch!(net, nd, "0", ("lr", Float32(5.0e-9), R_SH))
    end
    for (nd, V) in [("bias1", V_BIAS), ("bias2", V_BIAS), ("bias3", V_BIAS), ("bias4", V_BIAS),
                    ("bias11", V_BIAS), ("pulser2", V_BIAS)]
        b["bt" * nd] = add_branch!(net, "0", nd, ("dc", V, R_BIAS, Float32(5.0e-9)))
    end
    for (src, gate) in [("bias3", "N021"), ("bias1", "N039"),
                        ("N021", "N005"), ("N039", "N006")]
        b["$(src)->$(gate)"] = add_branch!(net, src, gate, ("lr", Float32(5.0e-9), Float32(20.0)))
    end
    b["cpA"] = add_branch!(net, "bias3", "N021", ("lr", LB, Float32(2.0) * R_LOOP))
    b["cpB"] = add_branch!(net, "bias1", "N039", ("lr", LB, Float32(2.0) * R_LOOP))
    net.bmap = b
    net.dt_ns = dt_ns
    build_mesh!(net)
    return net
end

struct FullPhysicsTransponder <: AbstractNeuronModel
    net::Net
    probe_branches::Vector{String}
    tstop::Float32
    h::Float32
end

function FullPhysicsTransponder(; bv=Float32(1.5), dt_ns=Float32(0.0), tstop=Float32(30.0e-9), h=Float32(1.0e-12), probe=("cpA",))
    net = build_snn2(bv=bv, dt_ns=dt_ns)
    FullPhysicsTransponder(net, collect(probe), tstop, h)
end

function Base.length(model::FullPhysicsTransponder)
    length(model.net.tr)
end

function update_equation!(dz, z, I, t, params, model::FullPhysicsTransponder; connections=nothing)
    net = model.net
    M = net.M
    nB = length(net.br)
    nL = size(M, 1)
    il = z[1:nL]
    td_ns = net.dt_ns
    V1_func(t) = pulse(t - Float32(5.0e-9), Float32(10.0e-9))
    V4_func(t) = pulse(t - Float32(5.0e-9), Float32(10.0e-9) - td_ns * Float32(1.0e-9))
    ib = M' * il
    rates_dict = Dict{String, SVector{5, Float32}}()
    flags_dict = Dict{String, Tuple{Bool, Bool, Bool}}()
    for (nm, dev) in net.tr
        ig = ib[net.bmap[nm * "g"]]
        idr = ib[net.bmap[nm * "d"]]
        isr = ib[net.bmap[nm * "s"]]
        rates_dict[nm], flags_dict[nm] = rates(dev, ig, idr, isr)
    end
    Lv = zeros(Float32, nB)
    Rv = zeros(Float32, nB)
    Ev = zeros(Float32, nB)
    for k in 1:nB
        br = net.br[k]
        tag = br.tag
        if tag[1] == "src"
            Lv[k], Rv[k] = tag[4], tag[3]
            Ev[k] = (tag[2] == "V1" ? V1_func(t) : V4_func(t))
        elseif tag[1] == "dc"
            Lv[k], Rv[k], Ev[k] = tag[4], tag[3], tag[2]
        elseif tag[1] == "lr"
            Lv[k], Rv[k], Ev[k] = tag[2], tag[3], 0.0f0
        else
            nm, term = tag[2], tag[3]
            dev = net.tr[nm]
            ig = ib[net.bmap[nm * "g"]]
            idr = ib[net.bmap[nm * "d"]]
            isr = ib[net.bmap[nm * "s"]]
            (Lg, Rg), (Ld, Rd), (Ls, Rs) = branch_LR(dev, ig, idr, isr)
            eg, ed, es = cross_EMF(dev, ig, idr, isr, rates_dict[nm])
            if term == "g"
                Lv[k], Rv[k], Ev[k] = Lg, Rg, eg
            elseif term == "d"
                Lv[k], Rv[k], Ev[k] = Ld, Rd, ed
            else
                Lv[k], Rv[k], Ev[k] = Ls, Rs, es
            end
        end
    end
    Lm = M * Diagonal(Lv) * M'
    Rm = M * Diagonal(Rv) * M'
    Em = M * Ev
    h_step = model.h
    A = Lm / h_step + Rm / 2.0f0
    rhs = (Lm / h_step - Rm / 2.0f0) * il + Em
    il_new = A \ rhs
    for (nm, dev) in net.tr
        step!(dev, rates_dict[nm], flags_dict[nm], h_step)
    end
    ibn = M' * il_new
    dz[1:nL] = (il_new - il) / h_step
end

function detect_spike(z, t, threshold, model::FullPhysicsTransponder)
    net = model.net
    indices = CartesianIndex{1}[]
    times = Float32[]
    for (i, nm) in enumerate(keys(net.tr))
        if nm in ["U2", "U5"]
            ib = net.M' * z[1:size(net.M, 1)]
            cp_curr = ib[net.bmap[nm == "U2" ? "cpA" : "cpB"]]
            if cp_curr > threshold
                push!(indices, CartesianIndex(i))
                push!(times, Float32(t))
            end
        end
    end
    shape = (length(net.tr),)
    return SpikeTrain(indices, times, shape, 0.0f0)
end

function initial_state(model::FullPhysicsTransponder)
    net = model.net
    il0 = dc_operating_point(net)
    return il0
end

function get_M_matrix(model::FullPhysicsTransponder)
    return model.net.M
end

function get_bmap(model::FullPhysicsTransponder)
    return model.net.bmap
end

function get_probe_currents(model::FullPhysicsTransponder, z::AbstractVector)
    net = model.net
    il = z[1:size(net.M, 1)]
    ib = net.M' * il
    return [ib[net.bmap[branch]] for branch in model.probe_branches]
end

# =====================================================================
# 3. ResonateAndFireTransponder — Explicit R&F (LC tank)
# =====================================================================

struct RnFParams
    L_tank::Float32
    C_tank::Float32
    R_damp::Float32
    I_thresh::Float32
    ref_period::Float32
    t_rise::Float32
    t_fall::Float32
    t_width::Float32
    output_tau::Float32
    output_x0::Float32
    output_mu0::Float32
end

function RnFParams(;
    L_tank = Float32(10.0e-9),
    C_tank = Float32(0.5e-12),
    R_damp = Float32(10.0),
    I_thresh = Float32(2.0e-6),
    ref_period = Float32(5.0e-9),
    t_rise = Float32(0.5e-9),
    t_fall = Float32(0.5e-9),
    t_width = Float32(1.0e-9),
    output_tau = Float32(1.0e-9),
    output_x0 = Float32(1.0e-9),
    output_mu0 = Float32(0.2e-9),
)
    RnFParams(
        Float32(L_tank), Float32(C_tank), Float32(R_damp),
        Float32(I_thresh), Float32(ref_period),
        Float32(t_rise), Float32(t_fall), Float32(t_width),
        Float32(output_tau), Float32(output_x0), Float32(output_mu0)
    )
end

resonance_frequency(p::RnFParams) = 1.0f0 / (2.0f0 * Float32(pi) * sqrt(p.L_tank * p.C_tank))
quality_factor(p::RnFParams) = p.R_damp * sqrt(p.C_tank / p.L_tank)

struct RnFState
    i_tank::Float32
    q_cap::Float32
    ref_timer::Float32
    output_firing::Float32
    output_pulse_start::Float32
end

RnFState() = RnFState(0.0f0, 0.0f0, 0.0f0, 0.0f0, -1.0f0)

mutable struct ResonateAndFireTransponder <: AbstractNeuronModel
    n::Int
    params::Vector{RnFParams}
    state::Vector{RnFState}
    W::Matrix{Float32}
    D::Matrix{Float32}
    L_coup::Matrix{Float32}
    pending_pulses::Vector{Tuple{Float32, Float32, Int}}
end

function ResonateAndFireTransponder(n::Int; params=nothing, W=nothing, D=nothing, L_coup=nothing)
    if params === nothing
        params = [RnFParams() for _ in 1:n]
    end
    state = [RnFState() for _ in 1:n]
    if W === nothing
        W = zeros(Float32, n, n)
    end
    if D === nothing
        D = zeros(Float32, n, n)
    end
    if L_coup === nothing
        L_coup = zeros(Float32, n, n)
    end
    ResonateAndFireTransponder(n, params, state, W, D, L_coup, Tuple{Float32, Float32, Int}[])
end

function Base.length(model::ResonateAndFireTransponder)
    model.n
end

function nanowire_resistance(t::Float32, t0::Float32, p::RnFParams)
    dt = t - t0
    if dt < -p.t_rise || dt > p.t_width + p.t_fall
        return 0.0f0
    elseif dt < 0.0f0
        return p.R_damp * (dt + p.t_rise) / p.t_rise
    elseif dt < p.t_width
        return p.R_damp
    else
        return p.R_damp * (p.t_fall - (dt - p.t_width)) / p.t_fall
    end
end

function output_pulse_shape(t::Float32, t_start::Float32, p::RnFParams)
    x = t - t_start
    if x < 0.0f0
        return 0.0f0
    end
    sigmoid = 1.0f0 - 1.0f0 / (exp((x - p.output_x0) / p.output_mu0) + 1.0f0)
    decay = exp(-x / p.output_tau)
    return sigmoid * decay
end

function compute_coupling_input(t::Float32, model::ResonateAndFireTransponder, i::Int)
    input = 0.0f0
    for (pulse_time, amplitude, src) in model.pending_pulses
        if t >= pulse_time
            delay = model.D[i, src]
            if t >= pulse_time + delay
                input += amplitude * model.W[i, src]
            end
        end
    end
    return input
end

function update_equation!(dz, z, I, t, params, model::ResonateAndFireTransponder; connections=nothing)
    n = model.n
    p_vec = model.params
    state_vec = model.state
    W = connections !== nothing ? connections : model.W

    dz_idx = 1
    for i in 1:n
        p = p_vec[i]
        s = state_vec[i]

        i_tank = z[dz_idx]
        q_cap = z[dz_idx + 1]
        ref_timer = z[dz_idx + 2]
        output_firing = z[dz_idx + 3]
        output_pulse_start = z[dz_idx + 4]

        input_coupling = compute_coupling_input(t, model, i)
        I_inj = input_coupling + I[i]

        if ref_timer > 0.0f0
            di_tank_dt = -i_tank / 1.0e-12
            dq_cap_dt = -q_cap / 1.0e-12
            dref_dt = 1.0f0
        else
            di_tank_dt = (-i_tank * p.R_damp - q_cap / p.C_tank + I_inj) / p.L_tank
            dq_cap_dt = i_tank
            dref_dt = 0.0f0
        end

        doutput_firing_dt = 0.0f0
        doutput_pulse_start_dt = 0.0f0

        if ref_timer <= 0.0f0 && abs(i_tank) > p.I_thresh && output_firing == 0.0f0
            output_firing = 1.0f0
            output_pulse_start = t
            ref_timer = p.ref_period
            doutput_firing_dt = 1.0f0 / 1.0e-12
            doutput_pulse_start_dt = 1.0f0
        elseif output_firing > 0.0f0 && t - output_pulse_start > p.t_width + p.t_fall
            output_firing = 0.0f0
            output_pulse_start = -1.0f0
        end

        dz[dz_idx] = di_tank_dt
        dz[dz_idx + 1] = dq_cap_dt
        dz[dz_idx + 2] = dref_dt
        dz[dz_idx + 3] = doutput_firing_dt
        dz[dz_idx + 4] = doutput_pulse_start_dt

        dz_idx += 5
    end

    new_pending = Tuple{Float32, Float32, Int}[]
    for (pulse_time, amplitude, src) in model.pending_pulses
        if t < pulse_time + maximum(model.D[:, src]) + maximum(p_vec[1].t_width + p_vec[1].t_fall, Float32(1.0e-9))
            push!(new_pending, (pulse_time, amplitude, src))
        end
    end
    model.pending_pulses = new_pending

    dz_idx = 1
    for i in 1:n
        p = p_vec[i]
        output_firing = z[dz_idx + 3]
        output_pulse_start = z[dz_idx + 4]
        if output_firing > 0.0f0 && output_pulse_start > 0.0f0 && abs(t - output_pulse_start) < 1.0e-12
            amplitude = p.I_thresh * 10.0f0
            push!(model.pending_pulses, (t, amplitude, i))
        end
        dz_idx += 5
    end
end

function detect_spike(z, t, threshold, model::ResonateAndFireTransponder)
    n = model.n
    indices = CartesianIndex{1}[]
    times = Float32[]
    for i in 1:n
        i_tank = z[(i-1)*5 + 1]
        if abs(i_tank) > threshold
            push!(indices, CartesianIndex(i))
            push!(times, Float32(t))
        end
    end
    shape = (n,)
    return SpikeTrain(indices, times, shape, 0.0f0)
end

function initial_state(model::ResonateAndFireTransponder)
    n = model.n
    z0 = zeros(Float32, n * 5)
    for i in 1:n
        idx = (i-1)*5
        z0[idx + 1] = 0.0f0
        z0[idx + 2] = 0.0f0
        z0[idx + 3] = 0.0f0
        z0[idx + 4] = 0.0f0
        z0[idx + 5] = -1.0f0
    end
    return z0
end