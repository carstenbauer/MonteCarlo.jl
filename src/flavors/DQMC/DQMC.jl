"""
Analysis data of determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMCAnalysis
    acc_rate::Float64
    prop_local::Int
    acc_local::Int
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int
    sweep_dur::Float64

    DQMCAnalysis() = new(0.,0,0,0.,0,0)
end

"""
Parameters of determinant quantum Monte Carlo (DQMC)
"""
mutable struct DQMCParameters
    global_moves::Bool
    global_rate::Int
    thermalization::Int # number of thermalization sweeps
    sweeps::Int # number of sweeps (after thermalization)

    DQMCParameters() = new()
end

"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, GreensType<:Number, ConfType} <: MonteCarloFlavor
    model::M
    conf::ConfType
    # greens::GreensType # should this be here or in DQMCStack?
    energy::Float64

    obs::Dict{String, Observable}
    s::DQMCStack
    p::DQMCParameters
    a::DQMCAnalysis

    function DQMC{M,GreensType}() where {M<:Model,GreensType<:Number}
        @assert isleaftype(GreensType)
        new()
    end
end






"""
    sweep(mc::DQMC)

Performs a sweep of local moves.
"""
function sweep(mc::DQMC{<:Model, S}) where S
    const N = mc.model.l.sites

    @inbounds for i in eachindex(mc.conf)
        ΔE, Δi = propose_local(mc.model, i, mc.conf, mc.energy)
        mc.a.prop_local += 1
        # Metropolis
        if ΔE <= 0 || rand() < exp(- ΔE)
            accept_local!(mc.model, i, mc.conf, mc.energy, Δi, ΔE)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy += ΔE
        end
    end

    nothing
end
