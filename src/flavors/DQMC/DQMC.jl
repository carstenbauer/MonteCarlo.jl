include("linalg.jl")
include("stack.jl")

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

    all_checks::Bool # e.g. check if propagation is stable/instable (default should be true)
    safe_mult::Int

    Δτ::Float64
    slices::Int
    β::Float64 # redundant (=slices*Δτ) but keep for convenience

    DQMCParameters() = new()
end

"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, GreensType<:Number, ConfType, Checkerboard<:Bool} <: MonteCarloFlavor
    model::M
    conf::ConfType
    # greens::GreensType # should this be here or in DQMCStack?
    energy::Float64
    s::DQMCStack

    p::DQMCParameters

    a::DQMCAnalysis
    obs::Dict{String, Observable}

    function DQMC{M,GreensType}() where {M<:Model,GreensType<:Number}
        @assert isleaftype(GreensType)
        new()
    end
end

"""
    DQMC(m::M; kwargs...) where M<:Model

Create a determinant quantum Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function DQMC(m::M; sweeps::Int=1000, thermalization::Int=0,
            slices::Int=0, β::Float64=1.0, Δτ::Float64::0.1, # typically a user wants to specify beta not slices
            global_moves::Bool=false, global_rate::Int=5,
            seed::Int=-1,
            checkerboard::Bool=false) where M<:Model
    mc = DQMC{M, greenstype(m), conftype(m), checkerboard}()
    mc.model = m

    # default params
    mc.p = MCParameters()

    # number of imaginary time slices
    if slices<=0 # user didn't specify slices (use beta and Δτ keywords)
        try
            mc.p.slices = β/Δτ
            mc.p.β = β
            mc.p.delta_tau = Δτ
        catch
            error("Number of imaginary time slices, i.e. β/Δτ, must be an integer.")
        end
    else # user did specify slices (ignore beta keyword)
        mc.p.slices = slices
        mc.p.Δτ = Δτ
        mc.p.β = slices * Δτ
    end

    mc.p.global_moves = global_moves
    mc.p.global_rate = global_rate
    mc.p.thermalization = thermalization
    mc.p.sweeps = sweeps

    init!(mc, seed=seed)
    return mc
end

"""
    DQMC(m::M; kwargs::Dict{String, Any})

Create a determinant quantum Monte Carlo simulation for model `m` with (keyword) parameters
as specified in the dictionary `kwargs`.
"""
function DQMC(m::M, kwargs::Dict{String, Any}) where M<:Model
    DQMC(m; convert(Dict{Symbol, Any}, kwargs)...)
end


"""
    init!(mc::DQMC[; seed::Real=-1])

Initialize the determinant quantum Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `srand(seed)`.
"""
function init!(mc::DQMC; seed::Real=-1)
    seed == -1 || srand(seed)

    mc.conf = rand(mc, mc.model)
    mc.energy = energy(mc, mc.model, mc.conf)

    mc.obs = prepare_observables(mc, mc.model)

    mc.a = DQMCAnalysis()
    nothing
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
