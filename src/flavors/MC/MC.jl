"""
Analysis data of classical Monte Carlo simulation
"""
mutable struct MCAnalysis
    acc_rate::Float64
    prop_local::Int
    acc_local::Int
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int
    sweep_dur::Float64

    MCAnalysis() = new(0.,0,0,0.,0,0)
end

"""
Parameters of classical Monte Carlo
"""
mutable struct MCParameters
    global_moves::Bool
    global_rate::Int
    thermalization::Int # number of thermalization sweeps
    sweeps::Int # number of sweeps (after thermalization)

    β::Float64

    MCParameters() = new()
end

"""
Classical Monte Carlo simulation
"""
mutable struct MC{M<:Model, C} <: MonteCarloFlavor
    model::M
    conf::C
    energy::Float64

    obs::Dict{String, Observable}
    p::MCParameters
    a::MCAnalysis

    MC{M,C}() where {M,C} = new()
end

"""
    MC(m::M; kwargs...) where M<:Model

Create a classical Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function MC(m::M; sweeps::Int=1000, thermalization::Int=0, β::Float64=1.0, global_moves::Bool=false, global_rate::Int=5, seed::Int=-1) where M<:Model
    mc = MC{M, conftype(m)}()
    mc.model = m

    # default params
    mc.p = MCParameters()
    mc.p.β = β
    mc.p.global_moves = global_moves
    mc.p.global_rate = global_rate
    mc.p.thermalization = thermalization
    mc.p.sweeps = sweeps

    init!(mc, seed=seed)
    return mc
end

"""
    MC(m::M; kwargs::Dict{String, Any})

Create a classical Monte Carlo simulation for model `m` with (keyword) parameters
as specified in the dictionary `kwargs`.
"""
function MC(m::M, kwargs::Dict{String, Any}) where M<:Model
    MC(m; convert(Dict{Symbol, Any}, kwargs)...)
end


"""
    init!(mc::MC[; seed::Real=-1])

Initialize the classical Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `srand(seed)`.
"""
function init!(mc::MC; seed::Real=-1)
    seed == -1 || srand(seed)

    mc.conf = rand(mc, mc.model)
    mc.energy = energy(mc, mc.model, mc.conf)

    mc.obs = prepare_observables(mc, mc.model)

    mc.a = MCAnalysis()
    nothing
end

"""
    run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given classical Monte Carlo simulation `mc`.
Progress will be printed to `STDOUT` if `verborse=true` (default).
"""
function run!(mc::MC; verbose::Bool=true, sweeps::Int=mc.p.sweeps, thermalization=mc.p.thermalization)
    mc.p.sweeps = sweeps
    mc.p.thermalization = thermalization
    const total_sweeps = mc.p.sweeps + mc.p.thermalization

    sweep_dur = Observable(Float64, "Sweep duration"; alloc=ceil(Int, total_sweeps/100))

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    tic()
    for i in 1:total_sweeps
        sweep(mc)

        if mc.p.global_moves && mod(i, mc.p.global_rate) == 0
            mc.a.prop_global += 1
            mc.a.acc_global += global_move(mc, mc.model, mc.conf, mc.energy)
        end

        (i > mc.p.thermalization) && measure_observables!(mc, mc.model, mc.obs, mc.conf, mc.energy)

        if mod(i, 1000) == 0
            mc.a.acc_rate = mc.a.acc_rate / 1000
            mc.a.acc_rate_global = mc.a.acc_rate_global / (1000 / mc.p.global_rate)
            add!(sweep_dur, toq()/1000)
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", sweep_dur[end])
                @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
                if mc.p.global_moves
                  @printf("\t\tacc rate (global): %.1f%%\n", mc.a.acc_rate_global*100)
                  @printf("\t\tacc rate (global, overall): %.1f%%\n", mc.a.acc_global/mc.a.prop_global*100)
                end
            end

            mc.a.acc_rate = 0.0
            mc.a.acc_rate_global = 0.0
            flush(STDOUT)
            tic()
        end
    end
    finish_observables!(mc, mc.model, mc.obs)
    toq();

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global
    mc.a.sweep_dur = mean(sweep_dur)

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)

    mc.obs
end

"""
    sweep(mc::MC)

Performs a sweep of local moves.
"""
function sweep(mc::MC)
    const N = mc.model.l.sites
    const beta = mc.p.β

    @inbounds for i in eachindex(mc.conf)
        ΔE, Δi = propose_local(mc, mc.model, i, mc.conf, mc.energy)
        mc.a.prop_local += 1
        # Metropolis
        if ΔE <= 0 || rand() < exp(- beta*ΔE)
            accept_local!(mc, mc.model, i, mc.conf, mc.energy, Δi, ΔE)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy += ΔE
        end
    end

    nothing
end

include("interface_mandatory.jl")
include("interface_optional.jl")
