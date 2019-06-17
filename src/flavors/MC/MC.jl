"""
Analysis data of Monte Carlo simulation
"""
@with_kw mutable struct MCAnalysis
    acc_rate::Float64 = 0.
    prop_local::Int = 0
    acc_local::Int = 0
    acc_rate_global::Float64 = 0.
    prop_global::Int = 0
    acc_global::Int = 0
    sweep_dur::Float64 = 0.
end

"""
Parameters of Monte Carlo
"""
@with_kw mutable struct MCParameters
    global_moves::Bool = false
    global_rate::Int = 5
    thermalization::Int = 0 # number of thermalization sweeps
    sweeps::Int = 1000 # number of sweeps (after thermalization)

    beta::Float64
end

"""
Monte Carlo simulation
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

Create a Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function MC(m::M; seed::Int=-1, kwargs...) where M<:Model
    mc = MC{M, conftype(MC, m)}()
    mc.model = m
    mc.p = MCParameters(; kwargs...) # forward kwargs to MCParameters
    init!(mc, seed=seed)
    return mc
end

"""
    MC(m::M; kwargs::Dict{String, Any})

Create a Monte Carlo simulation for model `m` with (keyword) parameters
as specified in the dictionary `kwargs`.
"""
MC(m::M, kwargs::Union{Dict{String, Any}, Dict{Symbol, Any}}) where M<:Model =
    MC(m; convert(Dict{Symbol, Any}, kwargs)...)

# cosmetics
import Base.summary
import Base.show
Base.summary(mc::MC) = "MC simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::MC)
    print(io, "Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, 3)))")
end
Base.show(io::IO, m::MIME"text/plain", mc::MC) = print(io, mc)

"""
    init!(mc::MC[; seed::Real=-1])

Initialize the Monte Carlo simulation `mc`.
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

Runs the given Monte Carlo simulation `mc`.
Progress will be printed to `STDOUT` if `verbose=true` (default).
"""
function run!(mc::MC; verbose::Bool=true, sweeps::Int=mc.p.sweeps, thermalization=mc.p.thermalization)
    @pack! mc.p = sweeps, thermalization
    total_sweeps = mc.p.sweeps + mc.p.thermalization

    sweep_dur = Observable(Float64, "Sweep duration"; alloc=ceil(Int, total_sweeps/1000))

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    _time = time()
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
            add!(sweep_dur, (time() - _time)/1000)
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
            _time = time()
        end
    end
    finish_observables!(mc, mc.model, mc.obs)

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global
    mc.a.sweep_dur = mean(sweep_dur)

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    nothing
end

"""
    sweep(mc::MC)

Performs a sweep of local moves.
"""
function sweep(mc::MC)
    N = mc.model.l.sites
    beta = mc.p.beta

    @inbounds for i in eachindex(mc.conf)
        delta_E, delta_i = propose_local(mc, mc.model, i, mc.conf, mc.energy)
        mc.a.prop_local += 1
        # Metropolis
        if delta_E <= 0 || rand() < exp(- beta*delta_E)
            accept_local!(mc, mc.model, i, mc.conf, mc.energy, delta_i, delta_E)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy += delta_E
        end
    end

    nothing
end

include("MC_mandatory.jl")
include("MC_optional.jl")
