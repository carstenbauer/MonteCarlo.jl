"""
Statistical data of Monte Carlo integration
"""
mutable struct IntegratorAnalysis
    acceptance_rate::Float64
    proposed::Int
    accepted::Int
    sweep_duration::Float64

    IntegratorAnalysis() = new(0., 0, 0, 0.)
end

"""
Parameters of Monte Carlo integration
"""
mutable struct IntegratorParameters
    thermalization::Int # number of thermalization sweeps
    sweeps::Int # number of sweeps (after thermalization)
    min_x::Vector{Float64} # integration bounds
    max_x::Vector{Float64}

    IntegratorParameters() = new()
end

"""
Monte Carlo integration
"""
mutable struct Integrator{M<:Model} <: MonteCarloFlavor
    model::M
    value::Vector{Float64}
    energy::Float64

    obs::Dict{String, Observable}
    p::IntegratorParameters
    a::IntegratorAnalysis

    Integrator{M}() where {M} = new()
end

"""
    Integrator(m::M; kwargs...) where M<:Model

Create a Monte Carlo integrator for model `m` with keyword parameters `kwargs`.
"""
function Integrator(m::M; sweeps::Int=1000, thermalization::Int=0, seed::Int=-1,
                    min_x::Vector{Float64}=[-10.], max_x::Vector{Float64} = [10.]) where M<:Model
    mc = Integrator{M}()
    mc.model = m

    # default params
    mc.p = IntegratorParameters()
    mc.p.thermalization = thermalization
    mc.p.sweeps = sweeps
    mc.p.min_x = min_x
    mc.p.max_x = max_x
    init!(mc, seed=seed)

    return mc
end

"""
    Integrator(m::M; kwargs::Dict{String, Any})

Create a Monte Carlo integrator for model `m` with (keyword) parameters
as specified in the dictionary `kwargs`.
"""
function Integrator(m::M, kwargs::Dict{String, Any}) where M<:Model
    Integrator(m; convert(Dict{Symbol, Any}, kwargs)...)
end


"""
    init!(mc::MC[; seed::Real=-1])

Initialize the Monte Carlo integrator `mc`.
If `seed !=- 1` the random generator will be initialized with `srand(seed)`.
"""
function init!(mc::Integrator; seed::Real=-1)
    seed == -1 || srand(seed)

    mc.value = rand(mc, mc.model)
    mc.energy = energy(mc, mc.model, mc.value)
    mc.obs = prepare_observables(mc, mc.model)
    mc.a = IntegratorAnalysis()
    nothing
end

"""
    run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given classical Monte Carlo simulation `mc`.
Progress will be printed to `STDOUT` if `verborse=true` (default).
"""
function run!(mc::Integrator; verbose::Bool=true, sweeps::Int=mc.p.sweeps, thermalization=mc.p.thermalization)
    mc.p.sweeps = sweeps
    mc.p.thermalization = thermalization
    const total_sweeps = mc.p.sweeps + mc.p.thermalization

    sweep_duration = Observable(Float64, "Sweep duration"; alloc=ceil(Int, total_sweeps/100))

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    tic()
    for i in 1:total_sweeps

        if i == mc.p.thermalization mc.a.proposed, mc.a.accepted = 0, 0 end

        sweep(mc)

        (i > mc.p.thermalization) && measure_observables!(mc, mc.model, mc.obs, mc.energy)

        if mod(i, 1000) == 0
            add!(sweep_duration, toq()/1000)
            if verbose
                println("\t", i)
                @printf("\t\tsweep duration: %.3fs\n", sweep_duration[end])
                @printf("\t\tacceptance rate: %.1f%%\n", mc.a.acceptance_rate * 100)
            end

            flush(STDOUT)
            tic()
        end
        mc.a.acceptance_rate = mc.a.accepted / mc.a.proposed

    end
    finish_observables!(mc, mc.model, mc.obs)
    toq();

    mc.a.sweep_duration = mean(sweep_duration)

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes\n", (end_time - start_time).value/1000./60.)
    mc.obs
end

"""
    sweep(mc::Integrator)

Performs a sweep of local moves.
"""
function sweep(mc::Integrator)
    proposed_value, r = propose(mc, mc.model, mc.value, mc.energy)
    mc.a.proposed += 1

    # Metropolis
    if rand() < r
        mc.value = proposed_value
        mc.energy *= r
        mc.a.accepted += 1
    end
    nothing
end

include("Integrator_mandatory.jl")
include("Integrator_optional.jl")

# cosmetics
import Base.summary
import Base.show
Base.summary(mc::Integrator) = "MC integration of $(summary(mc.model))"
function Base.show(io::IO, mc::Integrator)
    print(io, "Monte Carlo integration\n")
    print(io, "Function: ", mc.model, "\n")
    print(io, "Lower bounds: ", mc.p.min_x, ", Upper bounds: ", mc.p.max_x)
end
Base.show(io::IO, m::MIME"text/plain", mc::Integrator) = print(io, mc)
