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
end

"""
Parameters of Monte Carlo
"""
@with_kw mutable struct MCParameters
    global_moves::Bool = false
    global_rate::Int = 5
    thermalization::Int = 0 # number of thermalization sweeps
    sweeps::Int = 1000 # number of sweeps (after thermalization)
    measure_rate::Int = 1 # measure at every nth sweep (after thermalization)
    print_rate::Int = 1000

    beta::Float64
end

"""
Monte Carlo simulation
"""
mutable struct MC{M<:Model, C} <: MonteCarloFlavor
    model::M
    conf::C

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
    conf = rand(MC, m)
    mc = MC{M, typeof(conf)}()
    mc.model = m

    kwdict = Dict(kwargs)
    if :T in keys(kwargs)
        kwdict[:beta] = 1/kwargs[:T]
        delete!(kwdict, :T)
    end
    mc.p = MCParameters(; kwdict...)
    init!(mc, seed=seed, conf=conf)
    return mc
end

"""
    MC(m::M, params::Dict)
    MC(m::M, params::NamedTuple)

Create a Monte Carlo simulation for model `m` with (keyword) parameters
as specified in the dictionary/named tuple `params`.
"""
MC(m::Model, params::Dict{Symbol, T}) where T = MC(m; params...)
MC(m::Model, params::NamedTuple) = MC(m; params...)

# convenience
@inline beta(mc::MC) = mc.p.beta
@inline model(mc::MC) = mc.model
@inline conf(mc::MC) = mc.conf

# cosmetics
import Base.summary
import Base.show
Base.summary(mc::MC) = "MC simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::MC)
    print(io, "Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", round(beta(mc), sigdigits=3), " (T ≈ $(round(1/beta(mc), sigdigits=3)))")
end
Base.show(io::IO, m::MIME"text/plain", mc::MC) = print(io, mc)





# implement MonteCarloFlavor interface
"""
    init!(mc::MC[; seed::Real=-1])

Initialize the Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `Random.seed!(seed)`.
"""
function init!(mc::MC; seed::Real=-1, conf=rand(MC, mc.model))
    seed == -1 || Random.seed!(seed)
    mc.conf = conf
    mc.obs = prepare_observables(mc, mc.model)
    mc.a = MCAnalysis()
    nothing
end


"""
    run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given Monte Carlo simulation `mc`.
Progress will be printed to `stdout` if `verbose=true` (default).
"""
function run!(mc::MC; verbose::Bool=true, sweeps::Int=mc.p.sweeps, thermalization=mc.p.thermalization)
    @pack! mc.p = sweeps, thermalization
    total_sweeps = mc.p.sweeps + mc.p.thermalization

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    _time = time()
    for i in 1:total_sweeps
        sweep(mc)

        if mc.p.global_moves && mod(i, mc.p.global_rate) == 0
            mc.a.prop_global += 1
            mc.a.acc_global += global_move(mc, mc.model, conf(mc))
        end

        if i > mc.p.thermalization && iszero(mod(i, mc.p.measure_rate))
            measure_observables!(mc, mc.model, mc.obs, conf(mc))
        end

        print_rate = mc.p.print_rate
        if print_rate != 0 && iszero(mod(i, print_rate))
            mc.a.acc_rate /= print_rate
            mc.a.acc_rate_global /= print_rate / mc.p.global_rate
            sweep_dur = (time() - _time)/print_rate
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", sweep_dur)
                @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
                if mc.p.global_moves
                  @printf("\t\tacc rate (global): %.1f%%\n", mc.a.acc_rate_global*100)
                  @printf("\t\tacc rate (global, overall): %.1f%%\n", mc.a.acc_global/mc.a.prop_global*100)
                end
            end

            mc.a.acc_rate = 0.0
            mc.a.acc_rate_global = 0.0
            flush(stdout)
            _time = time()
        end
    end
    finish_observables!(mc, mc.model, mc.obs)

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    nothing
end

"""
    sweep(mc::MC)

Performs a sweep of local moves.
"""
@inline function sweep(mc::MC)
    c = conf(mc)
    m = model(mc)
    β = beta(mc)
    @inbounds for i in eachindex(c)
        ΔE, Δsite = propose_local(mc, m, i, c)
        mc.a.prop_local += 1
        # Metropolis
        if ΔE <= 0 || rand() < exp(- β*ΔE)
            accept_local!(mc, m, i, c, Δsite, ΔE)
            mc.a.acc_rate += 1/nsites(m)
            mc.a.acc_local += 1
        end
    end

    nothing
end

include("MC_mandatory.jl")
include("MC_optional.jl")
