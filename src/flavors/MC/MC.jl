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
@with_kw struct MCParameters
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

    thermalization_measurements::Dict{Symbol, AbstractMeasurement}
    measurements::Dict{Symbol, AbstractMeasurement}
    p::MCParameters
    a::MCAnalysis

    MC{M,C}() where {M,C} = new()
end


"""
    MC(m::M; kwargs...) where M<:Model

Create a Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function MC(m::M;
        seed::Int=-1,
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = :default,
        kwargs...
    ) where M<:Model

    conf = rand(MC, m)
    mc = MC{M, typeof(conf)}()
    mc.model = m
    kwdict = Dict(kwargs)
    if :T in keys(kwargs)
        kwdict[:beta] = 1/kwargs[:T]
        delete!(kwdict, :T)
    end
    mc.p = MCParameters(; kwdict...)

    init!(
        mc, seed = seed, conf = conf,
        thermalization_measurements = thermalization_measurements,
        measurements = measurements
    )
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
Base.summary(mc::MC) = "MC simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::MC)
    print(io, "Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", round(beta(mc), sigdigits=3), " (T ≈ $(round(1/beta(mc), sigdigits=3)))\n")
    N_th_meas = length(mc.thermalization_measurements)
    N_me_meas = length(mc.measurements)
    print(io, "Measurements: ", N_th_meas + N_me_meas, " ($N_th_meas + $N_me_meas)")
end
Base.show(io::IO, m::MIME"text/plain", mc::MC) = print(io, mc)





# implement MonteCarloFlavor interface
"""
    init!(mc::MC[; seed::Real=-1])

Initialize the Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `Random.seed!(seed)`.
"""
function init!(mc::MC;
        seed::Real=-1,
        conf=rand(MC, mc.model),
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = :default
    )
    seed == -1 || Random.seed!(seed)
    mc.conf = conf
    mc.a = MCAnalysis()

    mc.thermalization_measurements = thermalization_measurements
    if measurements isa Dict{Symbol, AbstractMeasurement}
        mc.measurements = measurements
    elseif measurements == :default
        mc.measurements = default_measurements(mc, mc.model)
    else
        @warn(
            "`measurements` should be of type Dict{Symbol, AbstractMeasurement}, but are " *
            "$(typeof(measurements)). No measurements have been set."
        )
        mc.measurements = Dict{Symbol, AbstractMeasurement}()
    end

    nothing
end


"""
    run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given Monte Carlo simulation `mc`.
Progress will be printed to `stdout` if `verbose=true` (default).
"""
function run!(mc::MC; verbose::Bool=true, sweeps::Int=mc.p.sweeps,
        thermalization=mc.p.thermalization)

    do_th_measurements = !isempty(mc.thermalization_measurements)
    do_me_measurements = !isempty(mc.measurements)
    !do_me_measurements && @warn(
        "There are no measurements set up for this simulation!"
    )
    total_sweeps = sweeps + thermalization

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    _time = time()
    do_th_measurements && prepare!(mc.thermalization_measurements, mc, mc.model)
    for i in 1:total_sweeps
        sweep(mc)

        if mc.p.global_moves && mod(i, mc.p.global_rate) == 0
            mc.a.prop_global += 1
            mc.a.acc_global += global_move(mc, mc.model, conf(mc))
        end

        # For optimal performance whatever is most likely to fail should be
        # checked first.
        if i <= thermalization && iszero(mod(i, mc.p.measure_rate)) && do_th_measurements
            measure!(mc.thermalization_measurements, mc, mc.model, i)
        end
        if (i == thermalization+1)
            do_th_measurements && finish!(mc.thermalization_measurements, mc, mc.model)
            do_me_measurements && prepare!(mc.measurements, mc, mc.model)
        end
        if i > thermalization && iszero(mod(i, mc.p.measure_rate)) && do_me_measurements
            measure!(mc.measurements, mc, mc.model, i)
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
    do_me_measurements && finish!(mc.measurements, mc, mc.model)

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
