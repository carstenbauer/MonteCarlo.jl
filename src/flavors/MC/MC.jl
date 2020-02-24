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

function resume_init!(mc::MC)
    mc.a = MCAnalysis()
    nothing
end


"""
    run!(mc::MC[; kwargs...])

Runs the given Monte Carlo simulation `mc`. Returns true if the run finished and
false if it cancelled early to generate a resumable save-file.


### Keyword Arguments:
- `verbose = true`: If true, print progress messaged to stdout.
- `thermalization`: Number of thermalization sweeps. Uses the value passed to
`DQMC` by default.
- `sweeps`: Number of measurement sweeps. Uses the value passed to `DQMC` by
default.
- `safe_before::Date`: If this date is passed, `run!` will generate a resumable
save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `resumable_filename`: Name of the resumable save file. The default is based on
`safe_before`.
- `force_overwrite = false`: If set to true a file with the same name as
`resumable_filename` will be overwritten. (This will create a temporary backup)
- `start=1`: The first sweep in the simulation. This will be changed when using
`resume!(save_file)`.

See also: [`resume!`](@ref)
"""
@bm function run!(
        mc::MC;
        verbose::Bool = true,
        sweeps::Int = mc.p.sweeps,
        thermalization = mc.p.thermalization,
        safe_before::TimeType = now() + Year(100),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_" * Dates.format(safe_before, "d_u_yyyy-HH_MM") * ".jld",
        force_overwrite = false,
        start = 1
    )

    do_th_measurements = !isempty(mc.thermalization_measurements)
    do_me_measurements = !isempty(mc.measurements)
    !do_me_measurements && @warn(
        "There are no measurements set up for this simulation!"
    )

    # Update number of sweeps
    if (mc.p.thermalization != thermalization) || (mc.p.sweeps != sweeps)
        verbose && println("Rebuilding DQMCParameters with new number of sweeps.")
        p = MCParameters(
            mc.p.global_moves,
            mc.p.global_rate,
            thermalization,
            sweeps,
            mc.p.measure_rate,
            mc.p.print_rate,
            mc.p.beta
        )
        mc.p = p
    end
    total_sweeps = sweeps + thermalization

    start_time = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    _time = time()
    do_th_measurements && prepare!(mc.thermalization_measurements, mc, mc.model)
    for i in start:total_sweeps
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
            max_sweep_duration = max(max_sweep_duration, sweep_dur)
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

        if safe_before - now() < Millisecond(grace_period) +
                Millisecond(round(Int, 2e3max_sweep_duration))

            println("Early save initiated for sweep #$i.\n")
            verbose && println("Current time: ", Dates.format(now(), "d.u yyyy HH:MM"))
            verbose && println("Target time:  ", Dates.format(safe_before, "d.u yyyy HH:MM"))

            if force_overwrite
                parts = splitpath(resumable_filename)
                parts[end] = "." * parts[end]
                temp_filename = _generate_unqiue_JLD_filename(joinpath(parts...))
                mv(resumable_filename, temp_filename)
            end

            # We create a backup manually here because we save extra stuff
            # In either case there should be no conflicting file, so there
            # should be nothing to overwrite.
            resumable_filename = save(resumable_filename, mc)
            save_rng(resumable_filename)
            jldopen(resumable_filename, "r+") do f
                write(f, "last_sweep", i)
            end

            if force_overwrite
                rm(temp_filename)
            end

            verbose && println("\nEarly save finished")

            return false
        end
    end
    do_me_measurements && finish!(mc.measurements, mc, mc.model)

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    return true
end

"""
    sweep(mc::MC)

Performs a sweep of local moves.
"""
@inline @bm function sweep(mc::MC)
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



"""
    replay(
        mc::MC[;
        configs::ConfigurationMeasurement = mc.measurements[:conf],
        reset_measurements = true,
        measure_rate = 1,
        kwargs...]
    )
    replay(mc::DQMC, configs::AbstractArray[; kwargs...])

Replays previously generated configurations and measures observables along the
way.

By default, the first method will search `mc` for a ConfigurationMeasurement,
remove it from the active measurements and replay it. If
`reset_measurements = true` it will also reset every active measurement before
replaying configurations.
The second method replays configurations directly, i.e. it does not modify any
measurements beforehand.

### Keyword Arguments (both):
- `verbose = true`: If true, print progress messaged to stdout.
- `safe_before::Date`: If this date is passed, `replay!` will generate a
resumable save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `filename`: Name of the save file. The default is based on `safe_before`.
- `start=1`: The first sweep in the simulation. This will be changed when using
`resume!(save_file)`.
"""
function replay!(
        mc::MC;
        configs::ConfigurationMeasurement = let
            for (k, v) in mc.measurements
                v isa ConfigurationMeasurement && return v
            end
            throw(ArgumentError(
                "Could not find a `ConfigurationMeasurement` in the given " *
                "mc::MC. Try supplying it manually."
            ))
        end,
        reset_measurements = true,
        measure_rate = 1,
        kwargs...
    )
    delete!(mc, ConfigurationMeasurement)
    reset_measurements && for (k, v) in mc.measurements
        mc.measurements[k] = typeof(v)(mc, mc.model)
    end
    mc.p.measure_rate = measure_rate
    replay(mc, timeseries(configs.obs); kwargs...)
end

function replay!(
            mc::MC,
            configs::AbstractArray;
            verbose::Bool=true,
            safe_before::TimeType = now() + Year(100),
            grace_period::TimePeriod = Minute(5),
            filename::String = "resumable_" * Dates.format(safe_before, "d_u_yyyy-HH_MM") * ".jld",
            force_overwrite = false,
            start = 1
        )

    # Check for measurements
    !isempty(mc.thermalization_measurements) && @warn(
        "There is no thermalization process in a replayed simulation."
    )
    !isempty(mc.measurements) && @warn(
        "There are no measurements set up for this simulation!"
    )

    start_time = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    mc.a = MCAnalysis()

    _time = time()
    verbose && println("\n\nReplaying measurement stage - ", length(configs))
    prepare!(mc.measurements, mc, mc.model)
    for i in start:mc.p.measure_rate:length(configs)
        mc.config = configs[i]
        # maybe calculate total Energy?
        measure!(mc.measurements, mc, mc.model, i)

        if mc.p.print_rate != 0 && iszero(mod(i, mc.p.print_rate))
            sweep_dur = (time() - _time)/print_rate
            max_sweep_duration = max(max_sweep_duration, sweep_dur)
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", sweep_dur)
            end
            flush(stdout)
            _time = time()
        end

        if safe_before - now() < Millisecond(grace_period) +
                Millisecond(round(Int, 2e3max_sweep_duration))

            println("Early save initiated for sweep #$i.\n")
            verbose && println("Current time: ", Dates.format(now(), "d.u yyyy HH:MM"))
            verbose && println("Target time:  ", Dates.format(safe_before, "d.u yyyy HH:MM"))
            filename = save(filename, mc, force_overwrite = force_overwrite)
            save_rng(filename)
            jldopen(filename, "r+") do f
                write(f, "last_sweep", i)
            end
            verbose && println("\nEarly save finished")

            return false
        end
    end
    finish!(mc.measurements, mc, mc.model)

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    return true
end


#     save_mc(filename, mc, entryname)
#
# Saves (minimal) information necessary to reconstruct a given `mc::MC` to a
# JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC`
function save_mc(file::JLD.JldFile, mc::MC, entryname::String="MC")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(mc))
    save_parameters(file, mc.p, entryname * "/parameters")
    write(file, entryname * "/conf", mc.conf)
    save_measurements(file, mc, entryname * "/Measurements")
    save_model(file, mc.model, entryname * "/Model")
    nothing
end

#     load_mc(data, ::Type{<: MC})
#
# Loads a MC from a given `data` dictionary produced by `JLD.load(filename)`.
function load_mc(data, ::Type{T}) where {T <: MC}
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end
    mc = data["type"]()
    mc.p = load_parameters(data["parameters"], data["parameters"]["type"])
    mc.conf = data["conf"]
    mc.model = load_model(data["Model"], data["Model"]["type"])

    measurements = load_measurements(data["Measurements"])
    mc.thermalization_measurements = measurements[:TH]
    mc.measurements = measurements[:ME]
    mc
end


#   save_parameters(file::JLD.JldFile, p::MCParameters, entryname="Parameters")
#
# Saves (minimal) information necessary to reconstruct a given
# `p::MCParameters` to a JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC/Parameters`
function save_parameters(file::JLD.JldFile, p::MCParameters, entryname::String="Parameters")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(p))

    write(file, entryname * "/global_moves", Int(p.global_moves))
    write(file, entryname * "/global_rate", p.global_rate)
    write(file, entryname * "/thermalization", p.thermalization)
    write(file, entryname * "/sweeps", p.sweeps)
    write(file, entryname * "/measure_rate", p.measure_rate)
    write(file, entryname * "/print_rate", p.print_rate)
    write(file, entryname * "/beta", p.beta)

    nothing
end

#     load_parameters(data, ::Type{<: MCParameters})
#
# Loads a MCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function load_parameters(data::Dict, ::Type{T}) where T <: MCParameters
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    data["type"](
        Bool(data["global_moves"]),
        data["global_rate"],
        data["thermalization"],
        data["sweeps"],
        data["measure_rate"],
        data["print_rate"],
        data["beta"]
    )
end


include("MC_mandatory.jl")
include("MC_optional.jl")
