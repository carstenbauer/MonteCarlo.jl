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
mutable struct MC{M<:Model, C, RT<:AbstractRecorder} <: MonteCarloFlavor
    model::M
    conf::C
    configs::RT
    last_sweep::Int

    thermalization_measurements::Dict{Symbol, AbstractMeasurement}
    measurements::Dict{Symbol, AbstractMeasurement}
    p::MCParameters
    a::MCAnalysis

    MC{M,C,RT}() where {M,C,RT} = new()
    function MC(m::M, c::C, cs::RT, ls, tm, mm, p, a) where {M, C, RT}
        new{M, C, RT}(m, c, cs, ls, tm, mm, p, a)
    end
end


"""
    MC(m::M; kwargs...) where M<:Model

Create a Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function MC(m::M;
        seed::Int=-1,
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = :default,
        last_sweep = 0,
        measure_rate = 1,
        recording_rate = measure_rate,
        recorder = Discarder(MC, M, recording_rate),
        kwargs...
    ) where M<:Model

    conf = rand(MC, m)
    mc = MC{M, typeof(conf), typeof(recorder)}()
    mc.conf = conf
    mc.model = m
    kwdict = Dict(kwargs)
    if :T in keys(kwargs)
        kwdict[:beta] = 1/kwargs[:T]
        delete!(kwdict, :T)
    end
    mc.p = MCParameters(measure_rate = measure_rate; kwdict...)
    mc.last_sweep = last_sweep
    mc.configs = recorder

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
@inline last_sweep(mc::MC) = mc.last_sweep
@inline configurations(mc::MC) = mc.configs
@inline lattice(mc::MC) = lattice(mc.model)
@inline parameters(mc::MC) = merge(parameters(mc.p), parameters(mc.model))
@inline function parameters(p::MCParameters)
    (beta = p.beta, thermalization = p.thermalization, sweeps = p.sweeps)
end


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

function ConfigRecorder(mc::MC, model::Model, rate = 10)
    ConfigRecorder{typeof(compress(mc, model, conf(mc)))}(rate)
end




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
            "`measurements` should be of type Dict{Symbol, AbstractMeasurement}" *
            ", but are $(typeof(measurements)). No measurements have been set."
        )
        mc.measurements = Dict{Symbol, AbstractMeasurement}()
    end
    init!(mc, mc.model)

    nothing
end

function resume_init!(mc::MC)
    mc.a = MCAnalysis()
    init!(mc, mc.model)
    nothing
end


"""
    run!(mc::MC[; kwargs...])

Runs the given Monte Carlo simulation `mc`. Returns `SUCCESS::ExitCode = 0` if 
the simulation finished normally or various other codes if failed or cancelled. 
See [`ExitCode`](@ref).


### Keyword Arguments:
- `verbose = true`: If true, print progress messaged to stdout.
- `thermalization`: Number of thermalization sweeps. Uses the value passed to
`DQMC` by default.
- `sweeps`: Number of measurement sweeps. Uses the value passed to `DQMC` by
default.
- `safe_every::TimePeriod`: Set the interval for regularly scheduled saves.
- `safe_before::Date`: If this date is passed, `run!` will generate a resumable
save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `resumable_filename`: Name of the resumable save file. The default is based on
`safe_before`.
- `overwrite = false`: If set to true a file with the same name as
`resumable_filename` will be overwritten. (This will create a temporary backup)

See also: [`resume!`](@ref)
"""
@bm function run!(
        mc::MC;
        verbose::Bool = true,
        sweeps::Int = mc.p.sweeps,
        thermalization = mc.p.thermalization,
        safe_before::TimeType = now() + Year(100),
        safe_every::TimePeriod = Hour(10000),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_$(Dates.format(safe_before, "d_u_yyyy-HH_MM")).jld",
        overwrite = false
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
    last_checkpoint = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    _time = time()
    do_th_measurements && prepare!(mc.thermalization_measurements, mc, mc.model)
    for i in mc.last_sweep+1:total_sweeps
        sweep(mc)
        mc.last_sweep = i

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
        (i > thermalization) && push!(mc.configs, mc, mc.model, i)


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
                    @printf(
                        "\t\tacc rate (global, overall): %.1f%%\n", 
                        mc.a.acc_global/mc.a.prop_global*100
                    )
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
            resumable_filename = save(resumable_filename, mc, overwrite=overwrite)
            verbose && println("\nEarly save finished")

            return CANCELLED_TIME_LIMIT
        elseif (now() - last_checkpoint) > safe_every
            verbose && println("Performing scheduled save.")
            last_checkpoint = now()
            save(resumable_filename, mc, overwrite = overwrite, rename = false)
        end
    end
    do_me_measurements && finish!(mc.measurements, mc, mc.model)

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    end_time = now()
    if verbose
        println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
        @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)
        println()
    end

    return SUCCESS
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
        ΔE, passthrough = propose_local(mc, m, i, c)
        mc.a.prop_local += 1
        # Metropolis
        if ΔE <= 0 || rand() < exp(- β*ΔE)
            accept_local!(mc, m, i, c, ΔE, passthrough)
            mc.a.acc_rate += 1.0
            mc.a.acc_local += 1
        end
    end
    mc.a.acc_rate /= length(c)
    nothing
end



"""
    replay(mc::MC[, configurations::Iterable = mc.configs; kwargs...])

Replays previously generated configurations and measures observables along the
way.

### Keyword Arguments:
- `verbose = true`: If true, print progress messaged to stdout.
- `safe_every::TimePeriod`: Set the interval for regularly scheduled saves.
- `safe_before::Date`: If this date is passed, `replay!` will generate a
resumable save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `filename`: Name of the save file. The default is based on `safe_before`.
- `measure_rate = 1`: Rate at which measurements are taken. Note that this is 
based on the recorded configurations, not actual sweeps.
"""
function replay!(
        mc::MC, configurations = mc.configs;
        verbose::Bool=true,
        safe_before::TimeType = now() + Year(100),
        safe_every::TimePeriod = Hour(10000),
        grace_period::TimePeriod = Minute(5),
        filename::String = "resumable_$(Dates.format(safe_before, "d_u_yyyy-HH_MM")).jld",
        overwrite = false,
        measure_rate = 1
    )
    if isempty(configurations)
        println("Nothin to replay (configurations empty). Exiting")    
        return GENERIC_FAILURE
    end

    start_time = now()
    last_checkpoint = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # Check for measurements
    !isempty(mc.thermalization_measurements) && @warn(
        "There is no thermalization process in a replayed simulation."
    )
    isempty(mc.measurements) && @warn(
        "There are no measurements set up for this simulation!"
    )

    mc.p = MCParameters(
        global_moves = mc.p.global_moves,
        global_rate = mc.p.global_rate,
        thermalization = mc.p.thermalization,
        sweeps = mc.p.sweeps,
        measure_rate = measure_rate,
        print_rate = mc.p.print_rate,
        beta = mc.p.beta
    )
    mc.conf = copy(decompress(mc, mc.model, configurations[1]))
    resume_init!(mc)

    _time = time()
    verbose && println("\n\nReplaying measurement stage - ", length(configurations))
    prepare!(mc.measurements, mc, mc.model)
    for i in mc.last_sweep+1:mc.p.measure_rate:length(configurations)
        mc.conf .= decompress(mc, mc.model, configurations[i])
        mc.last_sweep = i

        energy(mc, mc.model, mc.conf)
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
            filename = save(filename, mc, overwrite = overwrite)
            verbose && println("\nEarly save finished")

            return CANCELLED_TIME_LIMIT
        elseif (now() - last_checkpoint) > safe_every
            verbose && println("Performing scheduled save.")
            last_checkpoint = now()
            save(resumable_filename, mc, overwrite = overwrite, rename = false)
        end
    end
    finish!(mc.measurements, mc, mc.model)

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)

    return SUCCESS
end


function _save(file::FileLike, entryname::String, mc::MC)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "MC")
    write(file, entryname * "/type", typeof(mc))
    _save(file, entryname * "/parameters", mc.p)
    write(file, entryname * "/conf", mc.conf)
    _save(file, entryname * "/configs", mc.configs)
    write(file, entryname * "/last_sweep", mc.last_sweep)
    save_measurements(file, entryname * "/Measurements", mc)
    _save(file, entryname * "/Model", mc.model)
    nothing
end

function _load(data, ::Val{:MC})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end
    p = _load(data["parameters"], Val(:MCParameters))
    conf = data["conf"]
    configs = _load(data["configs"], to_tag(data["configs"]))
    last_sweep = data["last_sweep"]
    model = load_model(data["Model"], to_tag(data["Model"]))
    measurements = _load(data["Measurements"], Val(:Measurements))
    
    MC(
        model, conf, configs, last_sweep,
        measurements[:TH], measurements[:ME],
        p, MCAnalysis()
    )
end

function _save(file::FileLike, entryname::String, p::MCParameters)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "MCParameters")

    write(file, entryname * "/global_moves", Int(p.global_moves))
    write(file, entryname * "/global_rate", p.global_rate)
    write(file, entryname * "/thermalization", p.thermalization)
    write(file, entryname * "/sweeps", p.sweeps)
    write(file, entryname * "/measure_rate", p.measure_rate)
    write(file, entryname * "/print_rate", p.print_rate)
    write(file, entryname * "/beta", p.beta)

    nothing
end

function _load(data, ::Val{:MCParameters})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    MCParameters(
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
