"""
    DQMC(m::M; kwargs...) where M<:Model

Create a determinant quantum Monte Carlo simulation for model `m` with
keyword parameters `kwargs`.

### Keyword Arguments:
- `seed`: The random seed used by the simulation.
- `checkerboard=false`: If true, the simulation uses a generic checkerboard
decomposition.
- `thermalization_measurements::Dict{Symbol, AbstractMeasurement}`: A collection
of measurements run during the thermalization stage. By default, none are used.
- `measurements::Dict{Symbol, AbstractMeasurement}`: A collection of measurements
run during the measurement stage. Calls `default_measurements` if not specified.
- `recorder = ConfigRecorder`: Type of recorder used for saving configurations
generated during the simulation. Used (by default) when `replay!`ing simulations.
 (`Discarder` or `ConfigRecorder`)
- `recording_rate = measure_rate`: Rate at which configurations are recorded.
- `thermalization = 100`: Number of thermalization sweeps
- `sweeps`: Number of measurement sweeps
- `all_checks = true`: Check for Propagation instabilities and sign problems.
- `safe_mult = 10`: Number of "safe" matrix multiplications. Every `safe_mult`
multiplications, a UDT decomposition is used to stabilize the product.
- `delta_tau = 0.1`: Time discretization of the path integral
- `beta::Float64`: Inverse temperature used in the simulation
- `slices::Int = beta / delta_tau`: Number of imaginary time slice in the
simulation
- `measure_rate = 10`: Number of sweeps discarded between every measurement.
- `global_moves = false`:: Currently not used
- `global_rate = 5`: Currently not used
- `last_sweep = 0`: Sets the index of the last finished sweep. The simulation
will start with sweep `last_sweep + 1`.
"""
function DQMC(m::M;
        seed::Int=-1,
        checkerboard::Bool=false,
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = Dict{Symbol, AbstractMeasurement}(),
        last_sweep = 0,
        measure_rate = 10,
        recorder = ConfigRecorder,
        recording_rate = measure_rate,
        kwargs...
    ) where M<:Model
    # default params
    # paramskwargs = filter(kw->kw[1] in fieldnames(DQMCParameters), kwargs)
    p = DQMCParameters(measure_rate = measure_rate; kwargs...)

    HET = hoppingeltype(DQMC, m)
    GET = greenseltype(DQMC, m)
    HMT = hopping_matrix_type(DQMC, m)
    GMT = greens_matrix_type(DQMC, m)
    IMT = interaction_matrix_type(DQMC, m)
    stack = DQMCStack{GET, HET, GMT, HMT, IMT}()
    ut_stack = UnequalTimeStack{GET, GMT}()

    conf = rand(DQMC, m, p.slices)
    analysis = DQMCAnalysis()
    CB = checkerboard ? CheckerboardTrue : CheckerboardFalse

    mc = DQMC{M, CB, typeof(conf), recorder, DQMCStack, AbstractDQMCStack}(
        model = m, conf = conf, last_sweep = last_sweep, s = stack,
        ut_stack = ut_stack, p = p, a = analysis,
        thermalization_measurements = thermalization_measurements,
        measurements = measurements
    )

    mc.configs = recorder(mc, m, recording_rate)
    seed == -1 || Random.seed!(seed)
    mc.conf = conf
    init!(mc)
    return make_concrete!(mc)
end


"""
    DQMC(m::M, params::Dict)
    DQMC(m::M, params::NamedTuple)

Create a determinant quantum Monte Carlo simulation for model `m` with
(keyword) parameters as specified in the dictionary/named tuple `params`.
"""
DQMC(m::Model, params::Dict{Symbol, T}) where T = DQMC(m; params...)
DQMC(m::Model, params::NamedTuple) = DQMC(m; params...)


# convenience
@inline beta(mc::DQMC) = mc.p.beta
@inline nslices(mc::DQMC) = mc.p.slices
@inline model(mc::DQMC) = mc.model
@inline conf(mc::DQMC) = mc.conf
@inline current_slice(mc::DQMC) = mc.s.current_slice
@inline last_sweep(mc::DQMC) = mc.last_sweep
@inline configurations(mc::DQMC) = mc.configs
@inline lattice(mc::DQMC) = lattice(mc.model)
@inline parameters(mc::DQMC) = merge(parameters(mc.p), parameters(mc.model))
@inline parameters(p::DQMCParameters) = (
    beta = p.beta, delta_tau = p.delta_tau, thermalization = p.thermalization, sweeps = p.sweeps
)

# cosmetics
import Base.summary
import Base.show
Base.summary(mc::DQMC) = "DQMC simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::DQMC)
    print(io, "Determinant quantum Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, sigdigits=3)))\n")
    N_th_meas = length(mc.thermalization_measurements)
    N_me_meas = length(mc.measurements)
    print(io, "Measurements: ", N_th_meas + N_me_meas, " ($N_th_meas + $N_me_meas)")
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC) = print(io, mc)

function ConfigRecorder(mc::DQMC, model::Model, rate = 10)
    ConfigRecorder{typeof(compress(mc, model, conf(mc)))}(rate)
end



function init!(mc::DQMC)
    init_hopping_matrices(mc, mc.model)
    initialize_stack(mc, mc.s)
    nothing
end
@deprecate resume_init!(mc::DQMC) init!(mc) false


"""
    run!(mc::DQMC[; kwargs...])

Runs the given Monte Carlo simulation `mc`. Returns true if the run finished and
false if it cancelled early to generate a resumable save-file.

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
        mc::DQMC;
        verbose::Bool = true,
        ignore = tuple(),
        sweeps::Int = mc.p.sweeps,
        thermalization = mc.p.thermalization,
        safe_before::TimeType = now() + Year(100),
        safe_every::TimePeriod = Hour(10000),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_$(Dates.format(safe_before, "d_u_yyyy-HH_MM")).jld",
        overwrite = false
    )

    # Update number of sweeps
    if (mc.p.thermalization != thermalization) || (mc.p.sweeps != sweeps)
        verbose && println("Rebuilding DQMCParameters with new number of sweeps.")
        mc.p = DQMCParameters(mc.p, thermalization = thermalization, sweeps = sweeps)
    end
    total_sweeps = sweeps + thermalization

    # Generate measurement groups
    th_groups = generate_groups(
        mc, mc.model, 
        [mc.measurements[k] for k in keys(mc.thermalization_measurements) if !(k in ignore)]
    )
    groups = generate_groups(
        mc, mc.model, 
        [mc.measurements[k] for k in keys(mc.measurements) if !(k in ignore)]
    )

    start_time = now()
    last_checkpoint = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # fresh stack
    verbose && println("Preparing Green's function stack")
    init!(mc)
    build_stack(mc, mc.s)
    propagate(mc)

    _time = time()
    verbose && println("\n\nThermalization stage - ", thermalization)
    # do_th_measurements && prepare!(mc.thermalization_measurements, mc, mc.model)
    # mc.last_sweep:total_sweeps won't change when last_sweep is changed
    for i in mc.last_sweep+1:total_sweeps
        verbose && (i == thermalization + 1) && println("\n\nMeasurement stage - ", sweeps)
        for u in 1:2 * nslices(mc)
            update(mc, i)

            if current_slice(mc) == 1 && i ≤ thermalization && 
                mc.s.direction == 1 && iszero(i % mc.p.measure_rate)
                if iszero(i % mc.p.measure_rate)
                    for (requirement, group) in th_groups
                        apply!(requirement, group, mc, mc.model, i)
                    end
                end
            end
            if current_slice(mc) == 1 && mc.s.direction == 1 && i > thermalization
                push!(mc.configs, mc, mc.model, i)
                if iszero(i % mc.p.measure_rate)
                    for (requirement, group) in groups
                        apply!(requirement, group, mc, mc.model, i)
                    end
                end
            end
        end
        mc.last_sweep = i

        if mod(i, mc.p.print_rate) == 0
            mc.a.acc_rate = mc.a.acc_rate / (mc.p.print_rate * 2 * nslices(mc))
            mc.a.acc_rate_global = mc.a.acc_rate_global / (mc.p.print_rate / mc.p.global_rate)
            sweep_dur = (time() - _time)/mc.p.print_rate
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
            save(resumable_filename, mc, overwrite = overwrite, rename = false)
            verbose && println("\nEarly save finished")

            return false
        elseif (now() - last_checkpoint) > safe_every
            verbose && println("Performing scheduled save.")
            last_checkpoint = now()
            save(resumable_filename, mc, overwrite = overwrite, rename = false)
        end
    end

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    if verbose
        if length(mc.a.imaginary_probability) > 0
            s = mc.a.imaginary_probability
            println("\nImaginary Probability Errors: ($(s.count))")
            @printf("\tmax  = %0.3e\n", max(s))
            @printf("\tmean = %0.3e\n", mean(s))
            @printf("\tmin  = %0.3e\n\n", min(s))
        end
        if length(mc.a.negative_probability) > 0
            s = mc.a.negative_probability
            println("\nNegative Probability Errors: ($(s.count))")
            @printf("\tmax  = %0.3e\n", max(s))
            @printf("\tmean = %0.3e\n", mean(s))
            @printf("\tmin  = %0.3e\n\n", min(s))
        end
        if length(mc.a.propagation_error) > 0
            s = mc.a.propagation_error
            println("\nPropagation Errors: ($(s.count))")
            @printf("\tmax  = %0.3e\n", max(s))
            @printf("\tmean = %0.3e\n", mean(s))
            @printf("\tmin  = %0.3e\n\n", min(s))
        end
    end

    end_time = now()
    if verbose
        println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
        @printf("Duration: %.2f minutes", (end_time - start_time).value/1000. /60.)
        println()
    end

    return true
end

"""
    update(mc::DQMC, i::Int)

Propagates the Green's function and performs local and global updates at
current imaginary time slice.
"""
function update(mc::DQMC, i::Int)
    propagate(mc)

    # global move
    # note - current_slice and direction are critical here
    # if mc.p.global_moves && current_slice(mc) == nslices(mc) &&
    #        mc.s.direction == -1 && iszero(mod(i, mc.p.global_rate))
    if mc.p.global_moves && current_slice(mc) == 1 &&
        mc.s.direction == 1 && iszero(mod(i, mc.p.global_rate))
        mc.a.prop_global += 1
        b = global_move(mc, mc.model)
        mc.a.acc_global += b
        mc.a.acc_rate_global += b
    end

    # local moves
    sweep_spatial(mc)

    nothing
end

"""
    sweep_spatial(mc::DQMC)

Performs a sweep of local moves along spatial dimension at current
imaginary time slice.
"""
@bm function sweep_spatial(mc::DQMC)
    m = model(mc)
    N = size(conf(mc), 1)
    acc_rate = 0.0

    # @inbounds for i in rand(1:N, N)
    @inbounds for i in 1:N
        detratio, ΔE_boson, passthrough = propose_local(mc, m, i, current_slice(mc), conf(mc))
        mc.a.prop_local += 1

        if mc.p.check_sign_problem
            if abs(imag(detratio)) > 1e-6
                push!(mc.a.imaginary_probability, abs(imag(detratio)))
                mc.p.silent || @printf(
                    "Did you expect a sign problem? imag. detratio:  %.9e\n", 
                    abs(imag(detratio))
                )
            end
            if real(detratio) < 0.0
                push!(mc.a.negative_probability, real(detratio))
                mc.p.silent || @printf(
                    "Did you expect a sign problem? negative detratio %.9e\n",
                    real(detratio)
                )
            end
        end
        p = real(exp(- ΔE_boson) * detratio)

        # Gibbs/Heat bath
        # p = p / (1.0 + p)
        # Metropolis
        if p > 1 || rand() < p
            accept_local!(mc, m, i, current_slice(mc), conf(mc), detratio, ΔE_boson, passthrough)
            # Δ, detratio,ΔE_boson)
            acc_rate += 1.0
            mc.a.acc_local += 1
        end
    end
    mc.a.acc_rate += acc_rate / N
    nothing
end

"""
    replay(mc::DQMC[; configurations::Iterable = mc.configs; kwargs...])

Replays previously generated configurations and measures observables along the
way.

### Keyword Arguments (both):
- `verbose = true`: If true, print progress messaged to stdout.
- `safe_every::TimePeriod`: Set the interval for regularly scheduled saves.
- `safe_before::Date`: If this date is passed, `replay!` will generate a
resumable save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `filename`: Name of the save file. The default is based on `safe_before`.
- `start=1`: The first sweep in the simulation. This will be changed when using
`resume!(save_file)`.
- `ignore`: A collection of measurement keys to ignore. Defaults to the key of
the configuration measurement.
- `measure_rate = 1`: Rate at which measurements are taken. Note that this is 
based on the recorded configurations, not actual sweeps.
"""
function replay!(
        mc::DQMC, configurations = mc.configs;
        ignore = tuple(),
        verbose::Bool = true,
        safe_before::TimeType = now() + Year(100),
        safe_every::TimePeriod = Hour(10000),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_$(Dates.format(safe_before, "d_u_yyyy-HH_MM")).jld",
        overwrite = false,
        measure_rate = 1
    )
    start_time = now()
    last_checkpoint = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # Check for measurements
    isempty(mc.thermalization_measurements) && @debug(
        "There is no thermalization process in a replayed simulation."
    )
    isempty(mc.measurements) && @warn(
        "There are no measurements set up for this simulation!"
    )
    # Generate measurement groups
    groups = generate_groups(
        mc, mc.model, 
        [mc.measurements[k] for k in keys(mc.measurements) if !(k in ignore)]
    )


    if measure_rate != mc.p.measure_rate
        mc.p = DQMCParameters(
            mc.p.global_moves, mc.p.global_rate,
            mc.p.thermalization, mc.p.sweeps,
            mc.p.silent, mc.p.check_sign_problem,mc.p.check_propagation_error,
            mc.p.safe_mult, mc.p.delta_tau, mc.p.beta, mc.p.slices,
            measure_rate, mc.p.print_rate
        )
    end

    verbose && println("Preparing Green's function stack")
    init!(mc)
    build_stack(mc, mc.s)
    propagate(mc)
    mc.s.current_slice = 1
    mc.conf = rand(DQMC, mc.model, nslices(mc))

    _time = time()
    verbose && println("\n\nReplaying measurement stage - ", length(configurations))
    prepare!(mc.measurements, mc, mc.model)
    for i in mc.last_sweep+1:mc.p.measure_rate:length(configurations)
        copyto!(mc.conf, decompress(mc, mc.model, configurations[i]))
        calculate_greens(mc, 0) # outputs to mc.s.greens
        for (requirement, group) in groups
            apply!(requirement, group, mc, mc.model, i)
        end
        mc.last_sweep = i

        if mod(i, mc.p.print_rate) == 0
            sweep_dur = (time() - _time)/mc.p.print_rate
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
            save(resumable_filename, mc, overwrite = overwrite, rename = false)
            verbose && println("\nEarly save finished")

            return false
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

    return true
end
