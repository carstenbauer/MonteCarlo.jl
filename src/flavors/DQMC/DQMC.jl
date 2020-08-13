include("abstract.jl")

"""
Analysis data of determinant quantum Monte Carlo (DQMC) simulation
"""
@with_kw mutable struct DQMCAnalysis
    acc_rate::Float64 = 0.
    prop_local::Int = 0
    acc_local::Int = 0
    acc_rate_global::Float64 = 0.
    prop_global::Int = 0
    acc_global::Int = 0
end

"""
Parameters of determinant quantum Monte Carlo (DQMC)
"""
struct DQMCParameters
    global_moves::Bool
    global_rate::Int
    thermalization::Int
    sweeps::Int
    all_checks::Bool
    safe_mult::Int
    delta_tau::Float64
    beta::Float64
    slices::Int
    measure_rate::Int
end

function DQMCParameters(;global_moves::Bool = false,
                        global_rate::Int    = 5,
                        thermalization::Int = 100,
                        sweeps::Int         = 100,
                        all_checks::Bool    = true,
                        safe_mult::Int      = 10,
                        measure_rate::Int   = 10,
                        warn_round::Bool    = true,
                        kwargs...)
    nt = (;kwargs...)
    keys(nt) == (:beta,) && (nt = (;beta=nt.beta, delta_tau=0.1))
    @assert length(nt) >= 2 "Invalid keyword arguments to DQMCParameters: $nt"
    if (Set ∘ keys)(nt) == Set([:delta_tau, :beta, :slices])
        delta_tau, beta = nt.delta_tau, nt.beta
        slices = round(Int, beta/delta_tau)
        if slices != nt.slices
            error("Given slices ($(nt.slices)) does not match calculated slices beta/delta_tau ≈ $(slices)")
        end
    elseif (Set ∘ keys)(nt) == Set([:beta, :slices])
        beta, slices = nt.beta, nt.slices
        delta_tau = beta / slices
    elseif (Set ∘ keys)(nt) == Set([:delta_tau, :slices])
        delta_tau, slices = nt.delta_tau, nt.slices
        beta = delta_tau * slices
    elseif (Set ∘ keys)(nt) == Set([:delta_tau, :beta])
        delta_tau, beta = nt.delta_tau, nt.beta
        slices = round(beta/delta_tau)
        warn_round && !(slices ≈ beta/delta_tau) && @warn "beta/delta_tau = $(beta/delta_tau) not an integer. Rounded to $slices"
    else
        error("Invalid keyword arguments to DQMCParameters $nt")
    end
    DQMCParameters(global_moves,
                   global_rate,
                   thermalization,
                   sweeps,
                   all_checks,
                   safe_mult,
                   delta_tau,
                   beta,
                   slices,
                   measure_rate)
end




"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, CB<:Checkerboard, ConfType<:Any,
        Stack<:AbstractDQMCStack} <: MonteCarloFlavor
    model::M
    conf::ConfType
    last_sweep::Int
    s::Stack

    p::DQMCParameters
    a::DQMCAnalysis
    thermalization_measurements::Dict{Symbol, AbstractMeasurement}
    measurements::Dict{Symbol, AbstractMeasurement}

    DQMC{M, CB, ConfType, Stack}() where {M<:Model, CB<:Checkerboard,
        ConfType<:Any, Stack<:AbstractDQMCStack} = new()
end

include("stack.jl")
include("slice_matrices.jl")

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
        measurements = :default,
        last_sweep = 0,
        kwargs...
    ) where M<:Model
    # default params
    # paramskwargs = filter(kw->kw[1] in fieldnames(DQMCParameters), kwargs)
    p = DQMCParameters(; kwargs...)

    geltype = greenseltype(DQMC, m)
    heltype = hoppingeltype(DQMC, m)
    conf = rand(DQMC, m, p.slices)
    mc = DQMC{M, checkerboard ? CheckerboardTrue : CheckerboardFalse,
        typeof(conf), DQMCStack{geltype, heltype}}()
    mc.model = m
    mc.p = p
    mc.s = DQMCStack{geltype, heltype}()
    mc.last_sweep = last_sweep

    init!(
        mc, seed = seed, conf = conf,
        thermalization_measurements = thermalization_measurements,
        measurements = measurements
    )
    return mc
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


"""
    init!(mc::DQMC[; seed::Real=-1])

Initialize the determinant quantum Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `Random.seed!(seed)`.
"""
function init!(mc::DQMC;
        seed::Real = -1,
        conf = rand(DQMC,model(mc),nslices(mc)),
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = :default
    )
    seed == -1 || Random.seed!(seed)

    mc.conf = conf

    init_hopping_matrices(mc, mc.model)
    initialize_stack(mc)

    mc.a = DQMCAnalysis()

    mc.thermalization_measurements = thermalization_measurements
    if measurements isa Dict{Symbol, AbstractMeasurement}
        mc.measurements = measurements
    elseif measurements == :default
        mc.measurements = default_measurements(mc, mc.model)
    else
        @warn(
            "`measurements` should be of type Dict{Symbol, AbstractMeasurement}, but is " *
            "$(typeof(measurements)). No measurements have been set."
        )
        mc.measurements = Dict{Symbol, AbstractMeasurement}()
    end

    nothing
end


# Only the stack and DQMCAnalysis need to be intiialized when resuming.
# Everything else is loaded from the save file.
function resume_init!(mc::DQMC)
    init_hopping_matrices(mc, mc.model)
    initialize_stack(mc)
    mc.a = DQMCAnalysis()
    nothing
end


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
- `safe_before::Date`: If this date is passed, `run!` will generate a resumable
save file and exit
- `grace_period = Minute(5)`: Buffer between the current time and `safe_before`.
The time required to generate a save file should be included here.
- `resumable_filename`: Name of the resumable save file. The default is based on
`safe_before`.
- `force_overwrite = false`: If set to true a file with the same name as
`resumable_filename` will be overwritten. (This will create a temporary backup)

See also: [`resume!`](@ref)
"""
@bm function run!(
        mc::DQMC;
        verbose::Bool = true,
        sweeps::Int = mc.p.sweeps,
        thermalization = mc.p.thermalization,
        safe_before::TimeType = now() + Year(100),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_" * Dates.format(safe_before, "d_u_yyyy-HH_MM") * ".jld",
        force_overwrite = false
    )

    # Check for measurements
    do_th_measurements = !isempty(mc.thermalization_measurements)
    do_me_measurements = !isempty(mc.measurements)
    !do_me_measurements && @warn(
        "There are no measurements set up for this simulation!"
    )

    # Update number of sweeps
    if (mc.p.thermalization != thermalization) || (mc.p.sweeps != sweeps)
        verbose && println("Rebuilding DQMCParameters with new number of sweeps.")
        p = DQMCParameters(
            mc.p.global_moves,
            mc.p.global_rate,
            thermalization,
            sweeps,
            mc.p.all_checks,
            mc.p.safe_mult,
            mc.p.delta_tau,
            mc.p.beta,
            mc.p.slices,
            mc.p.measure_rate
        )
        mc.p = p
    end
    total_sweeps = sweeps + thermalization

    start_time = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # fresh stack
    verbose && println("Preparing Green's function stack")
    initialize_stack(mc) # redundant ?!
    build_stack(mc)
    propagate(mc)

    _time = time()
    verbose && println("\n\nThermalization stage - ", thermalization)
    do_th_measurements && prepare!(mc.thermalization_measurements, mc, mc.model)
    # dqmc.last_sweep:total_sweeps won't change when last_sweep is changed
    for i in mc.last_sweep+1:total_sweeps
        verbose && (i == thermalization + 1) && println("\n\nMeasurement stage - ", sweeps)
        for u in 1:2 * nslices(mc)
            update(mc, i)

            # For optimal performance whatever is most likely to fail should be
            # checked first.
            if current_slice(mc) == nslices(mc) && i <= thermalization && mc.s.direction == -1 &&
                    iszero(mod(i, mc.p.measure_rate)) && do_th_measurements
                measure!(mc.thermalization_measurements, mc, mc.model, i)
            end
            if (i == thermalization+1)
                do_th_measurements && finish!(mc.thermalization_measurements, mc, mc.model)
                do_me_measurements && prepare!(mc.measurements, mc, mc.model)
            end
            if current_slice(mc) == nslices(mc) && mc.s.direction == -1 && i > thermalization &&
                    iszero(mod(i, mc.p.measure_rate)) && do_me_measurements
                measure!(mc.measurements, mc, mc.model, i)
            end
        end
        mc.last_sweep = i

        if mod(i, 10) == 0
            mc.a.acc_rate = mc.a.acc_rate / (10 * 2 * nslices(mc))
            mc.a.acc_rate_global = mc.a.acc_rate_global / (10 / mc.p.global_rate)
            sweep_dur = (time() - _time)/10
            max_sweep_duration = max(max_sweep_duration, sweep_dur)
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", sweep_dur)
                @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
                if mc.p.global_moves
                  @printf("\t\tacc rate (global): %.1f%%\n", mc.a.acc_rate_global*100)
                  @printf("\t\tacc rate (global, overall): %.1f%%\n",
                    mc.a.acc_global/mc.a.prop_global*100)
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

            file_exists = isfile(resumable_filename)
            if force_overwrite && file_exists
                parts = splitpath(resumable_filename)
                parts[end] = "." * parts[end]
                temp_filename = _generate_unqiue_JLD_filename(joinpath(parts...))
                mv(resumable_filename, temp_filename)
            end

            # We create a backup manually here because we save extra stuff
            # In either case there should be no conflicting file, so there
            # should be nothing to overwrite.
            resumable_filename = save(resumable_filename, mc)

            if force_overwrite && file_exists
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
    update(mc::DQMC, i::Int)

Propagates the Green's function and performs local and global updates at
current imaginary time slice.
"""
function update(mc::DQMC, i::Int)
    propagate(mc)

    # global move
    # if mc.p.global_moves && (current_slice(mc) == mc.p.slices &&
    #        mc.s.direction == -1 && iszero(mod(i, mc.p.global_rate)))
    #     mc.a.prop_global += 1
    #     b = global_move(mc, mc.model, mc.conf) # not yet in DQMC_optional, i.e. unsupported
    #     mc.a.acc_global += b
    # end

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
    N = nsites(m)

    @inbounds for i in 1:N
        detratio, ΔE_boson, Δ = propose_local(mc, m, i, current_slice(mc), conf(mc))
        mc.a.prop_local += 1

        if abs(imag(detratio)) > 1e-6
            println("Did you expect a sign problem? imag. detratio: ",
                abs(imag(detratio)))
            @printf "%.10e" abs(imag(detratio))
        end
        if real(detratio) < 0.0
            @warn @sprintf(
                "Did you expect a sign problem? negative detratio %.6e\n",
                real(detratio)
            )
        end
        p = real(exp(- ΔE_boson) * detratio)

        # Metropolis
        if p > 1 || rand() < p
            accept_local!(mc, m, i, current_slice(mc), conf(mc), Δ, detratio,
                ΔE_boson)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
        end
    end
    nothing
end

"""
    replay(
        mc::DQMC[;
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
- `ignore`: A collection of measurement keys to ignore. Defaults to the key of
the configuration measurement.
"""
function replay!(
        mc::DQMC;
        configs::ConfigurationMeasurement = try
            k = findfirst(v -> v isa ConfigurationMeasurement, mc.measurements)
            mc.measurements[k]
        catch e
            @error(
                "Failed to find a ConfigurationMeasurement in the given " *
                "DQMC Simulation. Try supplying it manually."
            )
            rethrow(e)
        end,
        ignore = (findfirst(v -> v isa ConfigurationMeasurement, mc.measurements),),
        # reset_measurements = true,
        measure_rate = 1,
        kwargs...
    )
    mc.p = DQMCParameters(
        mc.p.global_moves, mc.p.global_rate,
        mc.p.thermalization, mc.p.sweeps,
        mc.p.all_checks, mc.p.safe_mult,
        mc.p.delta_tau, mc.p.beta, mc.p.slices,
        measure_rate
    )
    replay!(mc, timeseries(configs.obs); ignore=ignore, kwargs...)
end


function replay!(
        mc::DQMC,
        configs::AbstractArray;
        ignore = tuple(),
        verbose::Bool = true,
        safe_before::TimeType = now() + Year(100),
        grace_period::TimePeriod = Minute(5),
        resumable_filename::String = "resumable_" * Dates.format(safe_before, "d_u_yyyy-HH_MM") * ".jld",
        force_overwrite = false
    )
    # Check for measurements
    isempty(mc.thermalization_measurements) && @debug(
        "There is no thermalization process in a replayed simulation."
    )
    isempty(mc.measurements) && @warn(
        "There are no measurements set up for this simulation!"
    )

    start_time = now()
    max_sweep_duration = 0.0
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    verbose && println("Preparing Green's function stack")
    resume_init!(mc)
    initialize_stack(mc) # redundant ?!
    build_stack(mc)
    propagate(mc)

    _time = time()
    verbose && println("\n\nReplaying measurement stage - ", length(configs))
    prepare!(mc.measurements, mc, mc.model)
    for i in mc.last_sweep+1:mc.p.measure_rate:length(configs)
        mc.conf = configs[i]
        mc.s.greens, _ = calculate_greens_and_logdet(mc, nslices(mc))
        for (k, m) in mc.measurements
            k in ignore && continue
            measure!(m, mc, mc.model, i)
        end
        mc.last_sweep = i

        if mod(i, 10) == 0
            sweep_dur = (time() - _time)/10
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

"""
    greens(mc::DQMC)

Obtain the current equal-time Green's function, i.e. the fermionic expectation
value of `Gᵢⱼ = ⟨cᵢcⱼ^†⟩`. The indices relate to sites and flavors, but the
exact meanign depends on the model. For the attractive Hubbard model
`G[i, j] = ⟨c_{i, ↑} c_{j, ↑}^†⟩ = ⟨c_{i, ↓} c_{j, ↓}^†⟩` due to symmetry.

Internally, `mc.s.greens` is an effective Green's function. This method
transforms it to the actual Green's function by multiplying hopping matrix
exponentials from left and right.
"""
greens(mc::DQMC) = _greens!(mc)
function _greens!(
        mc::DQMC_CBFalse, target::Matrix = mc.s.Ul, 
        source::Matrix = mc.s.greens, temp::Matrix = mc.s.Ur
    )
    eThalfminus = mc.s.hopping_matrix_exp
    eThalfplus = mc.s.hopping_matrix_exp_inv
    greens = copy(mc.s.greens)
    greens .= greens * eThalfminus
    greens .= eThalfplus * greens
    vmul!(temp, source, eThalfminus)
    vmul!(target, eThalfplus, temp)
    @assert greens ≈ target
    return target
end
function _greens!(
        mc::DQMC_CBTrue, target::Matrix = mc.s.Ul, 
        source::Matrix = mc.s.greens, temp::Matrix = mc.s.Ur
    )
    chkr_hop_half_minus = mc.s.chkr_hop_half
    chkr_hop_half_plus = mc.s.chkr_hop_half_inv
    copyto!(target, source)

    @inbounds @views begin
        for i in reverse(1:mc.s.n_groups)
            vmul!(temp, target, chkr_hop_half_minus[i])
            copyto!(target, temp)
        end
        for i in reverse(1:mc.s.n_groups)
            vmul!(temp, chkr_hop_half_plus[i], target)
            copyto!(target, temp)
        end
    end
    return target
end


#     save_mc(filename, mc, entryname)
#
# Saves (minimal) information necessary to reconstruct a given `mc::DQMC` to a
# JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC`
function save_mc(file::JLD.JldFile, mc::DQMC, entryname::String="MC")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(mc))
    save_parameters(file, mc.p, entryname * "/Parameters")
    write(file, entryname * "/conf", mc.conf)
    write(file, entryname * "/last_sweep", mc.last_sweep)
    save_measurements(file, mc, entryname * "/Measurements")
    save_model(file, mc.model, entryname * "/Model")
    nothing
end

#     load_mc(data, ::Type{<: DQMC})
#
# Loads a DQMC from a given `data` dictionary produced by `JLD.load(filename)`.
function load_mc(data::Dict, ::Type{T}) where T <: DQMC
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    mc = data["type"]()
    mc.p = load_parameters(data["Parameters"], data["Parameters"]["type"])
    mc.conf = data["conf"]
    mc.last_sweep = data["last_sweep"]
    mc.model = load_model(data["Model"], data["Model"]["type"])

    measurements = load_measurements(data["Measurements"])
    mc.thermalization_measurements = measurements[:TH]
    mc.measurements = measurements[:ME]
    mc.s = MonteCarlo.DQMCStack{geltype(mc), heltype(mc)}()
    mc
end

#   save_parameters(file::JLD.JldFile, p::DQMCParameters, entryname="Parameters")
#
# Saves (minimal) information necessary to reconstruct a given
# `p::DQMCParameters` to a JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC/Parameters`
function save_parameters(file::JLD.JldFile, p::DQMCParameters, entryname::String="Parameters")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(p))

    write(file, entryname * "/global_moves", Int(p.global_moves))
    write(file, entryname * "/global_rate", p.global_rate)
    write(file, entryname * "/thermalization", p.thermalization)
    write(file, entryname * "/sweeps", p.sweeps)
    write(file, entryname * "/all_checks", Int(p.all_checks))
    write(file, entryname * "/safe_mult", p.safe_mult)
    write(file, entryname * "/delta_tau", p.delta_tau)
    write(file, entryname * "/beta", p.beta)
    write(file, entryname * "/slices", p.slices)
    write(file, entryname * "/measure_rate", p.measure_rate)

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function load_parameters(data::Dict, ::Type{T}) where T <: DQMCParameters
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end

    data["type"](
        Bool(data["global_moves"]),
        data["global_rate"],
        data["thermalization"],
        data["sweeps"],
        Bool(data["all_checks"]),
        data["safe_mult"],
        data["delta_tau"],
        data["beta"],
        data["slices"],
        data["measure_rate"],
    )
end


include("DQMC_mandatory.jl")
include("DQMC_optional.jl")
include("measurements.jl")
