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
@with_kw struct DQMCParameters
    global_moves::Bool = false
    global_rate::Int = 5
    thermalization::Int = 100 # number of thermalization sweeps
    sweeps::Int = 100 # number of sweeps (after thermalization)

    all_checks::Bool = true # e.g. check if propagation is stable/instable
    safe_mult::Int = 10

    delta_tau::Float64 = 0.1
    beta::Float64
    slices::Int = beta / delta_tau
    @assert isinteger(beta / delta_tau) string("beta/delta_tau", "
        (= number of imaginary time slices) must be an integer but is",
        beta / delta_tau, ".")

    measure_rate::Int = 10
end

"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, CB<:Checkerboard, ConfType<:Any,
        Stack<:AbstractDQMCStack} <: MonteCarloFlavor
    model::M
    conf::ConfType
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
"""
function DQMC(m::M;
        seed::Int=-1,
        checkerboard::Bool=false,
        thermalization_measurements = Dict{Symbol, AbstractMeasurement}(),
        measurements = :default,
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
false if it cancelled early to generate a save-file.

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
- `filename`: Name of the save file. The default is based on `safe_before`.
- `start=1`: The first sweep in the simulation. This will be changed when using
`resume!(save_file)`.

See also: [`resume!`](@ref)
"""
function run!(
        mc::DQMC;
        verbose::Bool = true,
        sweeps::Int = mc.p.sweeps,
        thermalization = mc.p.thermalization,
        safe_before::TimeType = now() + Year(100),
        grace_period::TimePeriod = Minute(5),
        filename::String = "resumable_" * Dates.format(safe_before, "d_u_yyyy-HH_MM") * ".jld",
        force_overwrite = false,
        start = 1
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
    for i in start:total_sweeps
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
            filename = save(filename, mc, force_overwrite = force_overwrite)
            save_rng(filename)
            jldopen(filename, "r+") do f
                write(f, "last_sweep", i)
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
function sweep_spatial(mc::DQMC)
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
    greens(mc::DQMC)

Obtain the current equal-time Green's function.

Internally, `mc.s.greens` is an effective Green's function. This method transforms
this effective one to the actual Green's function by multiplying hopping matrix
exponentials from left and right.
"""
function greens(mc::DQMC_CBFalse)
    eThalfminus = mc.s.hopping_matrix_exp
    eThalfplus = mc.s.hopping_matrix_exp_inv

    greens = copy(mc.s.greens)
    greens .= greens * eThalfminus
    greens .= eThalfplus * greens
    return greens
end
function greens(mc::DQMC_CBTrue)
    chkr_hop_half_minus = mc.s.chkr_hop_half
    chkr_hop_half_plus = mc.s.chkr_hop_half_inv

    greens = copy(mc.s.greens)

    @inbounds @views begin
        for i in reverse(1:mc.s.n_groups)
          greens .= greens * chkr_hop_half_minus[i]
        end
        for i in reverse(1:mc.s.n_groups)
          greens .= chkr_hop_half_plus[i] * greens
        end
    end
    return greens
end


#     save_mc(filename, mc, entryname)
#
# Saves (minimal) information necessary to reconstruct a given `mc::DQMC` to a
# JLD-file `filename` under group `entryname`.
#
# When saving a simulation the default `entryname` is `MC`
function save_mc(filename::String, mc::DQMC, entryname::String="MC")
    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, entryname * "/VERSION", 1)
        write(f, entryname * "/type", typeof(mc))
        write(f, entryname * "/parameters", mc.p)
        write(f, entryname * "/conf", mc.conf)
        # write(f, entryname * "/RNG", Random.GLOBAL_RNG)
    end
    save_measurements(
        filename, mc, entryname * "/Measurements",
        force_overwrite=true, allow_rename=false
    )
    save_model(filename, mc.model, entryname * "/Model")
    nothing
end

#     load_mc(data, ::Type{<: DQMC})
#
# Loads a DQMC from a given `data` dictionary produced by `JLD.load(filename)`.
function load_mc(data::Dict, ::Type{T}) where T <: DQMC
    @assert data["VERSION"] == 1

    mc = data["type"]()
    mc.p = data["parameters"]
    mc.conf = data["conf"]
    mc.model = load_model(data["Model"], data["Model"]["type"])

    measurements = load_measurements(data["Measurements"])
    mc.thermalization_measurements = measurements[:TH]
    mc.measurements = measurements[:ME]
    mc.s = MonteCarlo.DQMCStack{geltype(mc), heltype(mc)}()
    mc
end


include("DQMC_mandatory.jl")
include("DQMC_optional.jl")
include("measurements.jl")
