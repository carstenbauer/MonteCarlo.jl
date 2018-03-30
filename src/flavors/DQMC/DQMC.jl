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
    sweep_dur::Float64 = 0.
end

"""
Parameters of determinant quantum Monte Carlo (DQMC)
"""
@with_kw mutable struct DQMCParameters
    global_moves::Bool = false
    global_rate::Int = 5
    thermalization::Int = 100 # number of thermalization sweeps
    sweeps::Int = 100 # number of sweeps (after thermalization)

    all_checks::Bool = true # e.g. check if propagation is stable/instable (default should be true)
    safe_mult::Int = 10

    delta_tau::Float64 = 0.1
    beta::Float64
    slices::Int = beta / delta_tau
    @assert isinteger(beta / delta_tau) "beta/delta_tau (= number of imaginary time slices) must be an integer
                but is $(beta / delta_tau)."

    measure_every_nth::Int = 10
end

"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, CB<:Checkerboard, ConfType<:Any, Stack<:AbstractDQMCStack} <: MonteCarloFlavor
    model::M
    conf::ConfType
    energy_boson::Float64
    s::Stack

    p::DQMCParameters
    a::DQMCAnalysis
    obs::Dict{String, Observable}

    DQMC{M, CB, ConfType, Stack}() where {M<:Model, CB<:Checkerboard, ConfType<:Any, Stack<:AbstractDQMCStack} = new()
end

include("linalg.jl")
include("stack.jl")
include("slice_matrices.jl")

"""
    DQMC(m::M; kwargs...) where M<:Model

Create a determinant quantum Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function DQMC(m::M; seed::Int=-1, checkerboard::Bool=false, kwargs...) where M<:Model
    geltype = greenseltype(DQMC, m)
    mc = DQMC{M, checkerboard?CheckerboardTrue:CheckerboardFalse, conftype(DQMC, m), DQMCStack{geltype,Float64}}()
    mc.model = m

    # default params
    # paramskwargs = filter(kw->kw[1] in fieldnames(DQMCParameters), kwargs)
    mc.p = DQMCParameters(; kwargs...)

    mc.s = DQMCStack{geltype,Float64}()

    init!(mc, seed=seed)
    return mc
end

"""
    DQMC(m::M; kwargs::Dict{String, Any})

Create a determinant quantum Monte Carlo simulation for model `m` with (keyword) parameters
as specified in the dictionary `kwargs`.
"""
function DQMC(m::M, kwargs::Dict{String, Any}) where M<:Model
    DQMC(m; convert(Dict{Symbol, Any}, kwargs)...)
end


# cosmetics
import Base.summary
import Base.show
Base.summary(mc::DQMC) = "DQMC simulation of $(summary(mc.model))"
function Base.show(io::IO, mc::DQMC)
    print(io, "Determinant quantum Monte Carlo simulation\n")
    print(io, "Model: ", mc.model, "\n")
    print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, 3)))")
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC) = print(io, mc)


"""
    init!(mc::DQMC[; seed::Real=-1])

Initialize the determinant quantum Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `srand(seed)`.
"""
function init!(mc::DQMC; seed::Real=-1)
    seed == -1 || srand(seed)

    mc.conf = rand(mc, mc.model)
    mc.energy_boson = energy_boson(mc, mc.model, mc.conf)

    init_hopping_matrices(mc, mc.model)
    initialize_stack(mc)

    mc.obs = prepare_observables(mc, mc.model)

    mc.a = DQMCAnalysis()
    nothing
end

"""
    run!(mc::DQMC[; verbose::Bool=true, sweeps::Int, thermalization::Int])

Runs the given Monte Carlo simulation `mc`.
Progress will be printed to `STDOUT` if `verbose=true` (default).
"""
function run!(mc::DQMC; verbose::Bool=true, sweeps::Int=mc.p.sweeps, thermalization=mc.p.thermalization)
    mc.p.sweeps = sweeps
    mc.p.thermalization = thermalization
    const total_sweeps = mc.p.sweeps + mc.p.thermalization

    sweep_dur = Observable(Float64, "Sweep duration"; alloc=ceil(Int, total_sweeps/10))

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    # fresh stack
    verbose && println("Preparing Green's function stack")
    initialize_stack(mc) # redundant ?!
    build_stack(mc)
    propagate(mc)

    tic()
    verbose && println("\n\nThermalization stage - ", thermalization)
    for i in 1:total_sweeps
        verbose && (i == mc.p.thermalization + 1) && println("\n\nMeasurement stage - ", mc.p.sweeps)
        for u in 1:2 * mc.p.slices
            update(mc, i)

            if i > mc.p.thermalization && mc.s.current_slice == mc.p.slices && mc.s.direction == -1 && (i-1)%mc.p.measure_every_nth == 0 # measure criterium
                measure_observables!(mc, mc.model, mc.obs, mc.conf)
            end
        end

        if mod(i, 10) == 0
            mc.a.acc_rate = mc.a.acc_rate / (10 * 2 * mc.p.slices)
            mc.a.acc_rate_global = mc.a.acc_rate_global / (10 / mc.p.global_rate)
            add!(sweep_dur, toq()/10)
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

    nothing
end

"""
    update(mc::DQMC, i::Int)

Propagates the Green's function and performs local and global updates at
current imaginary time slice.
"""
function update(mc::DQMC, i::Int)
    propagate(mc)

    # global move
    # if mc.p.global_moves && (mc.s.current_slice == mc.p.slices && mc.s.direction == -1 && mod(i, mc.p.global_rate) == 0)
    #     mc.a.prop_global += 1
    #     b = global_move(mc, mc.model, mc.conf, mc.energy_boson) # not yet in DQMC_optional, i.e. unsupported
    #     mc.a.acc_global += b
    # end

    # local moves
    sweep_spatial(mc)

    nothing
end

"""
    sweep_spatial(mc::DQMC)

Performs a sweep of local moves along spatial dimension at current imaginary time slice.
"""
function sweep_spatial(mc::DQMC)
    const N = mc.model.l.sites
    const m = mc.model

    @inbounds for i in 1:N
        detratio, delta_E_boson, delta = propose_local(mc, m, i, mc.s.current_slice, mc.conf, mc.energy_boson)
        mc.a.prop_local += 1

        if abs(imag(detratio)) > 1e-6
            println("Did you expect a sign problem? imag. detratio: ", abs(imag(detratio)))
            @printf "%.10e" abs(imag(detratio))
        end
        p = real(exp(- delta_E_boson) * detratio)

        # Metropolis
        if p > 1 || rand() < p
            accept_local!(mc, m, i, mc.s.current_slice, mc.conf, delta, detratio, delta_E_boson)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy_boson += delta_E_boson
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
    const eThalfminus = mc.s.hopping_matrix_exp
    const eThalfplus = mc.s.hopping_matrix_exp_inv

    greens = copy(mc.s.greens)
    greens .= greens * eThalfminus
    greens .= eThalfplus * greens
    return greens
end
function greens(mc::DQMC_CBTrue)
    const chkr_hop_half_minus = mc.s.chkr_hop_half
    const chkr_hop_half_plus = mc.s.chkr_hop_half_inv

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

include("DQMC_mandatory.jl")
include("DQMC_optional.jl")