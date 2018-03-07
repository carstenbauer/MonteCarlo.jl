include("linalg.jl")
include("stack.jl")

"""
Analysis data of determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMCAnalysis
    acc_rate::Float64
    prop_local::Int
    acc_local::Int
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int
    sweep_dur::Float64

    DQMCAnalysis() = new(0.,0,0,0.,0,0)
end

"""
Parameters of determinant quantum Monte Carlo (DQMC)
"""
mutable struct DQMCParameters
    global_moves::Bool
    global_rate::Int
    thermalization::Int # number of thermalization sweeps
    sweeps::Int # number of sweeps (after thermalization)

    all_checks::Bool # e.g. check if propagation is stable/instable (default should be true)
    safe_mult::Int

    delta_tau::Float64
    slices::Int
    beta::Float64 # redundant (=slices*delta_tau) but keep for convenience

    DQMCParameters() = new()
end

"""
Determinant quantum Monte Carlo (DQMC) simulation
"""
mutable struct DQMC{M<:Model, Checkerboard<:Bool, GreensType<:Number, ConfType} <: MonteCarloFlavor
    model::M
    conf::ConfType
    # greens::GreensType # should this be here or in DQMCStack?
    energy::Float64
    s::DQMCStack

    p::DQMCParameters
    a::DQMCAnalysis
    obs::Dict{String, Observable}

    function DQMC{M,GreensType}() where {M<:Model,GreensType<:Number}
        @assert isleaftype(GreensType)
        new()
    end
end

"""
    DQMC(m::M; kwargs...) where M<:Model

Create a determinant quantum Monte Carlo simulation for model `m` with keyword parameters `kwargs`.
"""
function DQMC(m::M; sweeps::Int=1000, thermalization::Int=0,
            slices::Int=0, beta::Float64=1.0, delta_tau::Float64::0.1, # typically a user wants to specify beta not slices
            global_moves::Bool=false, global_rate::Int=5,
            seed::Int=-1,
            checkerboard::Bool=false) where M<:Model
    mc = DQMC{M, checkerboard, greenstype(m), conftype(m)}()
    mc.model = m

    # default params
    mc.p = MCParameters()

    # number of imaginary time slices
    if slices<=0 # user didn't specify slices (use beta and delta_tau keywords)
        try
            mc.p.slices = beta/delta_tau
            mc.p.beta = beta
            mc.p.delta_tau = delta_tau
        catch
            error("Number of imaginary time slices, i.e. beta/delta_tau, must be an integer.")
        end
    else # user did specify slices (ignore beta keyword)
        mc.p.slices = slices
        mc.p.delta_tau = delta_tau
        mc.p.beta = slices * delta_tau
    end

    mc.p.global_moves = global_moves
    mc.p.global_rate = global_rate
    mc.p.thermalization = thermalization
    mc.p.sweeps = sweeps

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


"""
    init!(mc::DQMC[; seed::Real=-1])

Initialize the determinant quantum Monte Carlo simulation `mc`.
If `seed !=- 1` the random generator will be initialized with `srand(seed)`.
"""
function init!(mc::DQMC; seed::Real=-1)
    seed == -1 || srand(seed)

    mc.conf = rand(mc, mc.model)
    mc.energy = energy(mc, mc.model, mc.conf)

    mc.obs = prepare_observables(mc, mc.model)

    mc.s = DQMCStack()

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

    sweep_dur = Observable(Float64, "Sweep duration"; alloc=ceil(Int, total_sweeps/100))

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    tic()
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
            add!(sweep_dur, toq()/1000)
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

    mc.obs
end

"""
    sweep(mc::DQMC)

Performs a sweep of local moves.
"""
function sweep(mc::DQMC{<:Model, S}) where S
    const N = mc.model.l.sites

    @inbounds for i in eachindex(mc.conf)
        delta_E, delta_i = propose_local(mc.model, i, mc.conf, mc.energy)
        mc.a.prop_local += 1
        # Metropolis
        if delta_E <= 0 || rand() < exp(- delta_E)
            accept_local!(mc.model, i, mc.conf, mc.energy, delta_i, delta_E)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy += delta_E
        end
    end

    nothing
end
