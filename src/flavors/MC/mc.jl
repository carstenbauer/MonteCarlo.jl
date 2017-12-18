mutable struct Analysis
    acc_rate::Float64
    prop_local::Int
    acc_local::Int
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int

    Analysis() = new(0.,0,0,0.,0,0)
end

mutable struct MCParameters
    global_moves::Bool
    global_rate::Int
    sweeps::Int

    MCParameters() = new()
end

mutable struct MC{T, S} <: MonteCarloFlavor where T<:Model
    model::T
    conf::S
    energy::Float64
    p::MCParameters
    a::Analysis

    MC{T,S}() where {T,S} = new()
end

function MC(m::M) where M<:Model
    mc = MC{M, conftype(m)}()
    mc.model = m

    # default params
    mc.p = MCParameters()
    mc.p.global_moves = false
    mc.p.global_rate = 5
    mc.p.sweeps = 1000

    init!(mc)
    return mc
end

# TODO: constructor that allows one to set of some MCParameters via positonal or keyword arguments

function init!(mc::MC{<:Model, S}; seed::Real=-1) where S
    seed == -1 || srand(seed)

    mc.conf = rand(mc.model)
    mc.energy = energy(mc.model, mc.conf)

    mc.a = Analysis()
    nothing
end

function run!(mc::MC{<:Model, S}; verbose::Bool=true, sweeps::Int=mc.p.sweeps) where S
    mc.p.sweeps = sweeps

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    tic()
    for i in 1:mc.p.sweeps
        sweep(mc)

        if mc.p.global_moves && mod(i, mc.p.global_rate) == 0
            mc.a.prop_global += 1
            mc.a.acc_global += global_move(mc.model, mc.conf, mc.energy)
        end

        if mod(i, 100) == 0
            mc.a.acc_rate = mc.a.acc_rate / 100
            mc.a.acc_rate_global = mc.a.acc_rate_global / (100 / mc.p.global_rate)
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", toq()/100)
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
    toq();

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
    mc.a.acc_rate_global = mc.a.acc_global / mc.a.prop_global

    end_time = now()
    verbose && println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    verbose && @printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
    nothing
end

function sweep(mc::MC{<:Model, S}) where S
    const N = mc.model.l.sites
    const beta = mc.model.p.β

    @inbounds for i in eachindex(mc.conf)
        ΔE, Δi = propose_local(mc.model, i, mc.conf, mc.energy)
        mc.a.prop_local += 1
        # Metropolis
        if ΔE <= 0 || rand() < exp(- beta*ΔE)
            accept_local!(mc.model, i, mc.conf, mc.energy, Δi, ΔE)
            mc.a.acc_rate += 1/N
            mc.a.acc_local += 1
            mc.energy += ΔE
        end
    end

    nothing
end
