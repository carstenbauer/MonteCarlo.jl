mutable struct Analysis
    acc_rate::Float64
    prop_local::Int
    acc_local::Int
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int

    Analysis() = new(0.,0.,0,0.)
end

mutable struct MC{T, S} <: MonteCarloFlavor where T<:Model
    model::T
    conf::S
    energy::Float64
    global_moves::Bool
    global_rate::Int
    sweeps::Int
    a::Analysis

    MC{T,S}() where {T,S} = new()
end

function MC(m::M) where M<:Model
    mc = MC{M, conftype(m)}()
    mc.model = m
    mc.global_moves = false
    init!(mc)
    return mc
end

function init!(mc::MC{<:Model, S}; seed::Real=-1) where S
    seed == -1 || srand(seed)
    mc.conf = rand(mc.model)
    mc.energy = energy(mc.model, mc.conf)
    mc.a = Analysis()
    nothing
end

function run!(mc::MC{<:Model, S}; verbose::Bool=true, sweeps::Int=mc.sweeps) where S
    mc.sweeps = sweeps

    start_time = now()
    verbose && println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

    tic()
    for i in 1:mc.sweeps
        sweep(mc)

        if mc.global_moves
            # TODO
        end

        if mod(i, 10) == 0
            mc.a.acc_rate = mc.a.acc_rate / 10
            # mc.a.acc_rate_global = mc.a.acc_rate_global / (10 / p.global_rate)
            if verbose
                println("\t", i)
                @printf("\t\tsweep dur: %.3fs\n", toq()/10)
                @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
            end
            # if p.global_updates
            #   @printf("\t\tacc rate (global): %.1f%%\n", mc.a.acc_rate_global*100)
            #   @printf("\t\tacc rate (global, overall): %.1f%%\n", mc.a.acc_global/mc.a.prop_global*100)
            # end

            mc.a.acc_rate = 0.0
            # mc.a.acc_rate_global = 0.0
            flush(STDOUT)
            tic()
        end
    end
    toq();

    mc.a.acc_rate = mc.a.acc_local / mc.a.prop_local
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
