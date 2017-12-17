mutable struct Analysis
    acc_rate::Float64
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int

    Analysis() = new(0.,0.,0,0.)
end

mutable struct MC{T, S} <: MonteCarloMethod where T<:Model
    model::T
    conf::S
    energy::Float64
    global_moves::Bool
    global_rate::Int
    sweeps::Int
    a::Analysis

    MC{T,S}() = new()
end

function MC(m::M) where M<:Model
    mc = MC{M, m.conftype}()
    mc.model = m
    mc.global_moves = false
    init!(mc)
    return mc
end

function init!(mc::MC{<:Model, S}; seed::Real=-1) where S
    seed == -1 || srand(seed)
    mc.conf = rand(mc.model)
    energy(mc.model, mc.conf)
    mc.a = Analysis()
    nothing
end

function run!(mc::MC{<:Model, S}) where S
    tic()
    for i in 1:mc.sweeps
        sweep(mc)

        if mc.global_moves &&
            # TODO
        end

        if mod(i, 10) == 0
            mc.a.acc_rate = mc.a.acc_rate / 10
            # mc.a.acc_rate_global = mc.a.acc_rate_global / (10 / p.global_rate)
            println("\t", i)
            @printf("\t\tsweep dur: %.2fs\n", toq()/10)
            @printf("\t\tacc rate (local) : %.1f%%\n", mc.a.acc_rate*100)
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
    nothing
end

function sweep(mc::MC{<:Model, S}) where S
    const L = mc.model.l.L
    const N = mc.model.l.sites
    const beta = mc.model.p.β

    @inbounds for i in eachindex(mc.conf)
        ΔE, Δi = propose_local(mc.model, i, mc.conf, mc.energy)
        # Metropolis
        if ΔE <= 0 || rand() < exp(- p.beta*ΔE)
            accept_local!(mc.model, i, mc.conf, mc.energy, Δi, ΔE)
            mc.a.acc_rate += 1/N
            mc.energy += ΔE
        end
    end

    nothing
end
