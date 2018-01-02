const IsingSpin = Int8
const IsingDistribution = IsingSpin[-1,1]
const IsingConf = Array{IsingSpin, 2}
const IsingConfs = Array{IsingSpin, 3}

const IsingTc = 1/(1/2*log(1+sqrt(2)))

"""
Famous Ising model on a cubic lattice.
"""
mutable struct IsingModel <: Model
    L::Int
    dims::Int
    β::Float64
    l::CubicLattice
end

"""
    IsingModel(dims::Int, L::Int, β::Float64)
    IsingModel(; dims::Int=2, L::Int=8, β::Float64=1.0)

Create Ising model on `dims`-dimensional cubic lattice
with linear system size `L` and inverse temperature `β`.
"""
function IsingModel(dims::Int, L::Int, β::Float64)
    if dims == 2
        return IsingModel(L, 2, β, SquareLattice(L))
    else
        error("Only `dims=2` supported for now.")
    end
end
IsingModel(; dims::Int=2, L::Int=8, β::Float64=1.0) = IsingModel(dims, L, β)

# methods
"""
    energy(m::IsingModel, conf::IsingConf)

Calculate energy of Ising configuration `conf` for Ising Model `m`.
"""
function energy(m::IsingModel, conf::IsingConf)
    const L = m.l.L
    const neigh = m.l.neighs_cartesian
    E = 0.0
    @simd for x in 1:L
        @simd for y in 1:L
            @inbounds E += - (conf[x,y]*conf[neigh[1,x,y]] + conf[x,y]*conf[neigh[2,x,y]])
        end
    end
    return E
end

import Base.rand
"""
    rand(m::IsingModel)

Draw random Ising configuration.
"""
rand(m::IsingModel) = rand(IsingDistribution, m.l.L, m.l.L)

"""
    conftype(m::IsingModel)

Returns the type of an Ising model configuration.
"""
conftype(m::IsingModel) = IsingConf

"""
    propose_local(m::IsingModel, i::Int, conf::IsingConf, E::Float64) -> ΔE, Δi

Propose a local spin flip at site `i` of current configuration `conf`
with energy `E`. Returns the local move `Δi = new[i] - conf[i]` and energy difference `ΔE = E_new - E_old`.
"""
@inline function propose_local(m::IsingModel, i::Int, conf::IsingConf, E::Float64)
    ΔE = 2. * conf[i] * sum(conf[m.l.neighs[:,i]])
    return ΔE, conf[i]==1?-2:2
end

"""
    accept_local(m::IsingModel, i::Int, conf::IsingConf, E::Float64)

Accept a local spin flip at site `i` of current configuration `conf`
with energy `E`. Arguments `Δi` and `ΔE` correspond to output of `propose_local()`
for that spin flip.
"""
@inline function accept_local!(m::IsingModel, i::Int, conf::IsingConf, E::Float64, Δi, ΔE::Float64)
    conf[i] *= -1
    nothing
end

"""
    global_move(m::IsingModel, conf::IsingConf, E::Float64) -> accepted::Bool

Constructs a Wolff cluster spinflip for configuration `conf` with energy `E`.
Returns wether a cluster spinflip has been performed (any spins have been flipped).
"""
function global_move(m::IsingModel, conf::IsingConf, E::Float64)
    const N = m.l.sites
    const neighs = m.l.neighs
    const beta = m.β

    cluster = Array{Int, 1}()
    tocheck = Array{Int, 1}()

    s = rand(1:N)
    push!(tocheck, s)
    push!(cluster, s)

    while !isempty(tocheck)
        cur = pop!(tocheck)
        @inbounds for n in neighs[:,cur]

            @inbounds if conf[cur] == conf[n] && !(n in cluster) && rand() < (1 - exp(- 2.0 * beta))
                push!(tocheck, n)
                push!(cluster, n)
            end

        end
    end

    for spin in cluster
        conf[spin] *= -1
    end

    return length(cluster)>1
    #return length(cluster) # technically a misuse. allows us to see the cluster size * 100 as acc_rate_global.
end

"""
    prepare_observables(m::IsingModel)

Initializes observables for the Ising model and returns a `Dict{String, Observable}`.

See also [measure_observables!](@ref) and [finish_observables!](@ref).
"""
function prepare_observables(m::IsingModel)
    obs = Dict{String,Observable}()
    obs["E"] = Observable(Float64, "Total energy")
    obs["E2"] = Observable(Float64, "Total energy squared")
    obs["e"] = Observable(Float64, "Energy (per site)")

    obs["M"] = Observable(Float64, "Total magnetization")
    obs["M2"] = Observable(Float64, "Total magnetization squared")
    obs["m"] = Observable(Float64, "Magnetization (per site)")

    obs["χ"] = Observable(Float64, "Susceptibility")

    obs["C"] = Observable(Float64, "Specific Heat")

    return obs
end

"""
    measure_observables!(m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)

Measures observables and updates corresponding `Observable` objects in `obs`.

See also [prepare_observables](@ref) and [finish_observables!](@ref).
"""
function measure_observables!(m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)
    const N = m.l.sites

    # energie
    E2 = E^2
    add!(obs["E"], E)
    add!(obs["E2"], E2)
    add!(obs["e"], E/N)

    # magnetization
    M::Float64 = abs(sum(conf))
    M2 = M^2
    add!(obs["M"], M)
    add!(obs["M2"], M2)
    add!(obs["m"], M/N)

    nothing
end

"""
    measure_observables!(m::IsingModel, obs::Dict{String,Observable}, conf::IsingConf, E::Float64)

Calculates magnetic susceptibility and specific heat and updates corresponding `Observable` objects in `obs`.

See also [prepare_observables](@ref) and [measure_observables!](@ref).
"""
function finish_observables!(m::IsingModel, obs::Dict{String,Observable})
    const N = m.l.sites
    const β = m.β

    # specific heat
    const E = mean(obs["E"])
    const E2 = mean(obs["E2"])
    add!(obs["C"], β*β*(E2/N - E*E/N))

    # susceptibility
    const M = mean(obs["M"])
    const M2 = mean(obs["M2"])
    add!(obs["χ"], β*(M2/N - M*M/N))

    nothing
end
