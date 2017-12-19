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
IsingModel(L::Int, β::Float64) = IsingModel(2, L, β)
IsingModel(β::Float64, L::Int) = IsingModel(L, β)

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
