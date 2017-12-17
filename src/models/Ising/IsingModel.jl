mutable struct IsingParameters
    L::Int
    dims::Int
    β::Float64
end

const IsingSpin = Int8
const IsingDistribution = IsingSpin[0,1]
const IsingConf = Array{IsingSpin, 2}
const IsingConfs = Array{IsingSpin, 3}

const IsingTc = 1/(1/2*log(1+sqrt(2)))

"""
Famous Ising model on a cubic lattice.
"""
mutable struct IsingModel <: Model
    p::IsingParameters
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
        return IsingModel(IsingParameters(L, 2, β), SquareLattice(L))
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
