import Base: @propagate_inbounds

const IsingSpin = Int64
const IsingDistribution = IsingSpin[-1,1]
const IsingConf = Array{IsingSpin}

const IsingTc = 1/(1/2*log(1+sqrt(2)))

"""
Famous Ising model on a cubic lattice.

    IsingModel(; dims, L)

Create Ising model on `dims`-dimensional cubic lattice
with linear system size `L`.
"""
@with_kw_noshow struct IsingModel{C<:AbstractLattice} <: Model # noshow because we override it below
    L::Int
    dims::Int
    l::C = choose_lattice(IsingModel, dims, L)
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    energy::Ref{Float64} = Ref(0.0)
end

function choose_lattice(::Type{IsingModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    IsingModel(params::Dict)
    IsingModel(params::NamedTuple)

Create an Ising model with (keyword) parameters as specified in the
dictionary/named tuple `params`.
"""
IsingModel(params::Dict{Symbol, T}) where T = IsingModel(; params...)
IsingModel(params::NamedTuple) = IsingModel(; params...)

# convenience
@inline Base.ndims(m::IsingModel) = m.dims

# cosmetics
import Base.summary
import Base.show
Base.summary(model::IsingModel) = "$(model.dims)D-Ising model"
# Base.show(io::IO, model::IsingModel{LT}) where LT<:AbstractCubicLattice =
    # print(io, "$(model.dims)D-Ising model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, model::IsingModel{LT}) where LT<:AbstractLattice =
    print(io, "Ising model on $(replace(string(LT), "MonteCarlo."=>"")), L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::IsingModel) = print(io, model)



# implement `Model` interface
@inline nsites(m::IsingModel) = length(m.l)



# implement `MC` interface
Base.rand(::Type{MC}, m::IsingModel) = rand(IsingDistribution, fill(m.L, ndims(m))...)

@propagate_inbounds function propose_local(mc::MC, m::IsingModel, i::Int, conf::IsingConf)
    field = 0.0
    @inbounds for nb in 1:size(m.neighs, 1)
        field += conf[m.neighs[nb, i]]
    end
    delta_E = 2. * conf[i] * field
    return delta_E, nothing
end

@propagate_inbounds function accept_local!(mc::MC, m::IsingModel, i::Int, conf::IsingConf, delta_i, delta_E::Float64)
    m.energy[] += delta_E
    conf[i] *= -1
    nothing
end

# optimized for 2D case
@propagate_inbounds function propose_local(mc::MC, m::IsingModel{SquareLattice}, i::Int, conf::IsingConf)
    neighs = m.neighs
    @inbounds delta_E = 2. * conf[i] * (conf[neighs[1, i]] + conf[neighs[2, i]] +
                              + conf[neighs[3, i]] + conf[neighs[4, i]])
    return delta_E, nothing
end


# optional functions/methods
"""
    global_move(mc::MC, m::IsingModel, conf::IsingConf) -> accepted::Bool

Constructs a Wolff cluster spinflip for configuration `conf`.
Returns wether a cluster spinflip has been performed (any spins have been flipped).
"""
function global_move(mc::MC, m::IsingModel, conf::IsingConf)
    N = nsites(m)
    neighs = m.neighs
    beta = mc.p.beta

    cluster = Array{Int, 1}()
    tocheck = Array{Int, 1}()

    s = rand(1:N)
    push!(tocheck, s)
    push!(cluster, s)

    while !isempty(tocheck)
        cur = pop!(tocheck)
        @inbounds for ni in 1:size(neighs, 1)
            n = neighs[ni,cur]

            @inbounds if conf[cur] == conf[n] && !(n in cluster) && rand() < (1 - exp(- 2.0 * beta))
                push!(tocheck, n)
                push!(cluster, n)
            end

        end
    end

    for spin in cluster
        conf[spin] *= -1
    end
    model.energy[] = energy(mc, m, conf)

    return length(cluster)>1
end



"""
    energy(mc::MC, m::IsingModel, conf::IsingConf)

Calculate energy of Ising configuration `conf` for Ising model `m`.
"""
function energy(mc::MC, m::IsingModel, conf::IsingConf)
    E = 0.0
    for (src, trg) in neighbors(m.l)
        E -= conf[src]*conf[trg]
    end
    return E
end

function energy(mc::MC, m::IsingModel{LT}, conf::IsingConf) where {
        LT <: Union{Chain, SquareLattice, CubicLattice}
    }
    E = 0.0
    for n in 1:ndims(m)
        @inbounds @simd for i in 1:nsites(m)
            E -= conf[i]*conf[m.neighs[n,i]]
        end
    end
    return E
end


"""
    energy(mc::MC, m::IsingModel{SquareLattice}, conf::IsingConf)

Calculate energy of Ising configuration `conf` for 2D Ising model `m`.
This method is a faster variant of the general method for the
square lattice case. (It is roughly twice as fast in this case.)
"""
function energy(mc::MC, m::IsingModel{SquareLattice}, conf::IsingConf)
    neighs = m.neighs
    E = 0.0
    @inbounds @simd for i in 1:nsites(m)
        E -= conf[i]*conf[neighs[1,i]] + conf[i]*conf[neighs[2,i]]
    end
    return E
end


include("measurements.jl")
