const HeisenbergSpin = SVector{3, Float64}
const HeisenbergConf = Array{HeisenbergSpin}

"""
Famous Heisenberg model on a cubic lattice.

    HeisenbergModel(; dims, L)

Create Heisenberg model on `dims`-dimensional cubic lattice
with linear system size `L`.
"""
@with_kw_noshow mutable struct HeisenbergModel{LT<:AbstractLattice} <: Model # noshow because we override it below
    L::Int
    dims::Int
    l::LT = choose_lattice(HeisenbergModel, dims, L)
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    energy::Float64 = 0.0
end

function choose_lattice(::Type{HeisenbergModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    HeisenbergModel(params::Dict)
    HeisenbergModel(params::NamedTuple)

Create an Heisenberg model with (keyword) parameters as specified in the
dictionary/named tuple `params`.
"""
HeisenbergModel(params::Dict{Symbol, T}) where T = HeisenbergModel(; params...)
HeisenbergModel(params::NamedTuple) = HeisenbergModel(; params...)

# convenience
@inline Base.ndims(m::HeisenbergModel) = m.dims

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HeisenbergModel) = "$(model.dims)D-Heisenberg model"
# Base.show(io::IO, model::HeisenbergModel{LT}) where LT<:AbstractCubicLattice =
    # print(io, "$(model.dims)D-Heisenberg model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, model::HeisenbergModel{LT}) where LT<:AbstractLattice =
    print(io, "Heisenberg model on $(replace(string(LT), "MonteCarlo."=>"")), L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HeisenbergModel) = print(io, model)



# implement `Model` interface
@inline nsites(m::HeisenbergModel) = length(m.l)



# implement `MC` interface
function Base.rand(::Type{MC}, m::HeisenbergModel)
    reshape([rand_spin(MC, m) for _ in 1:nsites(m)], fill(m.L, ndims(m))...)
end
function rand_spin(::Type{MC}, m::HeisenbergModel)
    phi = 2.0 * pi * rand(Float64)
    ct = 2.0 * rand(Float64) - 1.0
    st = sqrt(1. - ct*ct)

    SVector{3, Float64}(
        st * cos(phi),
        st * sin(phi),
        ct
    )
end

@propagate_inbounds function propose_local(mc::MC, m::HeisenbergModel, i::Int, conf::HeisenbergConf)
    new_spin = rand_spin(MC, m)
    field = HeisenbergSpin(0.0, 0.0, 0.0)
    @inbounds for nb in 1:size(m.neighs, 1)
        field += conf[m.neighs[nb, i]]
    end
    delta_E = -dot(new_spin - conf[i], field)
    return delta_E, new_spin
end

# Why is the order of arguments different here?
@propagate_inbounds function accept_local!(mc::MC, m::HeisenbergModel, i::Int, conf::HeisenbergConf, new_spin::HeisenbergSpin, delta_E::Float64)
    m.energy += delta_E
    conf[i] = new_spin
    nothing
end



"""
    energy(mc::MC, m::HeisenbergModel, conf::HeisenbergConf)

Calculate energy of Heisenberg configuration `conf` for Heisenberg model `m`.
"""
function energy(mc::MC, m::HeisenbergModel, conf::HeisenbergConf)
    E = 0.0
    for (src, trg) in neighbors(m.l)
        E -= dot(conf[src], conf[trg])
    end
    return E
end

function energy(mc::MC, m::HeisenbergModel{LT}, conf::HeisenbergConf) where {
        LT <: Union{Chain, SquareLattice, CubicLattice}
    }
    E = 0.0
    for n in 1:ndims(m)
        @inbounds @simd for i in 1:nsites(m)
            E -= dot(conf[i], conf[m.neighs[n,i]])
        end
    end
    return E
end


"""
    energy(mc::MC, m::HeisenbergModel{SquareLattice}, conf::HeisenbergConf)

Calculate energy of Heisenberg configuration `conf` for 2D Heisenberg model `m`.
This method is a faster variant of the general method for the
square lattice case. (It is roughly twice as fast in this case.)
"""
function energy(mc::MC, m::HeisenbergModel{SquareLattice}, conf::HeisenbergConf)
    neighs = m.neighs
    E = 0.0
    @inbounds @simd for i in 1:nsites(m)
        E -= dot(conf[i], conf[neighs[1,i]]) + dot(conf[i], conf[neighs[2,i]])
    end
    return E
end


include("measurements.jl")
