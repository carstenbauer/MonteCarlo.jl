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
    energy::Ref{Float64} = Ref(0.0)
end

init!(mc::MC, m::IsingModel) = energy(mc, m, mc.conf)

function choose_lattice(::Type{IsingModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    elseif dims == 3
        return CubicLattice(L)
    else
        error("Can't create lattice with dimension $dims")
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
function Base.show(io::IO, model::IsingModel{LT}) where LT<:AbstractLattice
    print(io, 
        "Ising model on $(replace(string(LT), "MonteCarlo."=>"")), " * 
        "L=$(model.L) ($(model.l.sites) sites)"
    )
end
Base.show(io::IO, m::MIME"text/plain", model::IsingModel) = print(io, model)



# implement `Model` interface
@inline lattice(m::IsingModel) = m.l



# implement `MC` interface
Base.rand(::Type{MC}, m::IsingModel) = rand(IsingDistribution, fill(m.L, ndims(m))...)

@propagate_inbounds @bm function propose_local(mc::MC, m::IsingModel, i::Int, conf::IsingConf)
    field = 0.0
    @inbounds for b in bonds(m.l, i)
        field += conf[b.to]
    end
    delta_E = 2.0 * conf[i] * field
    return delta_E, nothing
end

@propagate_inbounds @bm function accept_local!(
        mc::MC, m::IsingModel, i::Int, conf::IsingConf, delta_E::Float64, passthrough
    )
    m.energy[] += delta_E
    conf[i] *= -1
    nothing
end

# # optimized for 2D case
# @propagate_inbounds @bm function propose_local(
#         mc::MC, m::IsingModel{SquareLattice}, i::Int, conf::IsingConf
#     )
#     neighs = m.l.neighs
#     @inbounds delta_E = 2.0 * conf[i] * (
#         conf[neighs[1, i]] + conf[neighs[2, i]] +
#         conf[neighs[3, i]] + conf[neighs[4, i]]
#     )
#     return delta_E, nothing
# end


# optional functions/methods
"""
    global_move(mc::MC, m::IsingModel, conf::IsingConf) -> accepted::Bool

Constructs a Wolff cluster spinflip for configuration `conf`.
Returns wether a cluster spinflip has been performed (any spins have been flipped).
"""
@bm function global_move(mc::MC, m::IsingModel, conf::IsingConf)
    N = length(lattice(m))
    beta = mc.p.beta

    cluster = Array{Int, 1}()
    tocheck = Array{Int, 1}()

    s = rand(1:N)
    push!(tocheck, s)
    push!(cluster, s)

    while !isempty(tocheck)
        cur = pop!(tocheck)
        @inbounds for b in bonds(m.l, cur)

            if conf[cur] == conf[b.to] && !(b.to in cluster) && rand() < (1 - exp(- 2.0 * beta))
                push!(tocheck, b.to)
                push!(cluster, b.to)
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
    for b in bonds(m.l)
        E -= conf[b.from] * conf[b.to]
    end
    m.energy[] = E
    return E
end

# function energy(mc::MC, m::IsingModel{LT}, conf::IsingConf) where {
#         LT <: Union{Chain, SquareLattice, CubicLattice}
#     }
#     E = 0.0
#     @inbounds for (src, trg) in bonds(m.l, Val(false))
#         E -= conf[src]*conf[trg]
#     end
#     m.energy[] = E
#     return E
# end


# """
#     energy(mc::MC, m::IsingModel{Square}, conf::IsingConf)

# Calculate energy of Ising configuration `conf` for 2D Ising model `m`.
# This method is a faster variant of the general method for the
# square lattice case. (It is roughly twice as fast in this case.)
# """
# function energy(mc::MC, m::IsingModel{Square}, conf::IsingConf)
#     neighs = m.l.neighs
#     E = 0.0
#     @inbounds @simd for i in 1:length(lattice(m))
#         E -= conf[i]*conf[neighs[1,i]] + conf[i]*conf[neighs[2,i]]
#     end
#     m.energy[] = E
#     return E
# end



function _save(file::FileLike, entryname::String, m::IsingModel)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "IsingModel")

    write(file, entryname * "/L", m.L)
    write(file, entryname * "/dims", m.dims)
    _save(file, entryname * "/l", m.l)
    write(file, entryname * "/energy", m.energy[])
    nothing
end

function load_model(data, ::Val{:IsingModel})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load IsingModel version $(data["VERSION"])"))
    end

    l = _load(data["l"], to_tag(data["l"]))
    model = IsingModel(
        L = data["L"],
        dims = data["dims"],
        l = l
    )
    model.energy[] = data["energy"]
    model
end
to_tag(::Type{<: IsingModel}) = Val(:IsingModel)

compress(mc::MC, ::IsingModel, c) = BitArray(c .== 1)
compressed_conf_type(::Type{<: MC}, ::Type{<: IsingModel}) = BitArray
function decompress(mc::MC{M, CT}, ::IsingModel, c) where {M, CT}
    CT(2c .- 1)
end


include("measurements.jl")
