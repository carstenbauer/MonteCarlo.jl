"""
    HubbardModelRepulsive(lattice; params...)
    HubbardModelRepulsive(L, dims; params...)
    HubbardModelRepulsive(params::Dict)
    HubbardModelRepulsive(params::NamedTuple)
    HubbardModelRepulsive(; params...)

Defines a repulsive (positive `U`) Hubbard model on a given (or derived) 
`lattice`. If a linear system size `L` and dimensionality `dims` is given, the
`lattice` will be a Cubic lattice of fitting size.

Additional parameters (keyword arguments) include:
* `l::AbstractLattice = lattice`: The lattice the model uses. The keyword 
argument takes precedence over the argument `lattice`.
* `U::Float64 = 1.0 > 0.0` is the absolute value of the Hubbard Interaction.
* `t::Float64 = 1.0` is the hopping strength.

Internally, a discrete Hubbard Stratonovich transformation (Hirsch 
transformation) is used in the spin/magnetic channel to enable DQMC. The 
resulting Hubbard Stratonovich fiels is real.
To reduce computational cost a specialized `BlockDiagonal` representation of the
greens matrix is used. 
"""
@with_kw_noshow struct HubbardModelRepulsive{LT<:AbstractLattice} <: HubbardModel
    # user optional
    U::Float64 = -1.0
    t::Float64 = 1.0

    # mandatory (this or (L, dims))
    l::LT

    # # non-user fields
    # flv::Int = 2
    # # to avoid allocations (TODO always real?)
    # IG::Matrix{Float64}  = Matrix{Float64}(undef, 2length(l), 2)
    # IGR::Matrix{Float64} = Matrix{Float64}(undef, 2length(l), 2)
    # R::Matrix{Float64}   = Matrix{Float64}(undef, 2, 2)
    # Δ::Diagonal{Float64, Vector{Float64}} = Diagonal(Vector{Float64}(undef, 2))
    # RΔ::Matrix{Float64}  = Matrix{Float64}(undef, 2, 2)
end


HubbardModelRepulsive(params::Dict{Symbol, T}) where T = HubbardModelRepulsive(; params...)
HubbardModelRepulsive(params::NamedTuple) = HubbardModelRepulsive(; params...)
function HubbardModelRepulsive(lattice::AbstractLattice; kwargs...)
    HubbardModelRepulsive(l = lattice; kwargs...)
end
function HubbardModelRepulsive(L, dims; kwargs...)
    l = choose_lattice(HubbardModelRepulsive, dims, L)
    HubbardModelRepulsive(l = l; kwargs...)
end

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelRepulsive) = "repulsive Hubbard model"
function Base.show(io::IO, model::HubbardModelRepulsive)
    print(io, "repulsive Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelRepulsive) = print(io, model)


# Convenience
@inline parameters(m::HubbardModelRepulsive) = (N = length(m.l), t = m.t, U = m.U)


# optional optimization
hopping_matrix_type(::Type{DQMC}, ::HubbardModelRepulsive) = BlockDiagonal{Float64, 2, Matrix{Float64}}
greens_matrix_type( ::Type{DQMC}, ::HubbardModelRepulsive) = BlockDiagonal{Float64, 2, Matrix{Float64}}


choose_field(::HubbardModelRepulsive) = MagneticHirschField

"""
    hopping_matrix(mc::DQMC, m::HubbardModelRepulsive)

Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

# TODO
# Note that since we have a time reversal symmetry relating spin-up
# to spin-down we only consider one spin sector (one flavor) for the repulsive
# Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, model::HubbardModelRepulsive)
    N = length(model.l)
    T = zeros(N, N)

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(model.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -model.t
        end
    end

    return BlockDiagonal(T, copy(T))
end

greens(mc::DQMC, model::HubbardModelRepulsive) = greens(mc)

function save_model(
        file::JLDFile,
        m::HubbardModelRepulsive,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "HubbardModelRepulsive")

    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")
    write(file, entryname * "/flv", m.flv)

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function _load(data, ::Val{:HubbardModelRepulsive})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HubbardModelRepulsive version $(data["VERSION"])"))
    end

    l = _load(data["l"], to_tag(data["l"]))
    HubbardModelRepulsive(
        U = data["U"],
        t = data["t"],
        l = l,
        flv = data["flv"]
    )
end
to_tag(::Type{<: HubbardModelRepulsive}) = Val(:HubbardModelRepulsive)



################################################################################
### Measurement kernels
################################################################################



function intE_kernel(mc, model::HubbardModelRepulsive, G::GreensMatrix)
    # up-down zero
    - model.U * sum((diag(G.val.blocks[1]) .- 0.5) .* (diag(G.val.blocks[2]) .- 0.5))
end
