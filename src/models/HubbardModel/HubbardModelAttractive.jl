"""
    HubbardModelAttractive(lattice; params...)
    HubbardModelAttractive(L, dims; params...)
    HubbardModelAttractive(params::Dict)
    HubbardModelAttractive(params::NamedTuple)
    HubbardModelAttractive(; params...)

Defines an attractive (negative `U`) Hubbard model on a given (or derived) 
`lattice`. If a linear system size `L` and dimensionality `dims` is given, the
`lattice` will be a Cubic lattice of fitting size.

Additional parameters (keyword arguments) include:
* `l::AbstractLattice = lattice`: The lattice the model uses. The keyword 
argument takes precedence over the argument `lattice`.
* `U::Float64 = 1.0 > 0.0` is the absolute value of the Hubbard Interaction.
* `t::Float64 = 1.0` is the hopping strength.
* `mu::Float64` is the chemical potential.

Internally, a discrete Hubbard Stratonovich transformation (Hirsch 
transformation) is used in the spin/magnetic channel to enable DQMC. The 
resulting Hubbard Stratonovich fiels is real.
Furthermore, we use spin up/down symmetry to speed up the simulation. As a 
result the greens matrix is of size (N, N) with N the number of sites, and the
element G[i, j] corresponds to the up-up and down-down element. 
"""
@with_kw_noshow struct HubbardModelAttractive{LT<:AbstractLattice} <: HubbardModel
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0
    l::LT
end


HubbardModelAttractive(params::Dict{Symbol}) = HubbardModelAttractive(; params...)
HubbardModelAttractive(params::NamedTuple) = HubbardModelAttractive(; params...)
function HubbardModelAttractive(lattice::AbstractLattice; kwargs...)
    HubbardModelAttractive(l = lattice; kwargs...)
end
function HubbardModelAttractive(L, dims; kwargs...)
    l = choose_lattice(HubbardModelAttractive, dims, L)
    HubbardModelAttractive(l = l; kwargs...)
end


# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelAttractive) = "attractive Hubbard model"
function Base.show(io::IO, model::HubbardModelAttractive)
    print(io, "attractive Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelAttractive) = print(io, model)


# Convenience
@inline parameters(m::HubbardModelAttractive) = (N = length(m.l), t = m.t, U = m.U, mu = m.mu)
choose_field(::HubbardModelAttractive) = DensityHirschField



"""
Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

Note that since we have a time reversal symmetry relating spin-up
to spin-down we only consider one spin sector (one flavor) for the attractive
Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, m::HubbardModelAttractive{L}) where {L<:AbstractLattice}
    N = length(m.l)
    T = diagm(0 => fill(-m.mu, N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(m.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -m.t
        end
    end

    if max(nflavors(field(mc)), nflavors(m)) == 1
        return T
    else
        return BlockDiagonal(T, copy(T))
    end
end


function save_model(
        file::JLDFile,
        m::HubbardModelAttractive,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "HubbardModelAttractive")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function _load(data, ::Val{:HubbardModelAttractive})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HubbardModelAttractive version $(data["VERSION"])"))
    end

    l = _load(data["l"], to_tag(data["l"]))
    HubbardModelAttractive(
        mu = data["mu"],
        U = data["U"],
        t = data["t"],
        l = l,
    )
end
to_tag(::Type{<: HubbardModelAttractive}) = Val(:HubbardModelAttractive)