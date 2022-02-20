################################################################################
### Lattice via LatticePhysics
################################################################################



using LinearAlgebra, LatticePhysics, LatPhysUnitcellLibrary

function LatPhysUnitcellLibrary.getUnitcellSquare(
            unitcell_type  :: Type{U},
            implementation :: Val{17}
        ) :: U where {LS,LB,S<:AbstractSite{LS,2},B<:AbstractBond{LB,2}, U<:AbstractUnitcell{S,B}}

    # return a new Unitcell
    return newUnitcell(
        # Type of the unitcell
        U,

        # Bravais lattice vectors
        [[1.0, +1.0], [1.0, -1.0]],
        
        # Sites
        S[
            newSite(S, [0.0, 0.0], getDefaultLabelN(LS, 1)),
            newSite(S, [0.0, 1.0], getDefaultLabelN(LS, 2))
        ],

        # Bonds
        B[
            # NN, directed
            # bonds from ref plot, π/4 weight for spin up
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (0, 1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (-1, 0)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (+1, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (0, 0)),

            # NN reversal
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (0, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (+1, 0)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (-1, +1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (0, 0)),
            
            # NNN
            # positive weight (we need forward and backward facing bonds here too)
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (+1, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (-1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, +1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, -1)),
            # negative weight
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, +1)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, -1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (+1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (-1, 0)),
            
            # Fifth nearest neighbors
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, 2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, 2)),  
            # backwards facing bonds
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, -2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, -2)), 
        ]
    )
end



################################################################################
### Model implementation
################################################################################



using MonteCarlo, LinearAlgebra
using MonteCarlo: StructArray, CMat64, CVec64, AbstractField
using MonteCarlo: AbstractLattice, BlockDiagonal, vmul!, conf

"""
t5 = 0  <=> F = 0.2
t5 = (1 - sqrt(2)) / 4 <=> F = 0.009
"""
MonteCarlo.@with_kw_noshow struct HBCModel{LT<:AbstractLattice} <: MonteCarlo.Model
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0 / sqrt(2.0)
    t5::Float64 = (1 - sqrt(2)) / 4
    l::LT
end


HBCModel(params::Dict{Symbol}) = HBCModel(; params...)
HBCModel(params::NamedTuple) = HBCModel(; params...)
function HBCModel(lattice::AbstractLattice; kwargs...)
    HBCModel(l = lattice; kwargs...)
end
function HBCModel(L, dims; kwargs...)
    l = choose_lattice(HBCModel, dims, L)
    HBCModel(l = l; kwargs...)
end

MonteCarlo.nflavors(::HBCModel) = 2
MonteCarlo.lattice(m::HBCModel) = m.l
MonteCarlo.choose_field(m::HBCModel) = m.U < 0.0 ? MagneticHirschField : DensityHirschField

MonteCarlo.hopping_eltype(::HBCModel) = ComplexF64
MonteCarlo.hopping_matrix_type(::AbstractField, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
MonteCarlo.greens_eltype(::AbstractField, ::HBCModel) = ComplexF64
MonteCarlo.greens_matrix_type( ::AbstractField, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HBCModel) = "Hofmann Berg Hubbard model"
function Base.show(io::IO, model::HBCModel)
    print(io, "Hofmann Berg Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, m::MIME"text/plain", model::HBCModel) = print(io, model)


# Convenience
@inline MonteCarlo.parameters(m::HBCModel) = (N = length(m.l), t1 = m.t1, t2 = m.t2, t5 = m.t5, U = -m.U, mu = m.mu)


function MonteCarlo.hopping_matrix(m::HBCModel{<: LatPhysLattice})
    N = length(m.l)
    t1 = diagm(0 => fill(-ComplexF64(m.mu), N))
    t2 = diagm(0 => fill(-ComplexF64(m.mu), N))

    t1p = m.t1 * cis(+pi/4)
    t1m = m.t1 * cis(-pi/4)
    t2p = + m.t2
    t2m = - m.t2
    
    # Nearest neighbor hoppings
    for b in bonds(m.l.lattice)
        if b.label == 1 
            # ϕ_ij^↑ = + π/4
            t1[b.from, b.to] = - t1p
            # ϕ_ij^↓ = - π/4
            t2[b.from, b.to] = - t1m
        elseif b.label == 2
            # TODO do we use reverse NN? - doesn't look like it (sign problem)
            t1[b.from, b.to] = - t1m
            t2[b.from, b.to] = - t1p
            
        elseif b.label == 3
            t1[b.from, b.to] = - t2p
            t2[b.from, b.to] = - t2p
        elseif b.label == 4
            t1[b.from, b.to] = - t2m
            t2[b.from, b.to] = - t2m
        else
            t1[b.from, b.to] = - m.t5
            t2[b.from, b.to] = - m.t5
        end
    end

    return BlockDiagonal(StructArray(t1), StructArray(t2))
end

function MonteCarlo.save_model(
        file::MonteCarlo.JLDFile, m::HBCModel,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "HBCModel")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t1", m.t1)
    write(file, entryname * "/t2", m.t2)
    write(file, entryname * "/t5", m.t5)
    MonteCarlo.save_lattice(file, m.l, entryname * "/l")

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function MonteCarlo._load_model(data, ::Val{:HBCModel})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HBCModel version $(data["VERSION"])"))
    end

    l = MonteCarlo._load(data["l"], MonteCarlo.to_tag(data["l"]))
    HBCModel(
        mu = data["mu"],
        U = data["U"],
        t1 = data["t1"],
        t2 = data["t2"],
        t5 = data["t5"],
        l = l,
    )
end


################################################################################
### Measurement kernels
################################################################################


function MonteCarlo.intE_kernel(mc, model::HBCModel, G::GreensMatrix, ::Val{2})
    # ⟨U (n↑ - 1/2)(n↓ - 1/2)⟩ = ... 
    # = U [G↑↑ G↓↓ - G↓↑ G↑↓ - 0.5 G↑↑ - 0.5 G↓↓ + G↑↓ + 0.25]
    # = U [(G↑↑ - 1/2)(G↓↓ - 1/2) + G↑↓(1 + G↑↓)]
    # with up-up = down-down and up-down = 0
    - model.U * sum((diag(G.val) .- 0.5).^2)
end

true