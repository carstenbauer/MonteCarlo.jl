################################################################################
### New Version
################################################################################

using MonteCarlo, LinearAlgebra
using MonteCarlo: UnitCell, Bond, Lattice
using MonteCarlo: StructArray, CMat64, CVec64, AbstractField
using MonteCarlo: AbstractLattice, BlockDiagonal


function HBCLattice(Lx, Ly = Lx)
    uc = UnitCell(
        # name
        "HBC Square",

        # Bravais lattice vectors
        ([1.0, +1.0], [1.0, -1.0]),
        
        # Sites
        [[0.0, 0.0], [0.0, 1.0]],

        # Bonds
        [
            # NN, directed
            # bonds from ref plot, π/4 weight for spin up
            Bond(1, 2, ( 0,  1), 1),
            Bond(1, 2, (-1,  0), 1),
            Bond(2, 1, (+1, -1), 1),
            Bond(2, 1, ( 0,  0), 1),

            # NN reversal
            Bond(2, 1, ( 0, -1), 2),
            Bond(2, 1, (+1,  0), 2),
            Bond(1, 2, (-1, +1), 2),
            Bond(1, 2, ( 0,  0), 2),
            
            # NNN
            # positive weight (we need forward and backward facing bonds here too)
            Bond(1, 1, (+1,  0), 3),
            Bond(1, 1, (-1,  0), 3),
            Bond(2, 2, ( 0, +1), 3),
            Bond(2, 2, ( 0, -1), 3),
            # negative weight
            Bond(1, 1, ( 0, +1), 4),
            Bond(1, 1, ( 0, -1), 4),
            Bond(2, 2, (+1,  0), 4),
            Bond(2, 2, (-1,  0), 4),
            
            # Fifth nearest neighbors
            Bond(1, 1, (2, 0), 5),
            Bond(2, 2, (2, 0), 5),
            Bond(1, 1, (0, 2), 5),
            Bond(2, 2, (0, 2), 5),
            # backwards facing bonds
            Bond(1, 1, (-2,  0), 5),
            Bond(2, 2, (-2,  0), 5),
            Bond(1, 1, ( 0, -2), 5),
            Bond(2, 2, ( 0, -2), 5),
        ]
    )

    return Lattice(uc, (Lx, Ly))
end


################################################################################
### Model implementation
################################################################################





"""
t5 = 0  <=> F = 0.2
t5 = (1 - sqrt(2)) / 4 <=> F = 0.009
"""
MonteCarlo.@with_kw_noshow struct HBCModel <: MonteCarlo.Model
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0 / sqrt(2.0)
    t5::Float64 = (1 - sqrt(2)) / 4
    l::Lattice{2}
    @assert l.unitcell.name == "HBC Square"
end

# Constructors
HBCModel(params::Dict{Symbol}) = HBCModel(; params...)
HBCModel(params::NamedTuple) = HBCModel(; params...)
function HBCModel(lattice::Lattice{2}; kwargs...)
    HBCModel(l = lattice; kwargs...)
end
function HBCModel(L, dims; kwargs...)
    l = choose_lattice(HBCModel, dims, L)
    HBCModel(l = l; kwargs...)
end

# spin up and spin down are not equivalent so always 2 flavors
MonteCarlo.nflavors(::HBCModel) = 2
MonteCarlo.lattice(m::HBCModel) = m.l
# default field choice matching the normal Hubbard model
MonteCarlo.choose_field(m::HBCModel) = DensityHirschField

# We have a complex phase in the Hamiltonian so we work with Complex Matrices 
# regardless of the field type. We don't have terms mixing spin up and spin down 
# so we can use BlockDiagonal
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


function MonteCarlo.hopping_matrix(m::HBCModel)
    N = length(m.l)
    t1 = diagm(0 => fill(-ComplexF64(m.mu), N))
    t2 = diagm(0 => fill(-ComplexF64(m.mu), N))

    t1p = m.t1 * cis(+pi/4)
    t1m = m.t1 * cis(-pi/4)
    t2p = + m.t2
    t2m = - m.t2
    
    # Nearest neighbor hoppings
    for b in bonds(m.l, Val(true))
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

function MonteCarlo._save(file::MonteCarlo.FileLike, key::String, m::HBCModel)
    write(file, "$key/VERSION", 2)
    write(file, "$key/tag", "HBCModel")

    write(file, "$key/mu", m.mu)
    write(file, "$key/U", m.U)
    write(file, "$key/t1", m.t1)
    write(file, "$key/t2", m.t2)
    write(file, "$key/t5", m.t5)
    MonteCarlo._save(file, "$key/l", m.l, )

    nothing
end

function MonteCarlo.load_model(data, ::Val{:HBCModel})
    if data["VERSION"] == 1
        # This is compat for an earlier version which may not work due to other 
        # loading failing...
        N = length(data["l/lattice/sites/label"])
        L = round(Int, sqrt(N))
        @assert L*L == N
        l = HBCLattice(L, L)
    elseif data["VERSION"] == 2
        l = _load(data["l"])
    else
        throw(ErrorException("Failed to load HBCModel version $(data["VERSION"])"))
    end

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

# Nothing signifies no spatial indices (or lattice_iterator = nothing)
function MonteCarlo.intE_kernel(mc, model::HBCModel, ::Nothing, G::GreensMatrix, ::Val{2})
    # ⟨U (n↑ - 1/2)(n↓ - 1/2)⟩ = ... 
    # = U [G↑↑ G↓↓ - G↓↑ G↑↓ - 0.5 G↑↑ - 0.5 G↓↓ + G↑↓ + 0.25]
    # = U [(G↑↑ - 1/2)(G↓↓ - 1/2) + G↑↓(1 + G↑↓)]
    # with up-up = down-down and up-down = 0
    return - model.U * sum((diag(G.val) .- 0.5).^2)
end