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
MonteCarlo.unique_flavors(::HBCModel) = 2
MonteCarlo.total_flavors(::HBCModel) = 2
MonteCarlo.lattice(m::HBCModel) = m.l
# default field choice matching the normal Hubbard model
MonteCarlo.choose_field(::HBCModel) = DensityHirschField

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
    # number of sites
    N = length(m.l)

    # spin up and spin down blocks of T
    tup = diagm(0 => fill(-ComplexF64(m.mu), N))
    tdown = diagm(0 => fill(-ComplexF64(m.mu), N))

    # positive and negative prefactors for t1, t2
    t1p = m.t1 * cis(+pi/4) # ϕ_ij^↑ = + π/4
    t1m = m.t1 * cis(-pi/4) # ϕ_ij^↓ = - π/4
    t2p = + m.t2
    t2m = - m.t2
    
    for b in bonds(m.l, Val(true))
        # NN paper direction
        if b.label == 1 
            tup[b.from, b.to]   = - t1p
            tdown[b.from, b.to] = - t1m
        
        # NN reverse direction
        elseif b.label == 2
            tup[b.from, b.to]   = - t1m
            tdown[b.from, b.to] = - t1p
            
        # NNN solid bonds
        elseif b.label == 3
            tup[b.from, b.to]   = - t2p
            tdown[b.from, b.to] = - t2p

        # NNN dashed bonds
        elseif b.label == 4
            tup[b.from, b.to]   = - t2m
            tdown[b.from, b.to] = - t2m

        # Fifth nearest neighbors
        else
            tup[b.from, b.to]   = - m.t5
            tdown[b.from, b.to] = - m.t5
        end
    end

    return BlockDiagonal(StructArray(tup), StructArray(tdown))
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