"""
Two dimensional square lattice.
"""
struct SquareLattice <: AbstractLattice
    L::Int
    sites::Int
    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
    lattice::Matrix{Int}

    # for generic checkerboard decomposition
    # OPT: implement Assaad's two-group square lattice version
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

# constructors
"""
    SquareLattice(L::Int)

Create a square lattice with linear dimension `L`.
"""
function SquareLattice(L::Int)
    sites = L^2
    lattice = convert(Array, reshape(1:L^2, (L, L)))
    neighs, neighs_cartesian = build_neighbortable(SquareLattice, lattice, L)

    # for generic checkerboard decomposition
    n_bonds = 2*sites
    bonds = zeros(n_bonds, 3)
    bondid = 1
    for src in lattice
        nup = neighs[1, src]
        bonds[bondid,:] .= [src,nup,0]
        bondid += 1

        nright = neighs[2, src]
        bonds[bondid,:] .= [src,nright,0]
        bondid += 1
    end

    return SquareLattice(L, sites, neighs, neighs_cartesian, lattice, n_bonds, bonds)
end

function build_neighbortable(::Type{SquareLattice}, lattice, L)
    up = circshift(lattice,(-1,0))
    right = circshift(lattice,(0,-1))
    down = circshift(lattice,(1,0))
    left = circshift(lattice,(0,1))
    neighs = vcat(up[:]',right[:]',down[:]',left[:]')

    neighs_cartesian = Array{Int, 3}(undef, 4, L, L)
    neighs_cartesian[1,:,:] = up
    neighs_cartesian[2,:,:] = right
    neighs_cartesian[3,:,:] = down
    neighs_cartesian[4,:,:] = left
    return neighs, neighs_cartesian
end

# Implement AbstractLattice interface: mandatory
@inline Base.length(s::SquareLattice) = s.sites

# Implement AbstractLattice interface: optional
@inline neighbors_lookup_table(s::SquareLattice) = copy(s.neighs)

# HasNeighborsTable and HasBondsTable traits
has_neighbors_table(::SquareLattice) = HasNeighborsTable()
has_bonds_table(::SquareLattice) = HasBondsTable()