"""
One dimensional chain.
"""
struct Chain <: AbstractLattice
    sites::Int
    neighs::Matrix{Int} # row = right, left; col = siteidx

    # for generic checkerboard decomposition
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

# constructors
"""
    Chain(nsites::Int)

Create a chain with `nsites` (and periodic boundary conditions).
"""
function Chain(nsites::Int)
    neighs = build_neighbortable(Chain, nsites)

    # for generic checkerboard decomposition
    n_bonds = nsites
    bonds = zeros(n_bonds, 3)
    bondid = 1
    for src in 1:nsites
        nright = neighs[1, src]
        bonds[bondid,:] .= [src,nright,0]
        bondid += 1
    end

    return Chain(nsites, neighs, n_bonds, bonds)
end

function build_neighbortable(::Type{Chain}, nsites::Int)
    c = 1:nsites
    right = circshift(c,-1)
    left = circshift(c,1)
    return vcat(right[:]',left[:]')
end

# Implement AbstractLattice interface: mandatory
@inline Base.length(c::Chain) = c.sites
@inline Base.size(c::Chain) = (c.sites,)

# Implement AbstractLattice interface: optional
@inline neighbors_lookup_table(c::Chain) = copy(c.neighs)

# HasNeighborsTable and HasBondsTable traits
has_neighbors_table(::Chain) = HasNeighborsTable()
has_bonds_table(::Chain) = HasBondsTable()
positions(l::Chain) = [[i] for i in 1:l.sites]
