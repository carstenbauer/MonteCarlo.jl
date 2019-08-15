"""
One dimensional chain.
"""
struct Chain <: AbstractCubicLattice
    sites::Int
    neighs::Matrix{Int} # row = right, left; col = siteidx

    # for generic checkerboard decomposition
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

# constructors
"""
    Chain(nsites::Int)

Create a chain with `nsites`.
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

@inline nsites(c::Chain) = c.sites
@inline neighbors_lookup_table(c::Chain) = c.neighs
