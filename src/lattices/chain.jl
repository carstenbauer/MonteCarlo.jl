"""
One dimensional chain.
"""
mutable struct Chain <: AbstractCubicLattice
    sites::Int
    neighs::Matrix{Int} # row = right, left; col = siteidx

    # for generic checkerboard decomposition
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type

    Chain() = new()
end

# constructors
"""
    Chain(n::Int)

Create a chain with `n` sites.
"""
function Chain(n::Int)
    l = Chain()
    l.sites = n
    build_neighbortable!(l)

    # for generic checkerboard decomposition
    l.n_bonds = l.sites
    l.bonds = zeros(l.n_bonds, 3)
    bondid = 1
    for src in 1:l.sites
        nright = l.neighs[1, src]
        l.bonds[bondid,:] .= [src,nright,0]
        bondid += 1
    end

    return l
end

function build_neighbortable!(l::Chain)
    c = 1:l.sites
    right = circshift(c,-1)
    left = circshift(c,1)
    l.neighs = vcat(right[:]',left[:]')
end

@inline nsites(c::Chain) = c.sites
