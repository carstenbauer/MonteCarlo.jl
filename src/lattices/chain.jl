"""
One dimensional chain.
"""
mutable struct Chain <: AbstractCubicLattice
    sites::Int
    neighs::Matrix{Int} # row = right, left; col = siteidx

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
    return l
end

function build_neighbortable!(l::Chain)
    c = 1:l.sites
    right = circshift(c,-1)
    left = circshift(c,1)
    l.neighs = vcat(right[:]',left[:]')
end
