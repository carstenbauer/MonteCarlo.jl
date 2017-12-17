"""
Two dimensional square lattice.
"""
mutable struct SquareLattice <: CubicLattice
    L::Int
    sites::Int
    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
    sql::Matrix{Int}

    SquareLattice() = new()
end

# constructors
"""
    SquareLattice(L)

Create a square lattice with linear dimension `L`.
"""
function SquareLattice(L::Int)
    l = SquareLattice()
    l.L = L
    l.sites = l.L^2
    l.sql = convert(Array, reshape(1:l.L^2, (l.L, l.L)))
    build_neighbortable!(l)
    return l
end

function build_neighbortable!(l::SquareLattice)
    up = circshift(l.sql,(-1,0))
    right = circshift(l.sql,(0,-1))
    down = circshift(l.sql,(1,0))
    left = circshift(l.sql,(0,1))
    l.neighs = vcat(up[:]',right[:]',down[:]',left[:]')

    l.neighs_cartesian = Array{Int, 3}(4, l.L, l.L)
    l.neighs_cartesian[1,:,:] = up
    l.neighs_cartesian[2,:,:] = right
    l.neighs_cartesian[3,:,:] = down
    l.neighs_cartesian[4,:,:] = left
end
