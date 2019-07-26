"""
Two dimensional square lattice.
"""
mutable struct SquareLattice <: AbstractCubicLattice
    L::Int
    sites::Int
    neighs::Matrix{Int} # row = up, right, down, left; col = siteidx
    neighs_cartesian::Array{Int, 3} # row (1) = up, right, down, left; cols (2,3) = cartesian siteidx
    lattice::Matrix{Int}

    # for generic checkerboard decomposition
    # OPT: implement Assaad's two-group square lattice version
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type

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
    l.lattice = convert(Array, reshape(1:l.L^2, (l.L, l.L)))
    build_neighbortable!(l)

    # for generic checkerboard decomposition
    l.n_bonds = 2*l.sites
    l.bonds = zeros(l.n_bonds, 3)
    bondid = 1
    for src in l.lattice
        nup = l.neighs[1, src]
        l.bonds[bondid,:] .= [src,nup,0]
        bondid += 1

        nright = l.neighs[2, src]
        l.bonds[bondid,:] .= [src,nright,0]
        bondid += 1
    end

    return l
end

function build_neighbortable!(l::SquareLattice)
    up = circshift(l.lattice,(-1,0))
    right = circshift(l.lattice,(0,-1))
    down = circshift(l.lattice,(1,0))
    left = circshift(l.lattice,(0,1))
    l.neighs = vcat(up[:]',right[:]',down[:]',left[:]')

    l.neighs_cartesian = Array{Int, 3}(undef, 4, l.L, l.L)
    l.neighs_cartesian[1,:,:] = up
    l.neighs_cartesian[2,:,:] = right
    l.neighs_cartesian[3,:,:] = down
    l.neighs_cartesian[4,:,:] = left
end
