"""
Two dimensional Honeycomb lattice with nearest and next nearest neighbors.
"""
mutable struct HoneycombLattice <: AbstractLattice
    L::Int
    sites::Int
    # row = up, down, left/right; col = siteidx
    neighs::Matrix{Int}
    # row (1) = up, down, left/right; cols (2,3) = cartesian siteidx
    neighs_cartesian::Array{Int, 3}
    # 2nd neighbors
    NNNs::Matrix{Int}
    NNNs_cartesian::Array{Int, 3}

    lattice::Matrix{Int}

    # for generic checkerboard decomposition
    # OPT: implement Assaad's two-group square lattice version
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type

    HoneycombLattice() = new()
end

# constructors
"""
    HoneycombLattice(L)

Create a Honeycomb lattice with linear dimension `L`. Note that this generates
(2L)² sites as the lattice is bipartite.
"""
function HoneycombLattice(L::Int, include_NNN=false)
    l = HoneycombLattice()
    l.L = L
    l.sites = (2l.L)^2
    l.lattice = convert(Array, reshape(1:(2l.L)^2, (2l.L, 2l.L)))
    build_neighbortable!(l)
    include_NNN && build_NNneighbortable!(l)

    # for generic checkerboard decomposition
    l.n_bonds = include_NNN ? div(3l.sites, 2) + 3l.sites : div(3l.sites, 2)
    l.bonds = zeros(l.n_bonds, 3)
    bondid = 1

    for i in 1:2l.L, j in 1:2l.L
        src = l.lattice[i, j]
        nup = l.neighs[1, src]
        l.bonds[bondid,:] .= [src,nup,0]
        bondid += 1

        if isodd(i+j)
            nright = l.neighs[3, src]
            l.bonds[bondid,:] .= [src,nright,0]
            bondid += 1
        end

        if include_NNN
            upup        = l.NNNs[1, src]
            l.bonds[bondid,:] .= [src, upup, 0]
            bondid += 1
            upright     = l.NNNs[2, src]
            l.bonds[bondid,:] .= [src, upright, 0]
            bondid += 1
            downright   = l.NNNs[3, src]
            l.bonds[bondid,:] .= [src, downright, 0]
            bondid += 1
        end
    end

    return l
end

function build_neighbortable!(l::HoneycombLattice)
    up = circshift(l.lattice,(-1,0))
    right = circshift(l.lattice,(0,-1))
    down = circshift(l.lattice,(1,0))
    left = circshift(l.lattice,(0,1))
    alternating_left_right = [
        iseven(i+j) ? left[j, i] : right[j, i] for i in 1:2l.L for j in 1:2l.L
    ]
    l.neighs = vcat(up[:]', down[:]', alternating_left_right[:]')

    l.neighs_cartesian = Array{Int, 3}(undef, 3, 2l.L, 2l.L)
    l.neighs_cartesian[1,:,:] = up
    l.neighs_cartesian[2,:,:] = down
    l.neighs_cartesian[3,:,:] = alternating_left_right
end

function build_NNneighbortable!(l::HoneycombLattice)
    upup = circshift(l.lattice,(-2,0))
    upright = circshift(l.lattice,(-1,-1))
    upleft = circshift(l.lattice,(-1,1))
    downdown = circshift(l.lattice,(2,0))
    downright = circshift(l.lattice,(1,-1))
    downleft = circshift(l.lattice,(1,1))

    l.NNNs = vcat(
        upup[:]',
        upright[:]',
        downright[:]',
        downdown[:]',
        downleft[:]',
        upleft[:]'
    )

    l.NNNs_cartesian = Array{Int, 3}(undef, 6, 2l.L, 2l.L)
    l.NNNs_cartesian[1,:,:] = upup
    l.NNNs_cartesian[2,:,:] = upright
    l.NNNs_cartesian[3,:,:] = downright
    l.NNNs_cartesian[4,:,:] = downdown
    l.NNNs_cartesian[5,:,:] = downleft
    l.NNNs_cartesian[6,:,:] = upleft
end

@inline nsites(l::HoneycombLattice) = l.sites