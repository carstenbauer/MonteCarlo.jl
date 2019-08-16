"""
Two dimensional Honeycomb lattice with nearest and next nearest neighbors.
"""
struct HoneycombLattice <: AbstractLattice
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
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

# constructors
"""
    HoneycombLattice(L)

Create a Honeycomb lattice with linear dimension `L`. Note that this generates
(2*L)² sites as the lattice is bipartite.
"""
function HoneycombLattice(L::Int, include_NNN=false)
    sites = (2*L)^2
    lattice = convert(Array, reshape(1:(2*L)^2, (2*L, 2*L)))
    neighs, neighs_cartesian = build_neighbortable(HoneycombLattice, lattice, L)
    if include_NNN
        NNNs, NNNs_cartesian = build_NNneighbortable(HoneycombLattice, lattice, L)
    else
        NNNs = zeros(Int, 0,0)
        NNNs_cartesian = zeros(Int, 0,0,0)
    end

    # for generic checkerboard decomposition
    n_bonds = include_NNN ? div(3*sites, 2) + 3*sites : div(3*sites, 2)
    bonds = zeros(n_bonds, 3)
    bondid = 1

    for i in 1:2*L, j in 1:2*L
        src = lattice[i, j]
        nup = neighs[1, src]
        bonds[bondid,:] .= [src,nup,0]
        bondid += 1

        if isodd(i+j)
            nright = neighs[3, src]
            bonds[bondid,:] .= [src,nright,0]
            bondid += 1
        end

        if include_NNN
            upup        = NNNs[1, src]
            bonds[bondid,:] .= [src, upup, 0]
            bondid += 1
            upright     = NNNs[2, src]
            bonds[bondid,:] .= [src, upright, 0]
            bondid += 1
            downright   = NNNs[3, src]
            bonds[bondid,:] .= [src, downright, 0]
            bondid += 1
        end
    end

    return HoneycombLattice(L,sites,neighs,neighs_cartesian,NNNs,NNNs_cartesian,lattice,n_bonds,bonds)
end

function build_neighbortable(::Type{HoneycombLattice}, lattice, L)
    up = circshift(lattice,(-1,0))
    right = circshift(lattice,(0,-1))
    down = circshift(lattice,(1,0))
    left = circshift(lattice,(0,1))
    alternating_left_right = [
        iseven(i+j) ? left[j, i] : right[j, i] for i in 1:2*L for j in 1:2*L
    ]
    neighs = vcat(up[:]', down[:]', alternating_left_right[:]')

    neighs_cartesian = Array{Int, 3}(undef, 3, 2*L, 2*L)
    neighs_cartesian[1,:,:] = up
    neighs_cartesian[2,:,:] = down
    neighs_cartesian[3,:,:] = alternating_left_right

    return neighs, neighs_cartesian
end

function build_NNneighbortable(::Type{HoneycombLattice}, lattice, L)
    upup = circshift(lattice,(-2,0))
    upright = circshift(lattice,(-1,-1))
    upleft = circshift(lattice,(-1,1))
    downdown = circshift(lattice,(2,0))
    downright = circshift(lattice,(1,-1))
    downleft = circshift(lattice,(1,1))

    NNNs = vcat(
        upup[:]',
        upright[:]',
        downright[:]',
        downdown[:]',
        downleft[:]',
        upleft[:]'
    )

    NNNs_cartesian = Array{Int, 3}(undef, 6, 2*L, 2*L)
    NNNs_cartesian[1,:,:] = upup
    NNNs_cartesian[2,:,:] = upright
    NNNs_cartesian[3,:,:] = downright
    NNNs_cartesian[4,:,:] = downdown
    NNNs_cartesian[5,:,:] = downleft
    NNNs_cartesian[6,:,:] = upleft

    return NNNs, NNNs_cartesian
end

# Implement AbstractLattice interface: mandatory
@inline Base.length(l::HoneycombLattice) = l.sites

# Implement AbstractLattice interface: optional
@inline neighbors_lookup_table(l::HoneycombLattice) = copy(l.neighs)

# HasNeighborsTable and HasBondsTable traits
has_neighbors_table(::HoneycombLattice) = HasNeighborsTable()
has_bonds_table(::HoneycombLattice) = HasBondsTable()