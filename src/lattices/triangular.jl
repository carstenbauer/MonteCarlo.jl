# TODO
# remove Lx, Ly or save and use them consistently

struct TriangularLattice <: AbstractLattice
    L::Int
    sites::Int
    # row = up, down, left/right; col = siteidx
    neighs::Matrix{Int}
    # row (1) = up, down, left/right; cols (2,3) = cartesian siteidx
    neighs_cartesian::Array{Int, 3}
    # not really 2nd neighbors, but R_i + 2a_j
    ext_neighs::Matrix{Int}
    ext_neighs_cartesian::Array{Int, 3}

    lattice::Matrix{Int}
    isAsite::Vector{Bool}

    # for generic checkerboard decomposition
    n_bonds::Int
    bonds::Matrix{Int} # src, trg, type
end

function TriangularLattice(L::Int; Lx=L, Ly=L)
    sites = Lx * Ly
    lattice = convert(Array, reshape(1:sites, (Lx, Ly)))
    neighs, neighs_cartesian = build_neighbortable(TriangularLattice, lattice, Lx, Ly)
    ext_neighs, ext_neighs_cartesian = build_ext_neighbortable(
        TriangularLattice, lattice, Lx, Ly
    )

    # for generic checkerboard decomposition
    n_bonds = 6*sites
    bonds = zeros(n_bonds, 3)
    bondid = 1

    for src in lattice
        for trg in neighs[1:3, src]
            bonds[bondid, 1] = src
            bonds[bondid, 2] = trg
            bonds[bondid, 3] = 0
            bondid += 1
        end
        for trg in ext_neighs[1:3, src]
            bonds[bondid, 1] = src
            bonds[bondid, 2] = trg
            bonds[bondid, 3] = 0
            bondid += 1
        end
    end

    isAsite = [iseven(i) for i in 1:Lx, j in 1:Ly][:]

    return TriangularLattice(
        L, sites,
        neighs, neighs_cartesian,
        ext_neighs, ext_neighs_cartesian,
        lattice, isAsite, n_bonds, bonds
    )
end

function build_neighbortable(::Type{TriangularLattice}, lattice, Lx, Ly)
    up          = circshift(lattice, (-1,  0))
    upright     = circshift(lattice, (-1, -1))
    right       = circshift(lattice, ( 0, -1))
    down        = circshift(lattice, ( 1,  0))
    downleft    = circshift(lattice, ( 1,  1))
    left        = circshift(lattice, ( 0,  1))

    neighs = vcat(up[:]', upright[:]', right[:]', down[:]', downleft[:]', left[:]')

    neighs_cartesian = Array{Int, 3}(undef, 6, Lx, Ly)
    neighs_cartesian[1,:,:] = up
    neighs_cartesian[2,:,:] = upright
    neighs_cartesian[3,:,:] = right
    neighs_cartesian[4,:,:] = down
    neighs_cartesian[5,:,:] = downleft
    neighs_cartesian[6,:,:] = left

    return neighs, neighs_cartesian
end

function build_ext_neighbortable(::Type{TriangularLattice}, lattice, Lx, Ly)
    up          = circshift(lattice, (-2,  0))
    upright     = circshift(lattice, (-2, -2))
    right       = circshift(lattice, ( 0, -2))
    down        = circshift(lattice, ( 2,  0))
    downleft    = circshift(lattice, ( 2,  2))
    left        = circshift(lattice, ( 0,  2))

    ext_neighs = vcat(up[:]', upright[:]', right[:]', down[:]', downleft[:]', left[:]')

    ext_neighs_cartesian = Array{Int, 3}(undef, 6, Lx, Ly)
    ext_neighs_cartesian[1,:,:] = up
    ext_neighs_cartesian[2,:,:] = upright
    ext_neighs_cartesian[3,:,:] = right
    ext_neighs_cartesian[4,:,:] = down
    ext_neighs_cartesian[5,:,:] = downleft
    ext_neighs_cartesian[6,:,:] = left

    return ext_neighs, ext_neighs_cartesian
end

# Implement AbstractLattice interface: mandatory
@inline Base.length(l::TriangularLattice) = l.sites
@inline Base.size(l::TriangularLattice) = (l.L, l.L)

# Implement AbstractLattice interface: optional
@inline neighbors_lookup_table(l::TriangularLattice) = copy(l.neighs)

# HasNeighborsTable and HasBondsTable traits
has_neighbors_table(::TriangularLattice) = HasNeighborsTable()
has_bonds_table(::TriangularLattice) = HasBondsTable()

function positions(l::TriangularLattice)
    idxs = l.lattice |> CartesianIndices .|> Tuple .|> collect
    [[0.5, 0.8660254037844386] * idx[1] + [1, 0] * idx[2] for idx in idxs]
end
