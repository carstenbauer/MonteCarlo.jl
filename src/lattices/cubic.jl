"""
D-dimensional cubic lattice.
"""
struct CubicLattice{T<:AbstractArray{Int}} <: AbstractCubicLattice
    L::Int
    dim::Int
    sites::Int
    neighs::Matrix{Int} # row = first half uprights, second half downlefts, D in total; col = siteidx
    lattice::T

    # generic checkerboard
    n_bonds::Int
    bonds::Matrix{Int}
end

# constructors
"""
    CubicLattice(D, L)

Create a D-dimensional cubic lattice with linear dimension `L`.
"""
function CubicLattice(D::Int, L::Int)
    sites = L^D
    lattice = convert(Array, reshape(1:sites, (fill(L, D)...,)))
    n_bonds = D * sites
    neighs = build_neighbortable(CubicLattice, lattice, D)

    bonds = Matrix{Int}(undef, n_bonds, 3)
    bondid = 1
    for src in l.lattice
        for trg in l.neighs[1:D, src]
            l.bonds[bondid, 1] = src
            l.bonds[bondid, 2] = trg
            l.bonds[bondid, 3] = 0
            bondid += 1
        end
    end

    return CubicLattice(L,D,sites,Matrix(neighs),lattice,n_bonds,bonds)
end

function build_neighbortable(::Type{CubicLattice}, lattice, D)
    uprights = Vector{Vector{Int}}(undef, D)
    downlefts = Vector{Vector{Int}}(undef, D)

    for d in 1:D
        shift = zeros(Int, D); shift[d]=-1;
        uprights[d] = circshift(lattice, (shift...,))[:]
        shift[d]=1;
        downlefts[d] = circshift(lattice, (shift...,))[:]
    end

    return transpose(hcat(uprights..., downlefts...))
end

@inline nsites(c::CubicLattice) = c.sites
@inline neighbors_lookup_table(c::CubicLattice) = c.neighs