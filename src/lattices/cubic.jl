"""
D-dimensional cubic lattice.
"""
struct CubicLattice{T<:AbstractArray{Int}} <: AbstractCubicLattice
    L::Int
    D::Int
    sites::Int
    neighs::Matrix{Int} # row = first half uprights, second half downlefts, D in total; col = siteidx
    lattice::T

    #TODO: generic checkerboard
end

# constructors
"""
    CubicLattice(D, L)

Create a D-dimensional cubic lattice with linear dimension `L`.
"""
function CubicLattice(D::Int, L::Int)
    sites = L^D
    lattice = convert(Array, reshape(1:sites, (fill(L, D)...,)))

    neighs = build_neighbortable(CubicLattice, lattice, D)
    return CubicLattice(L,D,sites,Matrix(neighs),lattice)
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