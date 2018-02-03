"""
D-dimensional cubic lattice.
"""
mutable struct DCubicLattice{T<:AbstractArray{Int}} <: CubicLattice
    L::Int
    D::Int
    sites::Int
    neighs::Matrix{Int} # row = first half uprights, second half downlefts, D in total; col = siteidx
    lattice::T
end

# constructors
"""
    DCubicLattice(D, L)

Create a D-dimensional cubic lattice with linear dimension `L`.
"""
function DCubicLattice(D::Int, L::Int)
    sites = L^D
    lattice = convert(Array, reshape(1:sites, (fill(L, D)...)))

    l = DCubicLattice{Array{Int, D}}(L, D, sites, zeros(Int, 1,1), lattice)
    build_neighbortable!(l)
    return l
end

function build_neighbortable!(l::DCubicLattice{T}) where T

    uprights = Vector{Vector{Int}}(l.D)
    downlefts = Vector{Vector{Int}}(l.D)

    for d in 1:l.D
        shift = zeros(Int, l.D); shift[d]=-1;
        uprights[d] = circshift(l.lattice, (shift...))[:]
        shift[d]=1;
        downlefts[d] = circshift(l.lattice, (shift...))[:]
    end

    l.neighs = transpose(hcat(uprights..., downlefts...))
    nothing
end
