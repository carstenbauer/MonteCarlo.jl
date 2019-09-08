"""
Abstract definition of a lattice.
"""
abstract type AbstractLattice end

# AbstractLattice interface: mandatory
# TODO: This needs to be updated. There must be a general way to access sites and bonds.
"""
    length(l::AbstractLattice)

Number of lattice sites.
"""
@inline Base.length(l::AbstractLattice) = error("Lattice $(typeof(l)) doesn't implement `length`.")

"""
    neighbors(l::AbstractLattice[, directed=Val(false)])

Returns an iterator over bonds, given as tuples (source index, target index). If
`directed = Val(true)` bonds are assumed to be directed, i.e. both
`(1, 2)` and `(2, 1)` are included. If `directed = Val(false)` bonds are
assumed to be undirected, i.e. `(1, 2)` and `(2, 1)` are assumed to be
equivalent and only one of them will be included.
"""
@inline neighbors(l::AbstractLattice) = neighbors(l::AbstractLattice, Val(false))
@inline neighbors(l::AbstractLattice, directed::Val{true}) = _neighbors(has_neighbors_table(l), l, directed)
@inline neighbors(l::AbstractLattice, directed::Val{false}) = _neighbors(has_bonds_table(l), l, directed)


# AbstractLattice interface: optional
"""
    neighbors(l::AbstractLattice, site_index::Integer)

Returns a list of site indices neighboring the specified `site_index`.
"""
@inline neighbors(l::AbstractLattice, site_index::Integer) = _neighbors(has_neighbors_table(l), l, site_index)










# Abstract trait
abstract type LatticeProperties end

"""
    HasNeighborsTable trait

Indicates that a lattice `l` has a neighbors lookup table as a field `l.neighs`.

A neighbors lookup table is a `Matrix{Int}` of size `(nneighbors, nsites)` such that
`l.neighs[i,j]` gives the site index of the `i`th neighbor of site `j`.

A lattice should implement `has_neighbors_table(::MyLattice) = HasNeighborsTable()` to indicate the trait.
"""
struct HasNeighborsTable <: LatticeProperties end

"""
    HasBondsTable trait

Indicates that a lattice `l` has a bonds table as a field `l.bonds`.

A bonds table is a `Matrix{Int}` of size `(nbonds, 3)` where the columns mean
`src`, `trg`, `type`, respectively.

A lattice should implement `has_bonds_table(::MyLattice) = HasBondsTable()` to indicate the trait.
"""
struct HasBondsTable <: LatticeProperties end

has_neighbors_table(::AbstractLattice) = nothing
has_bonds_table(::AbstractLattice) = nothing





# neighbors(lattice, directed::Val)
@inline _neighbors(::Nothing, l::LT, directed::V) where {LT<:AbstractLattice, V<:Val} =
    error("Lattice $(LT) doesn't implement `neighbors(::$(LT), ::$V`")
@inline function _neighbors(::HasNeighborsTable, l::AbstractLattice, directed::Val{true})
    (
        [src, trg] for src in 1:length(l)
            for trg in l.neighs[:, src]
    )
end
@inline function _neighbors(::HasBondsTable, l::AbstractLattice, directed::Val{false})
    (
        l.bonds[i, 1:2] for i in 1:size(l.bonds, 1)
    )
end

# neighbors(lattice, site_index)
@inline _neighbors(::Nothing, l::LT, site_index::Integer) where {LT<:AbstractLattice} =
    error("Lattice $(LT) doesn't implement `neighbors(::$(LT), site_index`.")
@inline function _neighbors(::HasNeighborsTable, l::AbstractLattice, site_index::Integer)
    l.neighs[:, site_index]
end
