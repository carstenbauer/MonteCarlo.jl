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






# AbstractLattice interface: optional
"""
    neighbors(l::AbstractLattice, drop_repeats::Val{true})
    neighbors(l::AbstractLattice, drop_repeats::Val{false})

Returns an iterator over bonds, given as tuples (source index, target index). If
`drop_repeats = Val(false)` bonds are assumed to be directed, i.e. both
`(1, 2)` and `(2, 1)` are included. If `drop_repeats = Val(true)` bonds are
assumed to be undirected, i.e. `(1, 2)` and `(2, 1)` are assumed to be
equivalent and only one of them will be counted.
"""
@inline function neighbors(l::AbstractLattice, drop_repeats::Val{false})
    (
        [source, target] for source in 1:length(l)
            for target in l.neighs[:, source]
    )
end
@inline function neighbors(l::AbstractLattice, drop_repeats::Val{true})
    (
        l.bonds[i, 1:2] for i in 1:size(l.bonds, 1)
    )
end
"""
    neighbors(l::AbstractLattice, site_index::Integer)

Returns a list of site indices neighboring the specified `site_index`.
"""
@inline function neighbors(l::AbstractLattice, site_index::Integer)
    l.neighs[:, site_index]
end


# Typically, you also want to implement

#     - `neighbors_lookup_table(lattice)`: return a neighbors matrix where
#                                         row = neighbors and col = siteidx.




"""
    AbstractCubicLattice

AbstractCubicLattice includes 1D Chains, 2D square lattices and ND cubic
lattices.
"""
abstract type AbstractCubicLattice <: AbstractLattice end
