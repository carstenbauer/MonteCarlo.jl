struct LatPhysLattice{LT <: LatPhysBase.AbstractLattice} <: AbstractLattice
    lattice::LT
    neighs::Matrix{Int}
end

function LatPhysLattice(lattice::LatPhysBase.AbstractLattice)
    # Build lookup table for neighbors
    # neighs[:, site] = list of neighoring site indices
    bs = sort(LatPhysBase.bonds(lattice), by = LatPhysBase.from)
    bonds_per_site = findfirst(b -> LatPhysBase.from(b) > 1, bs) - 1
    neighs = Matrix{Int}(undef, bonds_per_site, LatPhysBase.numSites(lattice))
    for site_idx in 1:LatPhysBase.numSites(lattice)
        for bond_idx in 1:bonds_per_site
            bond = bs[bonds_per_site * (site_idx - 1) + bond_idx]
            @assert from(bond) == site_idx
            neighs[bond_idx, site_idx] = to(bond)
        end
    end

    LatPhysLattice(lattice, neighs)
end

@inline Base.length(l::LatPhysLattice) = LatPhysBase.numSites(l.lattice)
function Base.size(l::LatPhysLattice)
    N = LatPhysBase.numSites(l.lattice)
    D = ndims(l)
    @warn "Guessing size of LatPhys Lattice."
    if N % D == 0
        return tuple((div(N, D) for _ in 1:D)...)
    else
        error("Failed to guess size of LatPhys Lattice.")
    end
end

@inline function neighbors(l::LatPhysLattice, directed::Val{true})
    ((LatPhysBase.from(b), LatPhysBase.to(b)) for b in LatPhysBase.bonds(l))
end
@inline function neighbors(l::LatPhysLattice, directed::Val{false})
    (
        (LatPhysBase.from(b), LatPhysBase.to(b))
        for b in LatPhysBase.bonds(l)
        if LatPhysBase.from(b) < LatPhysBase.to(b)
    )
end
@inline function neighbors(l::LatPhysLattice, site_index::Integer)
    l.neighs[:, site_index]
end

@inline neighbors_lookup_table(l::LatPhysLattice) = copy(l.neighs)
