struct LatPhysLattice{LT <: LatPhysBase.AbstractLattice} <: AbstractLattice
    lattice::LT
    neighs::Matrix{Int}
end

function LatPhysLattice(lattice::LatPhysBase.AbstractLattice)
    # Build lookup table for neighbors
    # neighs[:, site] = list of neighoring site indices
    nested_bonds = [Int[] for _ in 1:numSites(lattice)]
    for b in bonds(lattice)
        push!(nested_bonds[from(b)], to(b))
    end
    max_bonds = maximum(length(x) for x in nested_bonds)

    neighs = fill(-1, max_bonds, numSites(lattice))
    for (src, targets) in enumerate(nested_bonds)
        for (idx, trg) in enumerate(targets)
            neighs[idx, src] = trg
        end
    end
    any(x -> x == -1, neighs) && @warn(
        "neighs is padded with -1 to indicated the lack of a bond. This is " *
        "due to the lattice having an irregular number of bonds per site."
    )
    LatPhysLattice(lattice, neighs)
end

@inline Base.length(l::LatPhysLattice) = LatPhysBase.numSites(l.lattice)
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
