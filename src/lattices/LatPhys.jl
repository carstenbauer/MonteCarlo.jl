using LatPhysBase: from, to, numSites, bonds, sites, latticeVectors, unitcell, point
import LatPhysBase

struct LatPhysLattice{LT <: LatPhysBase.AbstractLattice} <: AbstractLattice
    lattice::LT
    neighs::Matrix{Int}
end

function LatPhysLattice(lattice::LatPhysBase.AbstractLattice)
    # Build lookup table for neighbors
    # neighs[:, site] = list of neighoring site indices
    bs = sort(bonds(lattice), by = from)
    bonds_per_site = findfirst(b -> from(b) > 1, bs) - 1
    neighs = Matrix{Int}(undef, bonds_per_site, numSites(lattice))
    for site_idx in 1:numSites(lattice)
        for bond_idx in 1:bonds_per_site
            bond = bs[bonds_per_site * (site_idx - 1) + bond_idx]
            @assert from(bond) == site_idx
            neighs[bond_idx, site_idx] = to(bond)
        end
    end

    LatPhysLattice(lattice, neighs)
end

@inline Base.length(l::LatPhysLattice) = numSites(l.lattice)
@inline function Base.ndims(::LatPhysLattice{LT}) where {
        T, D, S <: AbstractSite{T, D}, LT <: LatPhysBase.AbstractLattice{S}
    }
    D
end
function Base.size(l::LatPhysLattice)
    N = numSites(l.lattice)
    D = ndims(l)
    @warn "Guessing size of LatPhys Lattice."
    if N % D == 0
        return tuple((div(N, D) for _ in 1:D)...)
    else
        error("Failed to guess size of LatPhys Lattice.")
    end
end

@inline function neighbors(l::LatPhysLattice, directed::Val{true})
    ((from(b), to(b)) for b in bonds(l))
end
@inline function neighbors(l::LatPhysLattice, directed::Val{false})
    (
        (from(b), to(b))
        for b in bonds(l)
        if from(b) < to(b)
    )
end
@inline function neighbors(l::LatPhysLattice, site_index::Integer)
    l.neighs[:, site_index]
end

@inline neighbors_lookup_table(l::LatPhysLattice) = copy(l.neighs)


function generate_combinations(vs::Vector{Vector{Float64}})
    out = [zeros(length(vs[1]))]
    for v in vs
        out = vcat([e.-v for e in out], out, [e.+v for e in out])
    end
    out
end


function DistanceMask(lattice::LatPhysLattice)
    targets = Array{Int64}(undef, length(lattice), length(lattice))
    positions = point.(sites(lattice.lattice))
    wrap = generate_combinations(latticeVectors(lattice.lattice))

    for origin in 1:length(lattice)
        dist_vecs = map(positions) do p
            d = positions[origin] .- p .+ wrap[1]
            for v in wrap[2:end]
                new_d = positions[origin] .- p .+ v
                if norm(new_d) < norm(d)
                    d .= new_d
                end
            end
            # This is necessary to get consistency
            round.(d, digits=6)
        end
        idxs = collect(eachindex(dist_vecs))
        for j in 1:length(dist_vecs[1])
            sort!(idxs, by = i -> dist_vecs[i][j], alg=MergeSort)
        end
        targets[origin, :] .= sort(idxs, by = i -> norm(dist_vecs[i]), alg=MergeSort)
    end

    DistanceMask(targets)
end
