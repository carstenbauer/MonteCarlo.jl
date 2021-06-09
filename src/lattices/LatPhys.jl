using LatPhysBase: from, to, numSites, bonds, sites, latticeVectors, unitcell,
                   point, AbstractSite
import LatPhysBase, LatticePhysics

struct LatPhysLattice{LT <: LatPhysBase.AbstractLattice} <: AbstractLattice
    lattice::LT
    neighs::Matrix{Int}
end
export LatPhysLattice

function LatPhysLattice(lattice::LatPhysBase.AbstractLattice)
    # Build lookup table for neighbors
    # neighs[:, site] = list of neighoring site indices
    nested_bonds = [Int[] for _ in 1:LatPhysBase.numSites(lattice)]
    for b in LatPhysBase.bonds(lattice)
        push!(nested_bonds[LatPhysBase.from(b)], LatPhysBase.to(b))
    end
    max_bonds = maximum(length(x) for x in nested_bonds)

    neighs = fill(-1, max_bonds, LatPhysBase.numSites(lattice))
    for (src, targets) in enumerate(nested_bonds)
        for (idx, trg) in enumerate(targets)
            neighs[idx, src] = trg
        end
    end
    any(x -> x == -1, neighs) && @debug(
        "neighs is padded with -1 to indicated the lack of a bond. This is " *
        "due to the lattice having an irregular number of bonds per site."
    )
    LatPhysLattice(lattice, neighs)
end

@inline Base.length(l::LatPhysLattice) = numSites(l.lattice)
@inline Base.ndims(l::LatPhysLattice) = ndims(l.lattice)
function Base.size(l::LatPhysLattice)
    lattice_extend = latticeVectors(l.lattice)
    uc_extend = latticeVectors(unitcell(l.lattice))
    Ls = map(lattice_extend, uc_extend) do lv, ucv
        scaling = lv ./ ucv
        if all(x -> x ≈ scaling[1], scaling)
            return scaling[1]
        else
            @warn "Irregular scaling of unitcell vectors $scaling"
            NaN
        end
    end

    return tuple(Int64.(Ls)...)
end

@inline function neighbors(l::LatPhysLattice, directed::Val{true})
    ((LatPhysBase.from(b), LatPhysBase.to(b)) for b in LatPhysBase.bonds(l.lattice))
end
@inline function neighbors(l::LatPhysLattice, directed::Val{false})
    (
        (LatPhysBase.from(b), LatPhysBase.to(b))
        for b in LatPhysBase.bonds(l.lattice)
        if LatPhysBase.from(b) < LatPhysBase.to(b)
    )
end
@inline function neighbors(l::LatPhysLattice, site_index::Integer)
    l.neighs[:, site_index]
end

@inline neighbors_lookup_table(l::LatPhysLattice) = copy(l.neighs)

positions(lattice::LatPhysLattice) = point.(sites(lattice.lattice))
lattice_vectors(l::LatPhysLattice) = latticeVectors(l.lattice)

function reciprocal_vectors(lattice::AbstractLattice, L)
    LatPhysBase.latticeVectors(
        LatticePhysics.getReciprocalUnitcell(
            LatPhysBase.unitcell(lattice.lattice)
        )
    )
end



################################################################################
### Saving & Loading
################################################################################



function save_lattice(file::JLDFile, lattice::LatPhysLattice, entryname::String)
    write(file, entryname * "/VERSION", 0)
    write(file, entryname * "/type", typeof(lattice))
    _save_lattice(file, lattice.lattice, entryname * "/lattice")
    write(file, entryname * "/neighs", lattice.neighs)
    nothing
end
function _save_lattice(file::JLDFile, lattice::LatPhysBase.AbstractLattice, entryname::String)
    write(file, entryname * "/type", typeof(lattice))
    write(file, entryname * "/lv/N", length(lattice.lattice_vectors))
    for (i, v) in enumerate(lattice.lattice_vectors)
        write(file, entryname * "/lv/$i", v)
    end

    write(file, entryname * "/sites/type", typeof(lattice.sites[1])) # eltype of lattice
    write(file, entryname * "/sites/points", [s.point for s in lattice.sites])
    write(file, entryname * "/sites/labels", [s.label for s in lattice.sites])

    write(file, entryname * "/bonds/type", typeof(lattice.bonds[1])) # eltype
    write(file, entryname * "/bonds/froms",  [b.from for b in lattice.bonds])
    write(file, entryname * "/bonds/tos",    [b.to for b in lattice.bonds])
    write(file, entryname * "/bonds/labels", [b.label for b in lattice.bonds])
    write(file, entryname * "/bonds/wraps",  [b.wrap for b in lattice.bonds])

    save_unitcell(file, lattice.unitcell, entryname * "/unitcell")
    nothing
end
function save_unitcell(file::JLDFile, uc::LatPhysBase.AbstractUnitcell, entryname::String)
    write(file, entryname * "/type", typeof(uc))
    write(file, entryname * "/lv", uc.lattice_vectors)
    
    write(file, entryname * "/sites/type", typeof(uc.sites[1])) # eltype of lattice
    write(file, entryname * "/sites/points", [s.point for s in uc.sites])
    write(file, entryname * "/sites/labels", [s.label for s in uc.sites])

    write(file, entryname * "/bonds/type", typeof(uc.bonds[1])) # eltype
    write(file, entryname * "/bonds/froms",  [b.from  for b in uc.bonds])
    write(file, entryname * "/bonds/tos",    [b.to    for b in uc.bonds])
    write(file, entryname * "/bonds/labels", [b.label for b in uc.bonds])
    write(file, entryname * "/bonds/wraps",  [b.wrap  for b in uc.bonds])
    nothing
end


function _load(data, ::Type{T}) where T <: LatPhysLattice
    @assert data["VERSION"] == 0
    data["type"](load_lattice(data["lattice"]), data["neighs"])
end
function load_lattice(data)
    sites = load_sites(data["sites"])
    bonds = load_bonds(data["bonds"])
    lattice_vectors = [data["lv"]["$i"] for i in 1:data["lv"]["N"]]
    unitcell = load_unitcell(data["unitcell"])

    data["type"](lattice_vectors, sites, bonds, unitcell)
end
function load_unitcell(data)
    sites = load_sites(data["sites"])
    bonds = load_bonds(data["bonds"])
    lattice_vectors = data["lv"]
    data["type"](lattice_vectors, sites, bonds)
end
load_sites(data) = data["type"].(data["points"], data["labels"])
load_bonds(data) = data["type"].(data["froms"], data["tos"], data["labels"], data["wraps"])



################################################################################
### Extensions
################################################################################



function to_angle(v)
    angle = acos(v[1] / norm(v))
    (v[2] ≥ 0) ? angle : -angle
end

function get_sorted_corners(lattice::LatPhysLattice)
    bz = LatticePhysics.getBrillouinZone(
        LatticePhysics.getReciprocalUnitcell(
            LatticePhysics.unitcell(lattice.lattice)
        )
    )
    cs = sort(LatticePhysics.corners(bz), by = to_angle)
end



################################################################################
### Deprecated
################################################################################



function DistanceMask(lattice::LatPhysLattice)
    wrap = generate_combinations(latticeVectors(lattice.lattice))
    VerboseDistanceMask(lattice, wrap)
end

function directions(mask::VerboseDistanceMask, lattice::LatPhysLattice)
    pos = positions(lattice)
    dirs = [pos[src] - pos[trg] for (src, trg) in first.(mask.targets)]
    wrap = generate_combinations(latticeVectors(lattice.lattice))
    map(dirs) do _d
        d = round.(_d .+ wrap[1], digits=8)
        for v in wrap[2:end]
            new_d = round.(_d .+ v, digits=8)
            if directed_norm(new_d, 1e-6) < directed_norm(d, 1e-6)
                d .= new_d
            end
        end
        d
    end
end