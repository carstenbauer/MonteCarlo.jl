using LatPhysBase: from, to, numSites, bonds, sites, latticeVectors, unitcell,
                   point, AbstractSite
import LatPhysBase

struct LatPhysLattice{LT <: LatPhysBase.AbstractLattice} <: AbstractLattice
    lattice::LT
    neighs::Matrix{Int}
end

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
    any(x -> x == -1, neighs) && @warn(
        "neighs is padded with -1 to indicated the lack of a bond. This is " *
        "due to the lattice having an irregular number of bonds per site."
    )
    LatPhysLattice(lattice, neighs)
end

@inline Base.length(l::LatPhysLattice) = numSites(l.lattice)
@inline Base.ndims(l::LatPhysLattice) = ndims(l.lattice)
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

function DistanceMask(lattice::LatPhysLattice)
    wrap = generate_combinations(latticeVectors(lattice.lattice))
    VerboseDistanceMask(lattice, wrap)
end

function directions(mask::VerboseDistanceMask, lattice::LatPhysLattice)
    pos = MonteCarlo.positions(lattice)
    dirs = [pos[trg] - pos[src] for (src, trg) in first.(mask.targets)]
    # marked = Set{Int64}()
    # dirs = Vector{eltype(pos)}(undef, maximum(first(x) for x in mask.targets))
    # for src in 1:size(mask.targets, 1)
    #     for (idx, trg) in mask.targets[src, :]
    #         if !(idx in marked)
    #             push!(marked, idx)
    #             dirs[idx] = pos[trg] - pos[src]
    #         end
    #     end
    # end
    wrap = MonteCarlo.generate_combinations(latticeVectors(lattice.lattice))
    map(dirs) do _d
        d = round.(_d .+ wrap[1], digits=6)
        for v in wrap[2:end]
            new_d = round.(_d .+ v, digits=6)
            if norm(new_d) < norm(d)
                d .= new_d
            end
        end
        d
    end
end


# Saving & Loading


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
# function save_bond(file::JLDFile, b::LatPhysBase.AbstractBond, entryname::String)
#     write(file, entryname * "/type", typeof(b))
#     write(file, entryname * "/from", b.from)
#     write(file, entryname * "/to", b.to)
#     write(file, entryname * "/label", b.label)
#     write(file, entryname * "/wrap", b.wrap)
#     nothing
# end
# function save_site(file::JLDFile, s::LatPhysBase.AbstractSite, entryname::String)
#     write(file, entryname * "/type", typeof(s))
#     write(file, entryname * "/point", s.point)
#     write(file, entryname * "/label", s.label)
#     nothing
# end


function _load(data, ::Type{T}) where T <: LatPhysLattice
    @assert data["VERSION"] == 0
    data["type"](load_lattice(data["lattice"]), data["neighs"])
end


# NEW LOAD
# function load_lattice(data)
#     sites = load_sites(data["sites"])
#     bonds = load_bonds(data["bonds"])
#     lattice_vectors = [data["lv"]["$i"] for i in 1:data["lv"]["N"]]
#     unitcell = load_unitcell(data["unitcell"])

#     data["type"](lattice_vectors, sites, bonds, unitcell)
# end
# function load_unitcell(data)
#     sites = load_sites(data["sites"])
#     bonds = load_bonds(data["bonds"])
#     lattice_vectors = data["lv"]
#     data["type"](lattice_vectors, sites, bonds)
# end
# load_sites(data) = data["type"].(data["points"], data["labels"])
# load_bonds(data) = data["type"].(data["froms"], data["tos"], data["labels"], data["wraps"])


function load_lattice(data)
    sites = [load_site(data["sites"]["$i"]) for i in 1:data["sites"]["N"]]
    bonds = [load_bond(data["bonds"]["$i"]) for i in 1:data["bonds"]["N"]]
    lattice_vectors = [data["lv"]["$i"] for i in 1:data["lv"]["N"]]
    unitcell = load_unitcell(data["unitcell"])

    data["type"](lattice_vectors, sites, bonds, unitcell)
end
function load_unitcell(data)
    sites = [load_site(data["sites"]["$i"]) for i in 1:data["sites"]["N"]]
    bonds = [load_bond(data["bonds"]["$i"]) for i in 1:data["bonds"]["N"]]
    lattice_vectors = [data["lv"]["$i"] for i in 1:data["lv"]["N"]]

    data["type"](lattice_vectors, sites, bonds)
end
load_bond(data) = data["type"](data["from"], data["to"], data["label"], data["wrap"])
load_site(data) = data["type"](data["point"], data["label"])
