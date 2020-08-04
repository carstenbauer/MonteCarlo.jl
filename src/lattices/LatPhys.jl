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



function generate_combinations(vs::Vector{Vector{Float64}})
    out = [zeros(length(vs[1]))]
    for v in vs
        out = vcat([e.-v for e in out], out, [e.+v for e in out])
    end
    out
end


# function DistanceMask(lattice::LatPhysLattice)
#     if length(sites(unitcell(lattice.lattice))) > 1
#         error("Lattices with a basis are currently not supported!")
#     end
#     targets = Array{Int64}(undef, length(lattice), length(lattice))
#     positions = point.(sites(lattice.lattice))
#     wrap = generate_combinations(latticeVectors(lattice.lattice))
#
#     for origin in 1:length(lattice)
#         dist_vecs = map(positions) do p
#             # Rounding is necessary to get consistency
#             d = round.(positions[origin] .- p .+ wrap[1], digits=6)
#             for v in wrap[2:end]
#                 new_d = round.(positions[origin] .- p .+ v, digits=6)
#                 if norm(new_d) < norm(d)
#                     d .= new_d
#                 end
#             end
#             d
#         end
#         idxs = collect(eachindex(dist_vecs))
#         for j in 1:length(dist_vecs[1])
#             sort!(idxs, by = i -> dist_vecs[i][j], alg=MergeSort)
#         end
#         targets[origin, :] .= sort(idxs, by = i -> norm(dist_vecs[i]), alg=MergeSort)
#     end
#
#     DistanceMask(targets)
# end


function DistanceMask(lattice::LatPhysLattice)
    positions = point.(sites(lattice.lattice))
    wrap = generate_combinations(latticeVectors(lattice.lattice))

    directions = Vector{Float64}[]
    # distance_idx, src, trg
    bonds = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]

    for origin in 1:length(lattice)
        for (trg, p) in enumerate(positions)
            d = round.(positions[origin] .- p .+ wrap[1], digits=6)
            for v in wrap[2:end]
                new_d = round.(positions[origin] .- p .+ v, digits=6)
                if norm(new_d) < norm(d)
                    d .= new_d
                end
            end
            # I think the rounding will allow us to use == here
            idx = findfirst(dir -> dir == d, directions)
            if idx == nothing
                push!(directions, d)
                push!(bonds[origin], (length(directions), trg))
            else
                push!(bonds[origin], (idx, trg))
            end
        end
    end

    targets = Array{Tuple{Int64, Int64}}(undef, length(lattice), length(lattice))
    temp = sortperm(directions, by=norm)
    sorted = Vector{Int64}(undef, length(directions))
    sorted[temp] .= eachindex(directions)

    for (src, bs) in enumerate(bonds)
        targets[src, :] = map(sort(bs, by = t -> norm(directions[t[1]]))) do b
            (sorted[b[1]], b[2])
        end
    end

    VerboseDistanceMask(targets)
end



# Saving & Loading


function save_lattice(file::JLD.JldFile, lattice::LatPhysLattice, entryname::String)
    write(file, entryname * "/VERSION", 0)
    write(file, entryname * "/type", typeof(lattice))
    _save_lattice(file, lattice.lattice, entryname * "/lattice")
    write(file, entryname * "/neighs", lattice.neighs)
    nothing
end
function _save_lattice(file::JLD.JldFile, lattice::LatPhysBase.AbstractLattice, entryname::String)
    write(file, entryname * "/type", typeof(lattice))
    write(file, entryname * "/lv/N", length(lattice.lattice_vectors))
    for (i, v) in enumerate(lattice.lattice_vectors)
        write(file, entryname * "/lv/$i", v)
    end
    write(file, entryname * "/sites/N", length(lattice.sites))
    for (i, s) in enumerate(lattice.sites)
        save_site(file, s, entryname * "/sites/$i")
    end
    write(file, entryname * "/bonds/N", length(lattice.bonds))
    for (i, b) in enumerate(lattice.bonds)
        save_bond(file, b, entryname * "/bonds/$i")
    end
    save_unitcell(file, lattice.unitcell, entryname * "/unitcell")
    nothing
end
function save_unitcell(file::JLD.JldFile, uc::LatPhysBase.AbstractUnitcell, entryname::String)
    write(file, entryname * "/type", typeof(uc))
    write(file, entryname * "/lv/N", length(uc.lattice_vectors))
    for (i, v) in enumerate(uc.lattice_vectors)
        write(file, entryname * "/lv/$i", v)
    end
    write(file, entryname * "/sites/N", length(uc.sites))
    for (i, s) in enumerate(uc.sites)
        save_site(file, s, entryname * "/sites/$i")
    end
    write(file, entryname * "/bonds/N", length(uc.bonds))
    for (i, b) in enumerate(uc.bonds)
        save_bond(file, b, entryname * "/bonds/$i")
    end
    nothing
end
function save_bond(file::JLD.JldFile, b::LatPhysBase.AbstractBond, entryname::String)
    write(file, entryname * "/type", typeof(b))
    write(file, entryname * "/from", b.from)
    write(file, entryname * "/to", b.to)
    write(file, entryname * "/label", b.label)
    write(file, entryname * "/wrap", b.wrap)
    nothing
end
function save_site(file::JLD.JldFile, s::LatPhysBase.AbstractSite, entryname::String)
    write(file, entryname * "/type", typeof(s))
    write(file, entryname * "/point", s.point)
    write(file, entryname * "/label", s.label)
    nothing
end


function load_lattice(data, ::Type{T}) where T <: LatPhysLattice
    @assert data["VERSION"] == 0
    data["type"](load_lattice(data["lattice"]), data["neighs"])
end


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
