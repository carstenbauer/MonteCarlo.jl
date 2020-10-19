#=
For the sake of DQMC measurements it's often useful to iterate through the
lattice in a specific way. Masks allow for exactly this.

For example:

Charge density correlations are given by ⟨nᵢnⱼ⟩, a N by N matrix (with N being
the numebr of lattice sites). However we are typically not interested in the
real space charge density correlation, but rather it's Fourier transform, the
charge density structure factor. It is given by
C(q) = ∑ᵢⱼ exp(im*q*(rᵢ - rⱼ)) ⟨nᵢnⱼ⟩
Given that the exponential uses the difference between two lattice positions r
we do not need to save the whole matrix ⟨nᵢnⱼ⟩. We can perform the summation
over equal rᵢ - rⱼ immediately to "compress" ⟨nᵢnⱼ⟩. To do so we need reframe
the observable as ⟨n[i]n[i+Δ(i)]⟩, where Δ[i] gives us indices ordered by
distance and direction, so that we can record
C(rι - rⱼ) = C(Δ) = ∑(Δ) ⟨n[i]n[i+Δ(i)]⟩
=#

abstract type AbstractMask end


"""
    RawMask(lattice)

Constructs a mask that returns `1:N` for each site.
"""
struct RawMask <: AbstractMask
    nsites::Int64
end
RawMask(lattice::AbstractLattice) = RawMask(length(lattice))
Base.getindex(mask::RawMask, source, ::Colon) = 1:mask.nsites
# TODO name?
getorder(mask::RawMask, source) = enumerate(1:mask.nsites)
Base.size(mask::RawMask) = (mask.nsites, mask.nsites)
Base.size(mask::RawMask, dim) = dim <= 2 ? mask.nsites : 1
function directions(mask::RawMask, lattice::AbstractLattice)
    pos = positions(lattice)
    [p2 .- p1 for p2 in pos for p1 in pos]
end


################################################################################
### dir -> (src, trg) masks
################################################################################


"""
    DistanceMask(lattice) <: AbstractMask

Constructs a mask that orders sites by distance and direction relative to each
other site.

Related Functions:
* `getorder(mask)`: Returns an iterable of (idx, src, trg) tuples, ordered by
distance. `idx` has a 1:1 correspondence with a directions, i.e. every
`(idx, trg, src)` pair which points in the same direction will have the same `idx`.
* `getdirorder(mask, idx)`: Returns all (src, trg) pairs which point in `idx`
direction.
* `directions(mask, lattice)`: Returns a list of directions which can be indexed
by the `idx` returned from `getorder` to get the corresponding direction.
* `length(mask)`: Number of unqiue directions.

Warning: For this to work correctly the lattice must provide the neighbors in
order. Furthermore each bond is assumed to be of equal length.
"""
abstract type DistanceMask <: AbstractMask end
DistanceMask(lattice::AbstractLattice) = MethodError(DistanceMask, (lattice))
DistanceMask(mc::MonteCarloFlavor, model::Model) = DistanceMask(lattice(model))

# SimpleDistanceMask deals with lattices where each distance vector exists for
# every site in the lattice (assuming periodic bonds)
struct SimpleDistanceMask <: DistanceMask
    targets::Matrix{Int64}
end
function default_distance_mask(lattice::AbstractLattice)
    targets = Array{Int64}(undef, length(lattice), length(lattice))
    for origin in 1:length(lattice)
        new_sites = [origin]
        sites = [origin]
        marked = fill(false, length(lattice))
        marked[origin] = true

        while !isempty(new_sites)
            old_sites = copy(new_sites)
            new_sites = Int64[]
            for site in old_sites
                append!(new_sites, mark_unmarked(lattice, marked, site))
            end
            append!(sites, new_sites)
        end
        targets[origin, :] .= sites
    end

    SimpleDistanceMask(targets)
end
function mark_unmarked(lattice, marked, from)
    @assert from != 0
    new_sites = Int64[]
    for to in neighbors(lattice, from)
        if !marked[to]
            marked[to] = true
            push!(new_sites, to)
        end
    end
    new_sites
end

# All (dir, src, trg) in mask
function getorder(mask::SimpleDistanceMask)
    ((dir, src, trg) for src in 1:size(mask.targets, 1)
                     for (dir, trg) in enumerate(mask.targets[src, :])
    )
end

# All (src, trg) in dir
function getdirorder(mask::SimpleDistanceMask, dir)
    ((src, mask.targets[src, dir]) for src in 1:size(mask.targets, 1))
end

# Number of source sites
nsources(mask::SimpleDistanceMask) = size(mask.targets, 1)
Base.length(mask::SimpleDistanceMask) = size(mask.targets, 1)
dirlength(mask::SimpleDistanceMask, dir_idx) = size(mask.targets, 1)
dirlengths(mask::SimpleDistanceMask) = (size(mask.targets, 1) for _ in 1:size(mask.targets, 1))

"""
    directions(mask, lattice)

Returns a vector of directions in the order emposed by the given mask.

For a `RawMask` the output directly mimics the mask, i.e.
`directions(mask, lattice)[idx]` is the direction to
`trg = mask[src, idx]`.
For a `DistanceMask` the `directions(mask, lattice)[dir_idx]` return the
direction of `getdirorder(mask, dir_idx)`.
"""
function directions(mask::SimpleDistanceMask, lattice::AbstractLattice)
    pos = positions(lattice)
    [p .- pos[1] for p in pos[mask.targets[1, :]]]
end



# VerboseDistanceMask can deal with lattices that have different bond 
# directions. 
struct VerboseDistanceMask <: DistanceMask
    targets::Vector{Vector{Tuple{Int64, Int64}}}
end
function getorder(mask::VerboseDistanceMask)
    ((dir, src, trg) for dir in eachindex(mask.targets)
                     for (src, trg) in mask.targets[dir]
    )
end
getdirorder(mask::VerboseDistanceMask, dir) = ((src, trg) for (src, trg) in mask.targets[dir])

# Number of source sites
function nsources(mask::VerboseDistanceMask)
    out = -1
    for xs in mask.targets
        out = max(out, mapreduce(first, max, xs))
    end
    out
end
Base.length(mask::VerboseDistanceMask) = length(mask.targets)
dirlength(mask::DistanceMask, dir_idx) = length(mask.targets[dir_idx])
dirlengths(mask::DistanceMask) = length.(mask.targets)

function directions(mask::VerboseDistanceMask, lattice::AbstractLattice)
    pos = MonteCarlo.positions(lattice)
    dirs = [pos[trg] - pos[src] for (src, trg) in first.(mask.targets)]
    dirs
end
# For shifting sites across periodic bounds
function generate_combinations(vs::Vector{Vector{Float64}})
    out = [zeros(length(vs[1]))]
    for v in vs
        out = vcat([e.-v for e in out], out, [e.+v for e in out])
    end
    out
end
function VerboseDistanceMask(lattice, wrap)
    _positions = positions(lattice)
    directions = Vector{Float64}[]
    # (src, trg), first index is dir, second index irrelevant
    bonds = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]

    for origin in 1:length(lattice)
        for (trg, p) in enumerate(_positions)
            d = round.(_positions[origin] .- p .+ wrap[1], digits=6)
            for v in wrap[2:end]
                new_d = round.(_positions[origin] .- p .+ v, digits=6)
                if norm(new_d) < norm(d)
                    d .= new_d
                end
            end
            # I think the rounding will allow us to use == here
            idx = findfirst(dir -> dir == d, directions)
            if idx === nothing
                push!(directions, d)
                if length(bonds) < length(directions)
                    push!(bonds, Tuple{Int64, Int64}[])
                end
                push!(bonds[length(directions)], (origin, trg))
            else
                push!(bonds[idx], (origin, trg))
            end
        end
    end

    temp = sortperm(directions, by=norm)
    VerboseDistanceMask(bonds[temp])
end



################################################################################
### src -> (dir, trg) masks (restriced)
################################################################################



"""
    RestrictedSourceMask(mask::DistanceMask, N_directions)

A RestrictedSourceMask is a mask that allows indexing `(dir, trg)` tuples by
`src`, where the number of available directions is initially restricted.

As such it implements `getorder(mask, src)`.
"""
struct RestrictedSourceMask
    targets::Vector{Vector{Tuple{Int64, Int64}}}
end

function RestrictedSourceMask(mask::DistanceMask, directions)
    targets = [Tuple{Int64, Int64}[] for _ in 1:nsources(mask)]
    for dir in 1:directions
        for (src, trg) in getdirorder(mask, dir)
            push!(targets[src], (dir, trg))
        end
    end
    RestrictedSourceMask(targets)
end

getorder(mask::RestrictedSourceMask, src) = mask.targets[src]