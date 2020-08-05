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



"""
    DistanceMask(lattice) <: AbstractMask

Constructs a mask that orders sites by distance and direction relative to each
other site.

Related Functions:
* `getorder(mask, src)`: Returns an iterable of (idx, trg) tuples, ordered by
distance. `idx` has a 1:1 correspondence with a directions, i.e. every
`(trg, src)` pair which points in the same direction will have the same `idx`.
* `directions(mask, lattice)`: Returns a list of directions which can be indexed
by the `idx` returned from `getorder` to get the corresponding direction.
* `mask[src, n]`: Returns the n-th closest site from `src`. Depending on the
lattice it may also return a directional `idx`. If it does not, that index is
given by `n`.

Warning: For this to work correctly the lattice must provide the neighbors in
order. Furthermore each bond is assumed to be of equal length.
"""
abstract type DistanceMask <: AbstractMask end
struct SimpleDistanceMask <: DistanceMask
    targets::Matrix{Int64}
end
DistanceMask(lattice::AbstractLattice) = MethodError(DistanceMask, (lattice))
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
Base.getindex(mask::DistanceMask, source, target_idx) = mask.targets[source, target_idx]
# TODO name?
getorder(mask::SimpleDistanceMask, source) = enumerate(mask.targets[source, :])
Base.size(mask::DistanceMask) = size(mask.targets)
Base.size(mask::DistanceMask, dim) = size(mask)[dim]
"""
    directions(mask, lattice)

Returns a vector of directions in the order emposed by the given mask.

For a `RawMask` the output directly mimics the mask, i.e.
`directions(mask, lattice)[src, idx]` is the direction to
`trg = mask[src, idx]`.
For a `DistanceMask` the `directions(mask, lattice)[dir_idx]` return the
direction of `dir_idx, trg = mask[src, idx]`.
"""
function directions(mask::SimpleDistanceMask, lattice::AbstractLattice)
    pos = positions(lattice)
    [pos[1] .- p for p in pos[mask.targets[1, :]]]
end


struct VerboseDistanceMask <: DistanceMask
    targets::Matrix{Tuple{Int64, Int64}}
end
# NOTE Does this definition makes sense?
# This would be (number of source sites, number of directions)
Base.size(mask::VerboseDistanceMask) = (size(mask.targets, 1), maximum(first(x) for x in mask.targets))
getorder(mask::VerboseDistanceMask, source) = mask.targets[source, :]
function directions(mask::VerboseDistanceMask, lattice::AbstractLattice)
    pos = MonteCarlo.positions(lattice)
    marked = Set{Int64}()
    dirs = Vector{eltype(pos)}(undef, maximum(first(x) for x in mask.targets))
    for src in 1:size(mask.targets, 1)
        for (idx, trg) in mask.targets[src, :]
            if !(idx in marked)
                push!(marked, idx)
                dirs[idx] = pos[trg] - pos[src]
            end
        end
    end
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
    # distance_idx, src, trg
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
