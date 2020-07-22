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


"""
    DistanceMask(lattice) <: AbstractMask

Constructs a mask that orders sites by distance and direction relative to each
other site.

For example:
- `mask[src, n]` returns the n-th furthest site from `src`
- `mask[:, n]` returns the n-th furthest site from each site. The
direction of each real space vector (target position - source positions) is
equal (up to periodic bonds).
- `mask[src, :]` returns a list of all sites, ordered by distance from `src`.
The first site in this list is `src`

Warning: For this to work correctly the lattice must provide the neighbors in
order. Furthermore each bond is assumed to be of equal length.
"""
struct DistanceMask <: AbstractMask
    targets::Matrix{Int64}
end
DistanceMask(lattice::AbstractLattice) = MethodError(DistanceMask, (lattice))
# These should be save!?
DistanceMask(lattice::Chain) = default_distance_mask(lattice)
DistanceMask(lattice::SquareLattice) = default_distance_mask(lattice)
DistanceMask(lattice::CubicLattice) = default_distance_mask(lattice)
DistanceMask(lattice::TriangularLattice) = default_distance_mask(lattice)
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

    DistanceMask(targets)
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
Base.getindex(mask::DistanceMask, source, target) = mask.targets[source, target]
