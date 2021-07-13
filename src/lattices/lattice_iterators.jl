abstract type AbstractLatticeIterator end
# All indices are sites
abstract type DirectLatticeIterator <: AbstractLatticeIterator end 
# first index is a meta index (e.g. direction), rest for sites
abstract type DeferredLatticeIterator <: AbstractLatticeIterator end 
# Wraps a lattice iterator to do change what happens with the output
abstract type LatticeIterationWrapper{LI <: AbstractLatticeIterator} <: AbstractLatticeIterator end


################################################################################
### EachSiteAndFlavor
################################################################################



"""
    EachSiteAndFlavor(mc, model)

Creates an iterator which iterates through the diagonal of the Greensfunctio
"""
struct EachSiteAndFlavor <: DirectLatticeIterator
    N::Int64
end
function EachSiteAndFlavor(mc::MonteCarloFlavor, model::Model)
    EachSiteAndFlavor(length(lattice(model)) * nflavors(model))
end

@bm Base.iterate(iter::EachSiteAndFlavor, i=1) = i ≤ iter.N ? (i, i+1) : nothing 
Base.length(iter::EachSiteAndFlavor) = iter.N
Base.eltype(::EachSiteAndFlavor) = Int64



################################################################################
### EachSite
################################################################################



"""
    EachSite(Nsites)
    EachSite(lattice)
    EachSite(mc, model)

Creates an iterator which iterates through every site of a given lattice.
"""
struct EachSite <: DirectLatticeIterator
    N::Int64
end
EachSite(l::AbstractLattice) = EachSite(length(l))
EachSite(mc::MonteCarloFlavor, model::Model) = EachSite(lattice(model))
eachsite(args...) = EachSite(args...)

@bm Base.iterate(iter::EachSite, i=1) = i ≤ iter.N ? (i, i+1) : nothing 
Base.length(iter::EachSite) = iter.N
Base.eltype(::EachSite) = Int64



################################################################################
### OnSite
################################################################################



"""
    OnSite(Nsites)
    OnSite(lattice)
    OnSite(mc, model)

Creates an iterator which iterates through every site of a given lattice, 
returning (site, site) at every step.
"""
struct OnSite <: DirectLatticeIterator
    N::Int64
end
OnSite(l::AbstractLattice) = OnSite(length(l))
OnSite(mc::MonteCarloFlavor, model::Model) = OnSite(lattice(model))
onsite(args...) = OnSite(args...)

@bm Base.iterate(iter::OnSite, i=1) = i ≤ iter.N ? ((i, i), i+1) : nothing 
Base.length(iter::OnSite) = iter.N
Base.eltype(::OnSite) = Tuple{Int64, Int64}



################################################################################
### EachSitePair
################################################################################



"""
    EachSitePair(Nsites)
    EachSitePair(lattice)
    EachSitePair(mc, model)

Creates an iterator which returns every pair of sites `(s1, s2)` with 
`s1, s2 ∈ 1:Nsites`.
"""
struct EachSitePair <: DirectLatticeIterator
    N::Int64
end
EachSitePair(l::AbstractLattice) = EachSitePair(length(l))
EachSitePair(mc::MonteCarloFlavor, model::Model) = EachSitePair(lattice(model))
eachsitepair(args...) = EachSitePair(args...)

@bm function Base.iterate(iter::EachSitePair, i=1)
    if i ≤ iter.N^2
        return ((div(i-1, iter.N)+1, mod1(i, iter.N)), i+1)
    else
        return nothing 
    end
end
Base.length(iter::EachSitePair) = iter.N^2
Base.eltype(::EachSitePair) = NTuple{2, Int64}



################################################################################
### EachSitePairByDistance
################################################################################



"""
    EachSitePairByDistance(lattice)
    EachSitePairByDistance(mc, model)

Creates an iterator which returns triplets `(direction index, source, target)`
sorted by distance. The `direction index` identifies each unqiue direction
`position(target) - position(source)`.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachSitePairByDistance <: DeferredLatticeIterator
    N::Int64
    pairs::Vector{Vector{Tuple{Int64, Int64}}}
end 

# For shifting sites across periodic bounds
function generate_combinations(vs::Vector{<: Vector})
    out = [zeros(length(vs[1]))]
    for v in vs
        out = vcat([e.-v for e in out], out, [e.+v for e in out])
    end
    out
end

# norm + ϵ * angle(v, e_x)
function directed_norm(v, ϵ)
    l = norm(v)
    if length(v) == 2 && l > ϵ
        angle = acos(dot([1, 0], v) / l)
        v[2] < 0 && (angle = 2pi - angle)
        return l + ϵ * angle
    else
        return l
    end
end

function EachSitePairByDistance(lattice::AbstractLattice, ϵ = 1e-6)
    _positions = positions(lattice)
    wrap = generate_combinations(lattice_vectors(lattice))
    directions = Vector{Float64}[]
    # (src, trg), first index is dir, second index irrelevant
    bonds = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]

    for origin in 1:length(lattice)
        for (trg, p) in enumerate(_positions)
            d = _positions[origin] .- p .+ wrap[1]
            for v in wrap[2:end]
                new_d = _positions[origin] .- p .+ v
                if directed_norm(new_d, ϵ) + ϵ < directed_norm(d, ϵ)
                    d .= new_d
                end
            end
            # The rounding will allow us to use == here
            idx = findfirst(dir -> isapprox(dir, d, atol=ϵ), directions)
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

    temp = sortperm(directions, by = v -> directed_norm(v, ϵ))

    EachSitePairByDistance(length(lattice)^2, bonds[temp])
end
function EachSitePairByDistance(mc::MonteCarloFlavor, model::Model)
    EachSitePairByDistance(lattice(model))
end


@bm function Base.iterate(iter::EachSitePairByDistance, state = (1, 1))
    dir, i = state
    # next_dir = dir + div(i, length(iter.pairs[dir]))
    # next_i = mod1(i+1, length(iter.pairs[dir]))
    # if next_dir > length(iter.pairs)
    #     return nothing
    # else
    #     src, trg = iter.pairs[dir][i]
    #     return (dir, src, trg), (next_dir, next_i)
    # end


    if dir ≤ length(iter.pairs)
        if i ≤ length(iter.pairs[dir])
            src, trg = iter.pairs[dir][i]
            return (dir, src, trg), (dir, i+1)
        else
            return iterate(iter, (dir+1, 1))
        end
    else
        return nothing
    end
end
ndirections(iter::EachSitePairByDistance) = size(iter.pairs)
Base.length(iter::EachSitePairByDistance) = iter.N
Base.eltype(::EachSitePairByDistance) = NTuple{3, Int64}


"""
    in_direction(iter::EachSitePairByDistance, dir::Int64)

Returns a vector of each `(src, trg)` pair in the given direction.
"""
in_direction(iter::EachSitePairByDistance, dir::Int64) = iter.pairs[dir]



################################################################################
### EachLocalQuadByDistance
################################################################################



"""
    EachLocalQuadByDistance{K}(lattice)
    EachLocalQuadByDistance{K}(mc, model)

Creates an iterator which returns a tuple 
`(dir12, dir1, dir2, src1, trg1, src2, trg2)` with each iteration. The 
directional index `dir12` identifies the direction `src2 - src1`. 
The type parameter `K` is the maximum direction index `dir1` (`dir2`) a pair 
`(src1, trg1)` (`(src2, trg2)`) may have. Picking for example `K = 5` for a 
`SquareLattice` will result in the `src` site (`K = 1`) and the four nearest 
neighbors of the `src` site being selected as targets.
Note that `K` is not always the number of nearest neighbors + 1. For example, 
the Honeycomb lattice has 3 NNs, but 6 unique directions. To include all NN 
here, one should pick `K = 7`.

The iterator is sorted by distance `src2 - src1`.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachLocalQuadByDistance{K} <: DeferredLatticeIterator
    N::Int64
    implied::Array{Vector{NTuple{4, UInt16}}, 3}
end


function EachLocalQuadByDistance{K}(lattice::AbstractLattice) where {K}
    # (src1, src2) pairs from dir
    pairs_by_dir = EachSitePairByDistance(lattice)

    # (dir, trg1) from src1
    trg_from_src = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]
    for dir in 1:K
        for (src, trg) in in_direction(pairs_by_dir, dir)
            push!(trg_from_src[src], (dir, trg))
        end
    end

    # NOTE: Creating 7-Tuples is faster ~15% faster but I'm worried about memory
    # usuage. Caching 4-tuples is still ~36% faster than running the loop below
    # during measurements, and is not quite as demanding on memory.
    total_length = sum(length(x) for x in trg_from_src)^2
    # linear = Vector{NTuple{7, UInt16}}(undef, total_length)
    # k = 1
    implied = [NTuple{4, UInt16}[] for _ in 1:length(pairs_by_dir.pairs), _ in 1:K, _ in 1:K]
    dir12, idx, i, j = (1,1,1,1)
    while true
        if dir12 ≤ length(pairs_by_dir.pairs)
            if idx ≤ length(pairs_by_dir.pairs[dir12])
                src1, src2 = pairs_by_dir.pairs[dir12][idx]
                if i ≤ length(trg_from_src[src1])
                    dir1, trg1 = trg_from_src[src1][i]
                    if j ≤ length(trg_from_src[src2])
                        dir2, trg2 = trg_from_src[src2][j]
                        # linear[k] = UInt16.((dir12, dir1, dir2, src1, trg1, src2, trg2))
                        push!(
                            implied[dir12, dir1, dir2],
                            UInt16.((src1, trg1, src2, trg2))
                        )
                        # k += 1
                        j += 1
                    else
                        i += 1
                        j = 1
                    end
                else
                    idx += 1
                    i = j = 1
                end

            else
                dir12 += 1
                idx = i = j = 1
            end
        else
            break
        end
    end

    EachLocalQuadByDistance{K}(total_length, implied)
end
function EachLocalQuadByDistance{K}(mc::MonteCarloFlavor, model::Model) where {K}
    EachLocalQuadByDistance{K}(lattice(model))
end

# @bm function Base.iterate(iter::EachLocalQuadByDistance, state=1)
#     if state <= length(iter.direct)
#         return (iter.direct[state], state+1)
#     else
#         return nothing
#     end
# end
@bm function Base.iterate(iter::EachLocalQuadByDistance{K}, state=(1,1)) where {K}
    i, j = state
    if i <= length(iter.implied)
        N = length(iter.implied[i])
        
        # This is required for lattices with a basis
        j = ifelse(N == 0, 1, j)
        while N == 0
            i += 1
            i <= length(iter.implied) || return nothing
            N = length(iter.implied[i])
        end

        next_j = mod1(j+1, N)
        next_i = i + div(j, N)
            
        t = iter.implied[i][j]
        return ((i, t[1], t[2], t[3], t[4]), (next_i, next_j))
    else
        return nothing
    end
end

ndirections(iter::EachLocalQuadByDistance{K}) where {K} = size(iter.implied)
Base.length(iter::EachLocalQuadByDistance) = iter.N
Base.eltype(::EachLocalQuadByDistance) = Tuple{Int64, UInt16, UInt16, UInt16, UInt16}



################################################################################
### EachLocalQuadBySyncedDistance
################################################################################



"""
    EachLocalQuadBySyncedDistance{K}(lattice)
    EachLocalQuadBySyncedDistance{K}(mc, model)

Creates an iterator which returns a tuple 
`(dir12, dir_ii, src1, trg1, src2, trg2)` with each iteration. The 
directional index `dir12` identifies the direction `src2 - src1`. 
The type parameter `K` is the maximum direction index `dir_ii` a pair 
`(src1, trg1)` (`(src2, trg2)`) may have. Picking for example `K = 5` for a 
`SquareLattice` will result in the `src` site (`K = 1`) and the four nearest 
neighbors of the `src` site being selected as targets.
Note that `K` is not always the number of nearest neighbors + 1. For example, 
the Honeycomb lattice has 3 NNs, but 6 unique directions. To include all NN 
here, one should pick `K = 7`.

The iterator is sorted by distance `src2 - src1`.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachLocalQuadBySyncedDistance{K} <: DeferredLatticeIterator
    N::Int64
    implied::Array{Vector{NTuple{4, UInt16}}, 2}
end


function EachLocalQuadBySyncedDistance{K}(lattice::AbstractLattice) where {K}
    # (src1, src2) pairs from dir
    pairs_by_dir = EachSitePairByDistance(lattice)

    # (dir, trg1) from src1
    trg_from_src = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]
    for dir in 1:K
        for (src, trg) in in_direction(pairs_by_dir, dir)
            push!(trg_from_src[src], (dir, trg))
        end
    end
    
    implied = [NTuple{4, UInt16}[] for _ in 1:length(pairs_by_dir.pairs), _ in 1:K]
    dir12, idx, i, j = (1,1,1,1)
    while true
        if dir12 ≤ length(pairs_by_dir.pairs)
            if idx ≤ length(pairs_by_dir.pairs[dir12])
                src1, src2 = pairs_by_dir.pairs[dir12][idx]
                if i ≤ length(trg_from_src[src1])
                    dir1, trg1 = trg_from_src[src1][i]
                    if j ≤ length(trg_from_src[src2])
                        dir2, trg2 = trg_from_src[src2][j]
                        j += 1
                        dir1 != dir2 && continue
                        push!(
                            implied[dir12, dir1],
                            UInt16.((src1, trg1, src2, trg2))
                        )
                    else
                        i += 1
                        j = 1
                    end
                else
                    idx += 1
                    i = j = 1
                end

            else
                dir12 += 1
                idx = i = j = 1
            end
        else
            break
        end
    end
    total_length = sum(length(x) for x in implied)

    EachLocalQuadBySyncedDistance{K}(total_length, implied)
end
function EachLocalQuadBySyncedDistance{K}(mc::MonteCarloFlavor, model::Model) where {K}
    EachLocalQuadBySyncedDistance{K}(lattice(model))
end


@bm function Base.iterate(iter::EachLocalQuadBySyncedDistance{K}, state = (1, 1)) where {K}
    i, j = state
    if i <= length(iter.implied)
        N = length(iter.implied[i])

        # This is required for lattices with a basis
        j = ifelse(N == 0, 1, j)
        while N == 0
            i += 1
            i <= length(iter.implied) || return nothing
            N = length(iter.implied[i])
        end

        next_j = mod1(j+1, N)
        next_i = i + div(j, N)
        t = iter.implied[i][j]
        return ((i, t[1], t[2], t[3], t[4]), (next_i, next_j))
    else
        return nothing
    end
end
ndirections(iter::EachLocalQuadBySyncedDistance{K}) where {K} = size(iter.implied)
Base.length(iter::EachLocalQuadBySyncedDistance) = iter.N
Base.eltype(::EachLocalQuadBySyncedDistance) = Tuple{Int64, UInt16, UInt16, UInt16, UInt16}



################################################################################
### Additonal stuff
################################################################################



# function directions(::EachSitePair, lattice::AbstractLattice)
#     pos = positions(lattice)
#     [p2 .- p1 for p2 in pos for p1 in pos]
# end

function directions(iter::EachSitePairByDistance, lattice::AbstractLattice, ϵ=1e-6)
    pos = MonteCarlo.positions(lattice)
    wrap = generate_combinations(lattice_vectors(lattice))

    dirs = map(iter.pairs) do pairs
        src, trg = pairs[1]
        _d = pos[src] - pos[trg]
        # Find lowest distance w/ periodic bounds
        d = _d .+ wrap[1]
        for v in wrap[2:end]
            new_d = _d .+ v
            if directed_norm(new_d, ϵ) + ϵ < directed_norm(d, ϵ)
                d .= new_d
            end
        end
        d
    end
end


directions(dqmc::MonteCarloFlavor, ϵ=1e-6) = directions(lattice(dqmc), ϵ)
directions(model::Model, ϵ=1e-6) = directions(lattice(model), ϵ)
function directions(lattice::AbstractLattice, ϵ = 1e-6)
    _positions = positions(lattice)
    wrap = generate_combinations(lattice_vectors(lattice))
    directions = Vector{Float64}[]
    for origin in 1:length(lattice)
        for (trg, p) in enumerate(_positions)
            d = _positions[origin] .- p .+ wrap[1]
            for v in wrap[2:end]
                new_d = _positions[origin] .- p .+ v
                if directed_norm(new_d, ϵ) + ϵ < directed_norm(d, ϵ)
                    d .= new_d
                end
            end
            idx = findfirst(dir -> isapprox(dir, d, atol=ϵ), directions)
            if idx === nothing
                push!(directions, d)
            end
        end
    end
    # temp = sortperm(directions, by=norm)
    # directions[temp]
    sort!(directions, by = v -> directed_norm(v, ϵ))
end



################################################################################
### Sum Wrapper
################################################################################



struct Sum{LI} <: LatticeIterationWrapper{LI}
    iter::LI
    function Sum{LI}(args...; kwargs...) where {LI <: AbstractLatticeIterator}
        new{LI}(LI(args...; kwargs...))
    end
    function Sum(iter::LI) where {LI <: AbstractLatticeIterator}
        new{LI}(iter)
    end
end

Base.iterate(s::Sum) = iterate(s.iter)
Base.iterate(s::Sum, state) = iterate(s.iter, state)
Base.length(s::Sum) = length(s.iter)
Base.eltype(s::Sum) = eltype(s.iter)



################################################################################
### Symmetry Wrapper
################################################################################



# This is a weird monster :(
# The idea is that we construct a thin wrapper
# li = ApplySymmetries{EachLocalQuadByDistance{5}}(sym1, sym2, ...)
# which contains weights for different neighbors in the given symmetries
# The backend then bundles all of these and constructs one thick wrapper by
# calling
# li(dqmc, model)
# This then actually contains the udnerlying lattice iterator

"""
    ApplySymmetries{lattice_iterator_type}(symmetries...)

`ApplySymmetries` is a wrapper for a `DeferredLatticeIterator`. It is meant to
specify how results from different directions are to be added up.

For example `T = EachLocalQuadByDistance{5}` specifies 4 site tuples where two
sites have set distances between them - one of the first 5 smallest ones. In a 
square lattice these would be on-site (1) and the four nearest neighbors (2-5).
We may use `iter = ApplySymmetries{}([1], [0, 1, 1, 1, 1])` to specify how 
results in these directions should be added up. The first rule would be s-wave,
the second extended s-wave.
These rules will be applied for DQMCMeasurements during the simulation. I.e. 
first, the normal iteration and summation from `EachLocalQuadByDistance` is 
performed. After that we have 5 values for each direction. These are then 
weighted by each "symmetry" in `ApplySymmetries` to give the final result, saved
in the DQMCMeasurement.
"""
struct ApplySymmetries{LI <: DeferredLatticeIterator, N, T} <: LatticeIterationWrapper{LI}
    symmetries::NTuple{N, Vector{T}}
end
function ApplySymmetries{LI}(symmetries::Vector{T}...) where {LI <: DeferredLatticeIterator, T}
    ApplySymmetries{LI, length(symmetries), T}(symmetries)
end

struct _ApplySymmetries{LI <: DeferredLatticeIterator, N, T} <: LatticeIterationWrapper{LI}
    iter::LI
    symmetries::NTuple{N, T}
end
function (x::ApplySymmetries{LI})(mc, model) where {LI}
    iter = LI(mc, model)
    _ApplySymmetries(iter, x.symmetries)
end

Base.iterate(s::_ApplySymmetries) = iterate(s.iter)
Base.iterate(s::_ApplySymmetries, state) = iterate(s.iter, state)
Base.length(s::_ApplySymmetries) = length(s.iter)
Base.eltype(s::_ApplySymmetries) = eltype(s.iter)



################################################################################
### Symmetry Wrapper
################################################################################



#=
CCS[dr, (0, x, y, -x, -y)]
Λxx = CCS[:, 2]
Λxxq^L = dot(CCS[:, 2], exp(-dirs .* (1/L, 0))
Λxxq^T = dot(CCS[:, 2], exp(-dirs .* (0, 1/L))
ρs = Λxxq^L - Λxxq^T

# Save:
directions(iter)
# Supply/Pick:
dir_idx (xx)
L

So is longitudinal/transversal just 
    1/L * normalize(NN_vector)
    1/L * normalize(⟂ NN_vector)
or is it a reciprocal lattice vector?
=#

"""
    SuperfluidDensity{lattice_iterator_type}(directions, longitudinal, transversal)

`SuperfluidDensity` works similarly to `ApplySymmetries` - it contains some 
additional information that should be used to transform the result of of the 
given lattice iterator. In this case the additional information is a set of 
directional indices `directions` and the related `longitudinal` and `transversal`
vectors in reciprocal space.

The point of this wrapper is to calculate the Fourier transform 
`O(q) = ∑_r0 ∑_Δr O(r_0, Δr) (exp(-i q_l Δr) - exp(-i q_t Δr))`
where q_l is `longitudinal` and q_t is transversal. The Δr ised in this formula 
are restricted to the given `directions`.
"""
struct SuperfluidDensity{LI <: DeferredLatticeIterator, N} <: LatticeIterationWrapper{LI}
    dir_idxs::NTuple{N, Int}
    long_qs::NTuple{N, Vector{Float64}}
    trans_qs::NTuple{N, Vector{Float64}}
end
function SuperfluidDensity{LI}(directions, longitudinal, transversal) where {LI <: DeferredLatticeIterator}
    SuperfluidDensity{LI, length(directions)}(
        tuple(directions...), tuple(longitudinal...), tuple(transversal...)
    )
end

struct _SuperfluidDensity{LI <: DeferredLatticeIterator, N} <: LatticeIterationWrapper{LI}
    iter::LI

    dirs::Vector{Vector{Float64}}
    dir_idxs::NTuple{N, Int}
    long_qs::NTuple{N, Vector{Float64}}
    trans_qs::NTuple{N, Vector{Float64}}
end
function (x::SuperfluidDensity{LI})(mc, model) where {LI}
    iter = LI(mc, model)
    dirs = directions(lattice(model))
    _SuperfluidDensity(iter, dirs, x.dir_idxs, x.long_qs, x.trans_qs)
end

Base.iterate(s::_SuperfluidDensity) = iterate(s.iter)
Base.iterate(s::_SuperfluidDensity, state) = iterate(s.iter, state)
Base.length(s::_SuperfluidDensity) = length(s.iter)
Base.eltype(s::_SuperfluidDensity) = eltype(s.iter)