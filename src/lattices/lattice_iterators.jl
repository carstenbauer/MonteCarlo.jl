################################################################################
### Lattice Iteration Cache
################################################################################

#=
Many of the iterators implemented here do some sorting or filtering based on 
distance vectors. For them to run efficiently they may need various maps 
connecting distance vector indices, source site indices and target site indices.

In order to avoid duplicating data across different iterators we construct a 
cache here. Each iterator will fetch the data it needs from the cache or 
generate the data if it is not available yet.
=#

abstract type LICacheKeys end
struct Dir2SrcTrg <: LICacheKeys end
struct Src2DirTrg <: LICacheKeys end
struct SrcTrg2Dir <: LICacheKeys end
# remaps dirs: filter -> 1:length(filter) to avoid empty data in output
struct FilteredSrc2IdxTrg{T} <: LICacheKeys
    filter::T
end

struct LatticeIteratorCache
    cache::Dict{Any, Any}
end
LatticeIteratorCache() = LatticeIteratorCache(Dict{Any, Any}())

# for simplicity
function Base.getindex(mc::MonteCarloFlavor, key::LICacheKeys)
    get!(mc.lattice_iterator_cache, key, lattice(mc))
end


Base.getindex(cache::LatticeIteratorCache, key::LICacheKeys) = cache.cache[key]
function Base.get!(cache::LatticeIteratorCache, key::LICacheKeys, lattice::AbstractLattice)
    !haskey(cache, key) && push!(cache, key, lattice)
    cache[key]
end
Base.empty!(cache::LatticeIteratorCache) = empty!(cache.cache)
Base.haskey(cache::LatticeIteratorCache, key::LICacheKeys) = haskey(cache.cache, key)


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

function Base.push!(cache::LatticeIteratorCache, key::Dir2SrcTrg, lattice::AbstractLattice, ϵ=1e-6)
    if !haskey(cache.cache, key)
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
        cache.cache[key] = bonds[temp]
    end
    nothing
end

function Base.push!(cache::LatticeIteratorCache, key::SrcTrg2Dir, lattice::AbstractLattice)
    if !haskey(cache.cache, key)
        push!(cache, Dir2SrcTrg(), lattice)
        dir2srctrg = cache[Dir2SrcTrg()]
        srctrg2dir = [-1 for _ in 1:length(lattice), __ in 1:length(lattice)]
        for dir in eachindex(dir2srctrg)
            for (src, trg) in dir2srctrg[dir]
                srctrg2dir[src, trg] = dir
            end
        end
        cache.cache[key] = srctrg2dir
    end
end
function Base.push!(cache::LatticeIteratorCache, key::Src2DirTrg, lattice::AbstractLattice)
    if !haskey(cache.cache, key)
        push!(cache, Dir2SrcTrg(), lattice)
        dir2srctrg = cache[Dir2SrcTrg()]
        trg_from_src = [Tuple{Int64, Int64}[] for _ in 1:length(lattice)]
        for dir in eachindex(dir2srctrg)
            for (src, trg) in dir2srctrg[dir]
                push!(trg_from_src[src], (dir, trg))
            end
        end
        cache.cache[key] = trg_from_src
    end
    nothing
end

function Base.push!(cache::LatticeIteratorCache, key::FilteredSrc2IdxTrg, lattice::AbstractLattice)
    if !haskey(cache.cache, key)
        push!(cache, Src2DirTrg(), lattice)

        dir2idx = [-1 for _ in 1:length(lattice)]
        for (idx, dir) in enumerate(key.filter)
            dir2idx[dir] = idx
        end

        cache.cache[key] = map(cache[Src2DirTrg()]) do dir_trg_list
            [(dir2idx[dir], trg) for (dir, trg) in dir_trg_list if dir in key.filter]
        end
    end
    nothing
end


################################################################################
### Abstract Iterator Types
################################################################################


abstract type AbstractLatticeIterator end
# All indices are sites
abstract type DirectLatticeIterator <: AbstractLatticeIterator end 
# first index is a meta index (e.g. direction), rest for sites
abstract type DeferredLatticeIterator <: AbstractLatticeIterator end 
# Wraps a lattice iterator to do change what happens with the output
abstract type LatticeIterationWrapper{LI <: AbstractLatticeIterator} <: AbstractLatticeIterator end

abstract type AbstractLatticeIteratorTemplate end
abstract type DirectLatticeIteratorTemplate <: AbstractLatticeIteratorTemplate end
abstract type DeferredLatticeIteratorTemplate <: AbstractLatticeIteratorTemplate end

# is this valid?
function (iter::AbstractLatticeIteratorTemplate)(mc::MonteCarloFlavor, ::Model)
    iter(mc.lattice_iterator_cache, lattice(mc))
end


################################################################################
### RangeIterator (1 index)
################################################################################


"""
    EachSiteAndFlavor()

Creates an iterator template which iterates through the diagonal of the Greensfunction.
"""
struct EachSiteAndFlavor <: DirectLatticeIteratorTemplate end
function (::EachSiteAndFlavor)(mc::MonteCarloFlavor, model::Model)
    RangeIterator(1 : length(lattice(mc)) * nflavors(mc))
end

"""
    EachSite()

Creates an iterator template which iterates through every site of a given lattice.
"""
struct EachSite <: DirectLatticeIteratorTemplate end
(::EachSite)(_, l::AbstractLattice) = RangeIterator(1 : length(l))


struct RangeIterator <: DirectLatticeIterator
    range::UnitRange{Int64}
end
Base.iterate(iter::RangeIterator) = iterate(iter.range)
Base.iterate(iter::RangeIterator, state) = iterate(iter.range, state)
Base.length(iter::RangeIterator) = length(iter.range)
Base.eltype(iter::RangeIterator) = eltype(iter.range)


################################################################################
### OnSite (src, trg)
################################################################################


"""
    OnSite()

Creates an iterator template which iterates through every site of a given lattice, 
returning the linear index equivalent to (site, site) at every step.
"""
struct OnSite <: DirectLatticeIteratorTemplate end
(::OnSite)(_, l::AbstractLattice) = _OnSite(length(l))

struct _OnSite <: DirectLatticeIterator
    N::Int64
end

Base.iterate(iter::_OnSite, i=1) = i ≤ iter.N ? ((i, i), i+1) : nothing 
Base.length(iter::_OnSite) = iter.N
Base.eltype(::_OnSite) = Tuple{Int64, Int64}


################################################################################
### EachSitePair (src, trg)
################################################################################


"""
    EachSitePair()

Creates an iterator template which returns every pair of sites `(s1, s2)` with 
`s1, s2 ∈ 1:Nsites`.
"""
struct EachSitePair <: DirectLatticeIteratorTemplate end
(::EachSitePair)(_, l::AbstractLattice) = _EachSitePair(length(l))

struct _EachSitePair <: DirectLatticeIterator
    N::Int64
end

function Base.iterate(iter::_EachSitePair, i=1)
    i ≤ iter.N^2 ? ((div(i-1, iter.N)+1, mod1(i, iter.N)), i+1) : nothing
end
Base.length(iter::_EachSitePair) = iter.N^2
Base.eltype(::_EachSitePair) = NTuple{2, Int64}


################################################################################
### EachSitePairByDistance (dir, src, trg)
################################################################################


"""
    EachSitePairByDistance()

Creates an iterator template which returns triplets 
`(direction index, source, target)` sorted by distance. The `direction index` 
identifies each unique direction `position(target) - position(source)`.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachSitePairByDistance <: DeferredLatticeIteratorTemplate end
function (::EachSitePairByDistance)(cache::LatticeIteratorCache, l::AbstractLattice)
    push!(cache, Dir2SrcTrg(), l)
    # this should only keep a reference I think?
    dir2srctrg = cache[Dir2SrcTrg()]
    _EachSitePairByDistance(length(l)^2, dir2srctrg)
end
function ndirections(mc, ::EachSitePairByDistance)
    push!(mc.lattice_iterator_cache, Dir2SrcTrg(), lattice(mc))
    Ndir = length(mc.lattice_iterator_cache[Dir2SrcTrg()])
    (Ndir, )
end

struct _EachSitePairByDistance <: DeferredLatticeIterator
    N::Int64
    dir2srctrg::Vector{Vector{Tuple{Int64, Int64}}}
end 


function Base.iterate(iter::_EachSitePairByDistance, state = (1, 1))
    dir, i = state
    # next_dir = dir + div(i, length(iter.pairs[dir]))
    # next_i = mod1(i+1, length(iter.pairs[dir]))
    # if next_dir > length(iter.pairs)
    #     return nothing
    # else
    #     src, trg = iter.pairs[dir][i]
    #     return (dir, src, trg), (next_dir, next_i)
    # end


    if dir ≤ length(iter.dir2srctrg)
        if i ≤ length(iter.dir2srctrg[dir])
            src, trg = iter.dir2srctrg[dir][i]
            return (dir, src, trg), (dir, i+1)
        else
            return iterate(iter, (dir+1, 1))
        end
    else
        return nothing
    end
end
Base.length(iter::_EachSitePairByDistance) = iter.N
Base.eltype(::_EachSitePairByDistance) = NTuple{3, Int64}


################################################################################
### EachLocalQuadByDistance
################################################################################



"""
    EachLocalQuadByDistance([directions])()

Creates an iterator template which yields 4-tuples of sites with 3 directions
inbetween. The output is given as `(combined_dir, src1, trg1, src2, trg2)`
where `combined_dir` is a linear index of `(dir12, dir1, dir2)` corresponding to
the distance vectors `(src2 - src1, trg1 - src1, trg2 - src2)`. 

The target-source directions can be limited with `directions`. If an integer is 
passed it will include `1:directions` as dir1 and dir2. If a range or vector is
passed it will use that range. Note that if that range or vector is 
discontinuous, the output will be remapped to avoid storing unnecessary data. 
I.e. `directions = [2, 5, 6]` will pick target sites matching directions 2, 5
and 6, but return 1, 2 and 3 as directions.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachLocalQuadByDistance{T} <: DeferredLatticeIteratorTemplate
    directions::T
    EachLocalQuadByDistance(N::Integer) = EachLocalQuadByDistance(1:N)
    EachLocalQuadByDistance(dirs::T) where T = new{T}(dirs)
end

function EachLocalQuadByDistance{K}(args...) where {K}
    EachLocalQuadByDistance(K)(args...)
end
function Base.:(==)(l::EachLocalQuadByDistance, r::EachLocalQuadByDistance)
    l.directions == r.directions
end
function ndirections(mc, iter::EachLocalQuadByDistance)
    push!(mc.lattice_iterator_cache, Dir2SrcTrg(), lattice(mc))
    Ndir = length(mc.lattice_iterator_cache[Dir2SrcTrg()])
    K = length(iter.directions)
    (Ndir, K, K)
end

struct _EachLocalQuadByDistance <: DeferredLatticeIterator
    mult::Tuple{Int64, Int64}
    N::Int64
    src_mask::Vector{Int64}
    srctrg2dir::Matrix{Int64}
    filtered_src2dirtrg::Vector{Vector{Tuple{Int64, Int64}}}
end
function (config::EachLocalQuadByDistance)(cache::LatticeIteratorCache, lattice::AbstractLattice)
    key = FilteredSrc2IdxTrg(config.directions)
    push!(cache, key, lattice)
    push!(cache, SrcTrg2Dir(), lattice)
    
    Ndir = length(cache[Dir2SrcTrg()])
    srctrg2dir = cache[SrcTrg2Dir()]
    filtered_src2dirtrg = cache[key]
    
    src_mask = [i for i in eachindex(filtered_src2dirtrg) if !isempty(filtered_src2dirtrg[i])]
    N = mapreduce(length, +, filtered_src2dirtrg)
    
    _EachLocalQuadByDistance(
        (Ndir, Ndir * length(config.directions)), 
        N^2, src_mask, srctrg2dir, filtered_src2dirtrg
    )
end

function Base.iterate(iter::_EachLocalQuadByDistance)
    src = iter.src_mask[1]
    dir12 = iter.srctrg2dir[src, src]
    dir, trg = iter.filtered_src2dirtrg[src][1]
    combined_dir = dir12 + iter.mult[1] * (dir-1) + iter.mult[2] * (dir-1)
    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src, trg, src, trg), (1, 1, 1, 1))
end

function Base.iterate(iter::_EachLocalQuadByDistance, state)
    midx1, midx2, fidx1, fidx2 = state
    src1 = iter.src_mask[midx1]
    src2 = iter.src_mask[midx2]

    # for reference:
    # if fidx2 == length(iter.filtered_src2dirtrg[src2])
    #     if fidx1 == length(iter.filtered_src2dirtrg[src1])
    #         if midx2 == length(iter.src_mask)
    #             if midx1 == length(iter.src_mask)
    #                 return nothing
    #             else
    #                 fidx1 = fidx2 = midx2 = 1
    #                 midx1 += 1
    #             end
    #         else
    #             fidx1 = fidx2 = 1
    #             midx2 += 1
    #         end
    #     else
    #         fidx2 = 1
    #         fidx1 += 1
    #     end
    # else
    #     fidx2 += 1
    # end

    # Branchless V2
    b1 = fidx2 == length(iter.filtered_src2dirtrg[src2])
    b2 = b1 && (fidx1 == length(iter.filtered_src2dirtrg[src1]))
    b3 = b2 && (midx2 == length(iter.src_mask))
    b4 = b3 && (midx1 == length(iter.src_mask))
    b4 && return nothing
    fidx2 = Int64(b1 || (fidx2 + 1))
    fidx1 = Int64(b2 || (fidx1 + b1))
    midx2 = Int64(b3 || (midx2 + b2))
    midx1 = Int64(midx1 + b3)

    src1 = iter.src_mask[midx1]
    src2 = iter.src_mask[midx2]
    dir12 = iter.srctrg2dir[src1, src2]
    dir1, trg1 = iter.filtered_src2dirtrg[src1][fidx1]
    dir2, trg2 = iter.filtered_src2dirtrg[src2][fidx2]
    combined_dir = dir12 + iter.mult[1] * (dir1-1) + iter.mult[2] * (dir2-1)
    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src1, trg1, src2, trg2), (midx1, midx2, fidx1, fidx2))
end


Base.length(iter::_EachLocalQuadByDistance) = iter.N
Base.eltype(::_EachLocalQuadByDistance) = NTuple{5, Int64}


################################################################################
### EachLocalQuadBySyncedDistance
################################################################################


"""
    EachLocalQuadBySyncedDistance(directions)

Creates an iterator template which yields 5-tuples of one combined direction and
4 sites `(combined_dir, src1, trg1, src2, trg2)`. The former is the linear 
index associated with `(dir12, dir)` where `dir12` refers to the distance 
vector between sources and `dir` to the distance vector between both target 
source pairs. 

The target-source directions can be limited with `directions`. If an integer is 
passed it will include `1:directions` as dir1 and dir2. If a range or vector is
passed it will use that range. Note that if that range or vector is 
discontinuous, the output will be remapped to avoid storing unnecessary data. 
I.e. `directions = [2, 5, 6]` will pick target sites matching directions 2, 5
and 6, but return 1, 2 and 3 as directions.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachLocalQuadBySyncedDistance{T} <: DeferredLatticeIteratorTemplate
    directions::T
    EachLocalQuadBySyncedDistance(N::Integer) = EachLocalQuadBySyncedDistance(1:N)
    EachLocalQuadBySyncedDistance(dirs::T) where T = new{T}(dirs)
end

function EachLocalQuadBySyncedDistance{K}(args...) where {K}
    EachLocalQuadBySyncedDistance(K)(args...)
end
function Base.:(==)(l::EachLocalQuadBySyncedDistance, r::EachLocalQuadBySyncedDistance)
    l.directions == r.directions
end
function ndirections(mc, iter::EachLocalQuadBySyncedDistance)
    push!(mc.lattice_iterator_cache, Dir2SrcTrg(), lattice(mc))
    Ndir = length(mc.lattice_iterator_cache[Dir2SrcTrg()])
    K = length(iter.directions)
    (Ndir, K)
end

struct _EachLocalQuadBySyncedDistance <: DeferredLatticeIterator
    N::Int64
    Ndir::Int64
    directions::Vector{Int64}
    dir2srctrg::Vector{Vector{Tuple{Int64, Int64}}}
    srctrg2dir::Matrix{Int64}
end

function (config::EachLocalQuadBySyncedDistance)(
        cache::LatticeIteratorCache, lattice::AbstractLattice
    )
    push!(cache, Dir2SrcTrg(), lattice)
    push!(cache, SrcTrg2Dir(), lattice)

    dir2srctrg = cache[Dir2SrcTrg()]
    srctrg2dir = cache[SrcTrg2Dir()]

    # Loop structure
    # for sync_dir in directions
    #     for (src1, trg1) in dir2srctrg[sync_dir]
    # 	    for (src2, trg2) in dir2srctrg[sync_dir]
    # 		    dir12 = srctrg2dir[src1, src2]
    N = mapreduce(dir -> length(dir2srctrg[dir])^2, +, config.directions)

    _EachLocalQuadBySyncedDistance(
        N, length(dir2srctrg), collect(config.directions), dir2srctrg, srctrg2dir
    )
end

function Base.iterate(iter::_EachLocalQuadBySyncedDistance, state = (1,1,0))
    sync_idx, idx1, idx2 = state

    # Branchless increments
    sync_dir = iter.directions[sync_idx]
    N = length(iter.dir2srctrg[sync_dir])

    b1 = idx2 == N
    b2 = b1 && (idx1 == N)
    b3 = b2 && (sync_idx == length(iter.directions))

    b3 && return nothing

    idx2 = Int64(b1 || (idx2 + 1))
    idx1 = Int64(b2 || (idx1 + b1))
    sync_idx = Int64(sync_idx + b2)

    sync_dir = iter.directions[sync_idx]
    src1, trg1 = iter.dir2srctrg[sync_dir][idx1]
    src2, trg2 = iter.dir2srctrg[sync_dir][idx2]
    dir12 = iter.srctrg2dir[src1, src2]
    combined_dir = dir12 + iter.Ndir * (sync_dir - 1)

    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src1, trg1, src2, trg2), (sync_idx, idx1, idx2))
end

Base.length(iter::_EachLocalQuadBySyncedDistance) = iter.N
Base.eltype(::_EachLocalQuadBySyncedDistance) = NTuple{5, Int64}


################################################################################
### Sum Wrapper
################################################################################


"""
    Sum(iteration_template)

Sum is a wrapper for iteration templates that indicates summation. It generates 
`_Sum(iter)` when constructed (constructing whatever template it holds) so that
it can be dispatched on. 
"""
struct Sum{T} <: AbstractLatticeIteratorTemplate
    template::T
end
# compat
Sum{T}(args...) where {T} = _Sum(T()(args...))
struct _Sum{LI} <: LatticeIterationWrapper{LI}
    iter::LI
end
function (config::Sum)(cache::LatticeIteratorCache, lattice::AbstractLattice)
    if config.template isa EachSitePairByDistance
        _Sum(EachSitePair()(cache, lattice))
    else
        _Sum(config.template(cache, lattice))
    end
end

Base.iterate(s::_Sum) = iterate(s.iter)
Base.iterate(s::_Sum, state) = iterate(s.iter, state)
Base.length(s::_Sum) = length(s.iter)
Base.eltype(s::_Sum) = eltype(s.iter)


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
struct ApplySymmetries{IT, N, T} <: AbstractLatticeIteratorTemplate
    template::IT
    symmetries::NTuple{N, Vector{T}}
end
function ApplySymmetries{LI}(symmetries::Vector...) where {LI}
    ApplySymmetries(LI(), symmetries)
end
function ApplySymmetries(template::DeferredLatticeIteratorTemplate, symmetries::Vector...)
    ApplySymmetries(template, symmetries)
end

struct _ApplySymmetries{LI <: DeferredLatticeIterator, N, T} <: LatticeIterationWrapper{LI}
    iter::LI
    symmetries::NTuple{N, T}
end
function (x::ApplySymmetries)(cache::LatticeIteratorCache, lattice::AbstractLattice)
    iter = x.template(cache, lattice)
    _ApplySymmetries(iter, x.symmetries)
end

Base.iterate(s::_ApplySymmetries) = iterate(s.iter)
Base.iterate(s::_ApplySymmetries, state) = iterate(s.iter, state)
Base.length(s::_ApplySymmetries) = length(s.iter)
Base.eltype(s::_ApplySymmetries) = eltype(s.iter)


################################################################################
### Utility (directions)
################################################################################


# function directions(::EachSitePair, lattice::AbstractLattice)
#     pos = positions(lattice)
#     [p2 .- p1 for p2 in pos for p1 in pos]
# end

function directions(iter::_EachSitePairByDistance, lattice::AbstractLattice, ϵ=1e-6)
    pos = MonteCarlo.positions(lattice)
    wrap = generate_combinations(lattice_vectors(lattice))

    map(iter.dir2srctrg) do pairs
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


