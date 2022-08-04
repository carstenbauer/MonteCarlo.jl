# Helpers

# This can be called with one less element in Ns than in idxs (tight left)
function _sub2ind(Ns, idxs)
    idx = idxs[end] - 1
    for d in length(idxs)-1:-1:1
        idx = idx * Ns[d] + (idxs[d] - 1)
    end
    return idx + 1
end


"""
    BondDirections()

Used to specify directions to iterate over. While giving specific indices 
considers all possible pairing of sites, this only considers sites connected by 
bonds.

Currently only works with EachLocalQuadByDistance.
"""
struct BondDirections end

function _dir_idxs(l::Lattice, x::BondDirections, src::Int)
    return _dir_idxs_uc(l, x, mod1(src, length(unitcell(l))))
end
function _dir_idxs_uc(l::Lattice, ::BondDirections, uc::Int)
    uc2bonddir = l[:uc2bonddir]::Tuple{Int, Vector{Vector{Pair{Int, Int}}}}
    @inbounds return uc2bonddir[2][uc]
end
_dir_idxs(::Lattice, iterable, ::Int) = iterable
_dir_idxs_uc(::Lattice, iterable, ::Int) = iterable

_length(l::Lattice, ::BondDirections) = l[:uc2bonddir][1]
_length(::Lattice, xs) = length(xs)


################################################################################
### Abstract Iterator Types
################################################################################


abstract type AbstractLatticeIterator end
# All indices are sites
abstract type DirectLatticeIterator <: AbstractLatticeIterator end # TODO I think this is kinda useless now?
# first index is a meta index (e.g. direction), rest for sites
abstract type DeferredLatticeIterator <: AbstractLatticeIterator end 

function _save(file::FileLike, key::String, m::T) where {T <: AbstractLatticeIterator}
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", nameof(T))
    write(file, "$key/fields", getfield.((m,), fieldnames(T)))
    return
end

function _load(data, ::Val{:LatticeIterator})
    # ifelse maybe long but should be better for compile time than adding a 
    # bunch more _load methods and better for runtime than an eval
    tag = data["tag"]
    fields = data["fields"]
    for T in _all_lattice_iterator_types
        if tag == nameof(T)
            return T(fields...)
        end
    end

    # Fallback
    return eval(:($tag($(fields)...)))
end

struct WithLattice{T, N} <: AbstractLatticeIterator
    iter::T
    lattice::Lattice{N}
end

# We need to two-step the construction so we can interfere with it
with_lattice(iter, lattice) = WithLattice(iter, lattice)

# Iterator interface
Base.iterate(iter::WithLattice) = _iterate(iter.iter, iter.lattice)
Base.iterate(iter::WithLattice, state) = _iterate(iter.iter, iter.lattice, state)
Base.length(iter::WithLattice) = _length(iter.iter, iter.lattice)
Base.eltype(iter::WithLattice) = _eltype(iter.iter, iter.lattice)

# for wrappers
_length(iter::AbstractLatticeIterator, l::Lattice) = _length(iter.iter, l)
_eltype(iter::AbstractLatticeIterator, l::Lattice) = _eltype(iter.iter, l)


################################################################################
### Simple Ranges (DirectLatticeIterator)
################################################################################


"""
    EachSiteAndFlavor()

Creates an iterator template which iterates through the diagonal of the Greensfunction.
"""
struct EachSiteAndFlavor <: DirectLatticeIterator
    Nflv::Int
end
EachSiteAndFlavor(mc::MonteCarloFlavor) = EachSiteAndFlavor(nflavors(mc))
with_lattice(iter::EachSiteAndFlavor, l::Lattice) = 1 : length(l) * iter.Nflv
output_size(iter::EachSiteAndFlavor, l::Lattice) = (length(l) * iter.Nflv,)


"""
    EachSite()

Creates an iterator template which iterates through every site of a given lattice.
"""
struct EachSite <: DirectLatticeIterator end
EachSite(::MonteCarloFlavor) = EachSite()
with_lattice(::EachSite, l::Lattice) = eachindex(l)
output_size(::EachSite, l::Lattice) = (length(l),)

"""
    EachSitePair()

Creates an iterator template which returns every pair of sites `(s1, s2)` with 
`s1, s2 ∈ 1:Nsites`.
"""
struct EachSitePair <: DirectLatticeIterator end
EachSitePair(::MonteCarloFlavor) = EachSitePair()
output_size(::EachSitePair, l::Lattice) = (length(l), length(l))
function _iterate(::EachSitePair, l::Lattice, state = 0)
    state == length(l)^2 ? nothing : (fldmod1(state+1, length(l)), state+1)
end
_length(::EachSitePair, l::Lattice) = length(l)^2
_eltype(::EachSitePair, l::Lattice) = NTuple{2, Int}


################################################################################
### DeferredLatticeIterator
################################################################################


"""
    OnSite()

Creates an iterator template which iterates through every site of a given lattice, 
returning the linear index equivalent to (site, site) at every step.
"""
struct OnSite <: DeferredLatticeIterator end
OnSite(::MonteCarloFlavor) = OnSite()
output_size(::OnSite, l::Lattice) = (length(l), )
function _iterate(::OnSite, l::Lattice, i = 0)
    i == length(l) ? nothing : begin i += 1; ((i,i,i), i) end
end
_length(::OnSite, l::Lattice) = length(l)
_eltype(::OnSite, l::Lattice) = NTuple{3, Int}


#-----------------------------------------------------------
# EachSitePairByDistance (combined_dir, src, trg)
#-----------------------------------------------------------

"""
    EachSitePairByDistance()

Creates an iterator template which returns triplets 
`(direction index, source, target)` sorted by distance. The `direction index` 
identifies each unique direction `position(target) - position(source)`.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachSitePairByDistance <: DeferredLatticeIterator end
EachSiteByDistance(::MonteCarloFlavor) = EachSiteByDistance()

function output_size(::EachSitePairByDistance, l::Lattice)
    B = length(l.unitcell.sites)
    Ndir = length(l[:Bravais_dir2srctrg])
    return (B, B, Ndir)
end

function _iterate(::EachSitePairByDistance, l::Lattice, state = (0, 1, 1, 1))
    B = length(l.unitcell.sites)
    dir2srctrg = l[:Bravais_dir2srctrg]
    uc1, uc2, dir, Bravais_src = state

    #= # IF structure
    if uc1 == B
        if uc2 == B
            if i == length(dir2srctrg[dir])
                if dir == length(dir2srctrg)
                    return nothing
                else
                    dir += 1
                end
                i = 1
            else
                i += 1
            end
            uc2 = 1
        else
            uc2 += 1
        end
        uc1 = 1
    else
        uc1 += 1
    end
    =#

    # branchless version:
    b1 = uc1 == B
    b2 = b1 && (uc2 == B)
    b3 = b2 && (Bravais_src == length(dir2srctrg[dir]))
    b4 = b3 && (dir == length(dir2srctrg))

    b4 && return nothing

    uc1 = Int(b1 || (uc1 + 1))
    uc2 = Int(b2 || (uc2 + b1))
    Bravais_src = Int(b3 || (Bravais_src + b2))
    dir = Int(dir + b3)

    # Bravais sites -> lattice sites
    Bravais_trg = dir2srctrg[dir][Bravais_src]
    src = (Bravais_src-1) * B + uc1
    trg = (Bravais_trg-1) * B + uc2

    # (uc1, uc2, dir) -> flat index
    combined_dir = _sub2ind((B, B), (uc1, uc2, dir))

    return ((combined_dir, src, trg), (uc1, uc2, dir, Bravais_src))
end
_length(::EachSitePairByDistance, l::Lattice) = length(l)^2
_eltype(::EachSitePairByDistance, l::Lattice) = NTuple{3, Int}


#-----------------------------------------------------------
# EachLocalQuadBySyncedDistance (combined_dir, src, src + Δ, trg, trg + Δ)
#-----------------------------------------------------------

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
and 6, but return 1, 2 and 3 as indices.

Requires `lattice` to implement `positions` and `lattice_vectors`.
"""
struct EachLocalQuadBySyncedDistance{T} <: DeferredLatticeIterator
    directions::T
    EachLocalQuadBySyncedDistance(N::Integer) = EachLocalQuadBySyncedDistance(1:N)
    EachLocalQuadBySyncedDistance(dirs::T) where T = new{T}(dirs)
end

function EachLocalQuadBySyncedDistance(::MonteCarloFlavor, arg)
    EachLocalQuadBySyncedDistance(arg)
end

function Base.:(==)(l::EachLocalQuadBySyncedDistance, r::EachLocalQuadBySyncedDistance)
    l.directions == r.directions
end

function output_size(iter::EachLocalQuadBySyncedDistance, l::Lattice)
    B = length(l.unitcell.sites)
    Ndir = length(l[:Bravais_dir2srctrg])
    K = length(iter.directions)
    return (B, B, Ndir, K)
end

function _iterate(iter::EachLocalQuadBySyncedDistance, l::Lattice, state = (1,1,0))
    #  sync_idx                sync_idx
    #     🡣                       🡣
    # (sync_dir, idx1)        (sync_dir, idx2)
    #    🡧        🡦             🡧        🡦
    # src' <----- src ------> trg -----> trg'
    #              🡣  🡦     🡧  🡣        
    #             uc1   dir   uc2

    sync_idx, idx1, idx2 = state

    # Branchless increments
    sync_dir = iter.directions[sync_idx]
    dir2srctrg = l[:dir2srctrg]
    Ndir = length(l[:Bravais_dir2srctrg])
    N = length(dir2srctrg[sync_dir])
    B = length(l.unitcell.sites)

    b1 = idx2 == N
    b2 = b1 && (idx1 == N)
    b3 = b2 && (sync_idx == length(iter.directions))

    b3 && return nothing

    idx2 = Int64(b1 || (idx2 + 1))
    idx1 = Int64(b2 || (idx1 + b1))
    sync_idx = Int64(sync_idx + b2)

    sync_dir = iter.directions[sync_idx]
    
    # These are lattice indices
    src1, trg1 = dir2srctrg[sync_dir][idx1]
    src2, trg2 = dir2srctrg[sync_dir][idx2]
    
    # convert to uc idx + Bravais lattice index
    Bsrc1, uc1 = fldmod1(src1, B)
    Bsrc2, uc2 = fldmod1(src2, B)
    dir12 = l[:Bravais_srctrg2dir][Bsrc1, Bsrc2]

    # combine (uc1, uc2, dir12, sync_idx) to linear index
    combined_dir = _sub2ind((B, B, Ndir, length(iter.directions)), (uc1, uc2, dir12, sync_idx))

    # state = (src1 mask index, src2 mask index, filter1 index, filter2 index)
    return ((combined_dir, src1, trg1, src2, trg2), (sync_idx, idx1, idx2))
end


function _length(iter::EachLocalQuadBySyncedDistance, l::Lattice)
    dir2srctrg = l[:dir2srctrg]
    return mapreduce(dir -> length(dir2srctrg[dir])^2, +, iter.directions)
end
_eltype(::EachLocalQuadBySyncedDistance, ::Lattice) = NTuple{5, Int64}


#-----------------------------------------------------------
# EachLocalQuadByDistance (combined_dir, src, src', trg, trg')
#-----------------------------------------------------------


"""
    EachLocalQuadByDistance(directions)

Returns a lattice iterator that iterates through combinations of four sites.

In each step the iterator returns `(out_idx, src, src', trg, trg')`. The first
index combines `(b1, b2, dir, idxs1, idx2)` where `b` refer to site indices
within the unit cell, `dir` refers to a direction on the Bravais lattice and 
idx1, idx2 refer to indices into `directions`. This can be visualized as

```
          b1      b2
          ↓       ↓
src' <-- src --> trg --> trg'
      ↑       ↑       ↑
    dir1     dir     dir2
      ↑               ↑
    idx1             idx2
```

"""
struct EachLocalQuadByDistance{T} <: DeferredLatticeIterator
    directions::T
    
    EachLocalQuadByDistance(N::Integer) = EachLocalQuadByDistance(1:N)
    EachLocalQuadByDistance(dirs::BondDirections) = new{BondDirections}(dirs)
    function EachLocalQuadByDistance(dirs)
        pairs = Pair.(eachindex(dirs), dirs)
        new{typeof(pairs)}(pairs)
    end
end

function EachLocalQuadByDistance(::MonteCarloFlavor, arg)
    EachLocalQuadByDistance(arg)
end

function Base.:(==)(l::EachLocalQuadByDistance, r::EachLocalQuadByDistance)
    l.directions == r.directions
end

function output_size(iter::EachLocalQuadByDistance, l::Lattice)
    B = length(l.unitcell.sites)
    Ndir = length(l[:Bravais_dir2srctrg]) # TODO this is a lot for just a number
    K = length(iter.directions)
    return (B, B, Ndir, K, K)
end
function output_size(iter::EachLocalQuadByDistance{BondDirections}, l::Lattice)
    B = length(l.unitcell.sites)
    Ndir = length(l[:Bravais_dir2srctrg]) # TODO this is a lot for just a number
    K = length(hopping_directions(l))
    return (B, B, Ndir, K, K)
end

function _iterate(iter::EachLocalQuadByDistance, l::Lattice, state = (1,1, 1,1))
    state == (0,0,0,0) && return nothing
    src1, src2, sub_idx1, sub_idx2 = state

    # convert to uc idx + Bravais lattice index
    B = length(l.unitcell.sites)
    Bsrc1, uc1 = fldmod1(src1, B)
    Bsrc2, uc2 = fldmod1(src2, B)
    dir12 = l[:Bravais_srctrg2dir][Bsrc1, Bsrc2]

    # Get src -- trg directions (idx is for the output matrix)
    dirs1 = _dir_idxs_uc(l, iter.directions, uc1)
    idx1, sub_dir1 = dirs1[sub_idx1]
    dirs2 = _dir_idxs_uc(l, iter.directions, uc2)
    idx2, sub_dir2 = dirs2[sub_idx2]

    # target sites
    srcdir2trg = l[:srcdir2trg]
    trg1 = srcdir2trg[src1, sub_dir1]
    trg2 = srcdir2trg[src2, sub_dir2]

    # Prepare next iteration

    # fast changing
    b1 = sub_idx2 == length(dirs2)
    b2 = b1 && (sub_idx1 == length(dirs1))
    b3 = b2 && (src2 == length(l))
    b4 = b3 && (src1 == length(l))
    # slow changing

    sub_idx2 = Int64(b1 || (sub_idx2 + 1))
    sub_idx1 = Int64(b2 || (sub_idx1 + b1))
    next_src2 = Int64(b3 || (src2 + b2))
    next_src1 = Int64(src1 + b3)

    next_state = Int(!b4) .* (next_src1, next_src2, sub_idx1, sub_idx2)

    # Check validity
    if trg1 == 0 || trg2 == 0
        return _iterate(iter, l, next_state)
    end

    # TODO
    subN = _length(l, iter.directions)
    Ndir = length(l[:Bravais_dir2srctrg])
    combined_dir = _sub2ind(
        (B, B, Ndir, subN, subN), (uc1, uc2, dir12, idx1, idx2)
    )

    return ((combined_dir, src1, trg1, src2, trg2), next_state)
end


function _length(iter::EachLocalQuadByDistance, l::Lattice)
    dir2srctrg = l[:dir2srctrg]
    return mapreduce(dir -> length(dir2srctrg[dir[2]]), +, iter.directions)^2
end
_eltype(::EachLocalQuadByDistance, ::Lattice) = NTuple{5, Int64}


################################################################################
### Temp friends
################################################################################

struct Sum{T} <: DeferredLatticeIterator
    iter::T
    # drop::Vector{Bool}
    # Sum(iter::AbstractLatticeIterator, drop = Bool[]) = new(iter, drop)
end
Sum(::MonteCarloFlavor, arg) = Sum(arg)

function _iterate(iter::Sum, l::Lattice)
    idxs_state = _iterate(iter.iter, l)
    idxs_state === nothing && return nothing
    return (tuple(1, idxs_state[1][2:end]...), idxs_state[2])
end
function _iterate(iter::Sum, l::Lattice, state)
    idxs_state = _iterate(iter.iter, l, state)
    idxs_state === nothing && return nothing
    return (tuple(1, idxs_state[1][2:end]...), idxs_state[2])
end

# In this case the get and set indices are the same and the iterator only 
# returns one index (or set of indices). So instead of adjusting indices we 
# want to add one here.
function _iterate(iter::Sum{<: DirectLatticeIterator}, l::Lattice)
    idxs_state = _iterate(iter.iter, l)
    idxs_state === nothing && return nothing
    return (tuple(1, idxs_state[1]), idxs_state[2])
end
function _iterate(iter::Sum{<: DirectLatticeIterator}, l::Lattice, state)
    idxs_state = _iterate(iter.iter, l, state)
    idxs_state === nothing && return nothing
    return (tuple(1, idxs_state[1]), idxs_state[2])
end

function _eltype(iter::Sum{<: DirectLatticeIterator}, l::Lattice)
    s = _size(iter, l); typeof(s) # Yuck. It's an NTuple{N+1, Int}
end
_size(iter::Sum{<: DirectLatticeIterator}, l::Lattice) = (1, _size(iter.iter, l)...)
output_size(::Sum, l::Lattice) = (1,)


# struct ApplySymmetries{IT, N, T} <: DeferredLatticeIterator
#     iter::IT
#     symmetries::NTuple{N, Vector{T}}
# end

const _all_lattice_iterator_types = [
    EachSiteAndFlavor, EachSite, EachSitePair, EachSiteByDistance,
    OnSite, EachLocalQuadByDistance, EachLocalQuadBySyncedDistance,
    Sum
]