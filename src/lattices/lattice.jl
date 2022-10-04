# In case there is need to implement a different interface in the future
abstract type AbstractLattice end
abstract type AbstractBond end
abstract type AbstractUnitCell end

struct LatticeCache
    cache::Dict{Symbol, Any}
    constructors::Dict{Symbol, Function}
end

# to fix loading errors when Lattice is saved directly
JLD2.writeas(::Type{LatticeCache}) = String
JLD2.wconvert(::Type{String}, a::LatticeCache) = "LatticeCache"
JLD2.rconvert(::Type{LatticeCache}, a::String) = LatticeCache()



################################################################################
### Bonds
################################################################################


"""
    Bond(from::Int, to::Int, uc_shift::NTuple{N, Int}[, label::Int = 1])
    Bond{N}(from::Int, to::Int, label::Int)

Constructs a bond going from a site `from` to a site `to`.

In the context of a unit cell `from` and `to` represent sites in it. `uc_shift`
represents a translation to another unitcell, given by
`sum(uc_shift .* unitcell.lattice_vector)`. For example on a square lattice the 
shifts `(1, 0), (-1, 0), (0, 1), (0, -1)` describe bonds that connect to a 
neighboring unit cell.

In the context of a lattice (i.e. through `bonds(l)`) `from` and `to` 
represent sites on the lattice and `uc_shift = ntuple(_ -> 0, N)` for all bonds.
"""
struct Bond{N} <: AbstractBond
    from::Int
    to::Int
    uc_shift::NTuple{N, Int}
    label::Int
end
Bond(from::Int, to::Int, uc_shift::NTuple) = Bond(from, to, uc_shift, 1)
Bond{N}(from::Int, to::Int, label::Int) where N = Bond(from, to, ntuple(i -> 0, N), label)
Bond{N}(::Nothing) where N = Bond(0, 0, ntuple(i -> 0, N), 0)

function _is_reversal(b1::Bond{N}, b2::Bond{N}) where N
    b1.from == b2.to && b1.to == b2.from && 
    all(((a, b),) -> a == -b, zip(b1.uc_shift, b2.uc_shift))
end

"""
    from(b::Bond)

Returns the (linear) site index corresponding to the origin of a bond `b`.
"""
from(b::Bond)  = b.from

"""
    to(b::Bond)

Returns the (linear) site index corresponding to the target of a bond `b`.
"""
to(b::Bond)    = b.to

"""
    label(b::Bond)

Returns the label of a bond `b`.
"""
label(b::Bond) = b.label

function Base.:(==)(a::Bond, b::Bond)
    a.from == b.from && a.to == b.to && a.label == b.label && a.uc_shift == b.uc_shift
end


################################################################################
### UnitCell
################################################################################


"""
    UnitCell(
        name::String,
        lattice_vectors::NTuple{N, Vector{Float64}}, 
        sites::Vector{Vector{Float64}}, 
        bonds::Vector{Bonds{N}}
    )

The unit cell represents the (infinite) Bravais lattice, the sites that are 
scattered at each Bravais lattice position and the bonds between each site.

The Bravais lattice of an N dimensional lattice is given by `lattice_vectors`, 
which include N N-dimensional vectors.

The basis of the lattice, i.e. the positions of sites placed at each Bravais 
lattice position is given by `sites`. These positions are relative to the 
Bravais lattice positions.

Finally `bonds` describe pairs of "connected" sites. Reverse bonds (i.e. a -> b
and b -> a) must be included. See (@ref)[`Bond`] for more information.
"""
struct UnitCell{N} <: AbstractUnitCell
    name::String

    # Bravais lattice vectors point to nearby unit cells
    lattice_vectors::NTuple{N, Vector{Float64}}

    # site positions within the unit cell
    sites::Vector{Vector{Float64}}

    # bonds involving each site in a unit cell
    # these may cross over the boundary of this unit cell into the next
    bonds::Vector{Bond{N}}

    # indices into bonds that only include one bond direction
    _directed_indices::Vector{Int}

    function UnitCell(
            name::String,
            lattice_vectors::NTuple{N, Vector{Float64}}, 
            sites::Vector{Vector{Float64}}, 
            bonds::Vector{Bond{N}}
        ) where {N}

        directed = Int[]
        for (i, b) in enumerate(bonds)
            if !any(j -> _is_reversal(b, bonds[j]), directed)
                push!(directed, i)
            end
        end

        new{N}(name, lattice_vectors, sites, bonds, directed)
    end
end

function Base.:(==)(a::UnitCell, b::UnitCell)
    a.lattice_vectors == b.lattice_vectors && a.sites == b.sites &&
    all(a.bonds .== b.bonds) && a._directed_indices == b._directed_indices
end

positions(uc::UnitCell) = uc.sites
Base.length(uc::UnitCell) = length(uc.sites)

function reverse_bond_map(uc::UnitCell)
    map(uc.bonds) do b
        findfirst(uc.bonds) do r
            r.to == b.from &&
            r.from == b.to &&
            all(((x, y),) -> (x == -y), zip(r.uc_shift, b.uc_shift))
        end::Int
    end
end


################################################################################
### Lattice
################################################################################


struct Lattice{N} <: AbstractLattice
    unitcell::UnitCell{N}
    Ls::NTuple{N, Int}
    cache::LatticeCache
end

function Lattice(uc, Ls)
    l = Lattice(uc, Ls, LatticeCache())
    return l
end

function Base.show(io::IO, l::Lattice{N}) where N
    print(io, "Lattice{$N}($(l.unitcell.name), $(l.Ls))")
end

function Base.show(io::IO, ::MIME"text/plain", l::Lattice{N}) where N
    print(io, "$N-dimensional $(l.unitcell.name) Lattice\n")
    print(io, "\t Linear System Size L = $(l.Ls)\n")
    print(io, "\t Number of Sites N = $(length(l))\n")
    print(io, "\t Number of Sites in unitcell: $(length(l.unitcell.sites))")
end

"""
    length(l::Lattice)

Returns the total number of sites of a given lattice.
"""
Base.length(l::Lattice) = length(l.unitcell.sites) * prod(l.Ls)

"""
    size(l::Lattice)

Returns the linear system size of a given lattice. 

Note that this does not include the number of sites in a unit cell.
"""
Base.size(l::Lattice) = l.Ls

"""
    eachindex(l::Lattice)

Returns an iterator which iterates through each site of the lattice.
"""
Base.eachindex(l::AbstractLattice) = 1:length(l)

"""
    lattice_vectors(l::Lattice)

Returns the basis of the Bravais lattice. 

Note that the total extent of the lattice can be calculated as 
`size(l) .* lattice_vectors(l)`.
"""
lattice_vectors(l::Lattice) = l.unitcell.lattice_vectors

"""
    unitcell(l::Lattice)

Returns the unitcell of a given lattice.
"""
unitcell(l::Lattice) = l.unitcell

reverse_bond_map(l::AbstractLattice) = reverse_bond_map(unitcell(l))


Base.:(==)(a::Lattice, b::Lattice) = a.Ls == b.Ls && a.unitcell == b.unitcell

# flat index to tuple
function _ind2sub(l::Lattice, idx::Int)
    # this is kinda slow
    map((l.Ls..., length(l.unitcell.sites))) do L
        idx, x = fldmod1(idx, L)
        x
    end
end

# tuple to flat index
function _sub2ind(l::Lattice, t::NTuple)
    # this is very fast
    # idx = t[end] - 1
    # for d in length(l.Ls)-1:-1:1
    #     idx = idx * l.Ls[d] + (t[d+1] - 1)
    # end
    # return idx * length(l.unitcell.sites) + t[1]

    # t was (b, x, y, z), now (x, y, z, b)
    idx = t[end] - 1
    for d in length(l.Ls):-1:1
        idx = idx * l.Ls[d] + (t[d] - 1)
    end
    return idx + 1
end

# flat index shifted by change in Bravais indices
function _shift(l::Lattice{N}, flat::Int, shift::NTuple{N, Int}) where N
    # tested
    # flat_fld, t = fldmod1(flat, length(l.unitcell.sites))

    # flat_out = t
    # f = length(l.unitcell.sites)
    # for d in eachindex(l.Ls)
    #     flat_fld, t = fldmod1(flat_fld, l.Ls[d])
    #     t = mod1(t + shift[d], l.Ls[d])
    #     flat_out += f * (t - 1)
    #     f *= l.Ls[d]
    # end

    # return flat_out

    # new version
    flat_fld = flat
    flat_out = 1
    f = 1
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    return flat_out + (flat_fld-1) * f

end

# flat index shifted by bond
function _shift(l::Lattice{N}, flat::Int, b::Bond{N}) where N
    # @boundscheck begin
    #     b1 = 1 <= flat <= length(l) # flat in bounds
    #     flat_fld, t = fldmod1(flat, length(l.unitcell.sites)) # b.from correct
    #     b2 = b.from == t
    #     b1 && b2
    # end

    # flat_out = b.to
    # flat_fld = fld1(flat, length(l.unitcell.sites))
    # f = length(l.unitcell.sites)
    # for d in eachindex(l.Ls)
    #     flat_fld, t = fldmod1(flat_fld, l.Ls[d])
    #     t = mod1(t + b.uc_shift[d], l.Ls[d])
    #     flat_out += f * (t - 1)
    #     f *= l.Ls[d]
    # end

    # return flat_out

    # mostly the same as _shift
    @boundscheck 1 <= flat <= length(l) # flat in bounds

    flat_out = 1
    flat_fld = flat
    f = 1
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + b.uc_shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    @boundscheck flat_fld == b.from # b.from matches basis index in flat

    return flat_out + (b.to-1) * f
end

# flat index that doesn't include basis shifted by bond
function _shift_Bravais(l::Lattice{N}, flat::Int, b::Bond{N}) where N
    # @boundscheck 1 <= flat <= length(l) # flat in bounds

    # flat_out = b.to
    # flat_fld = fld1(flat, length(l.unitcell.sites))
    # f = length(l.unitcell.sites)
    # for d in eachindex(l.Ls)
    #     flat_fld, t = fldmod1(flat_fld, l.Ls[d])
    #     t = mod1(t + b.uc_shift[d], l.Ls[d])
    #     flat_out += f * (t - 1)
    #     f *= l.Ls[d]
    # end

    # return Bond{N}(flat + b.from - 1, flat_out, b.label)

    @boundscheck 1 <= flat <= length(l) # flat in bounds

    flat_out = 1
    flat_fld = flat
    f = 1
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + b.uc_shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    @boundscheck flat_fld == 0

    return Bond{N}(flat + (b.from - 1) * f, flat_out + (b.to - 1) * f, b.label)
end

"""
    bonds(l::AbstractLattice[, directed=Val(false)])

Returns an iterator over all bonds of the lattice. If `directed = Val(true)` this 
includes bonds going from site `a -> b` as well as `b -> a`. Otherwise only one
of these is given.
"""
bonds(l::AbstractLattice) = bonds(l, Val(false))
function bonds(l::Lattice{N}, directed::Val{true}) where N
    (
        _shift_Bravais(l, idx, b)
            for idx in 1:prod(l.Ls)
            for b in l.unitcell.bonds
            # for idx in 1:length(l.unitcell.sites):length(l)
    )
end
function bonds(l::Lattice{N}, directed::Val{false}) where {N}
    bonds = l.unitcell.bonds
    (
        _shift_Bravais(l, idx, bonds[j])
            for idx in 1:prod(l.Ls)
            for j in l.unitcell._directed_indices
            # for idx in 1:length(l.unitcell.sites):length(l)
    )
end

"""
    bonds(l::AbstractLattice, source::Int)

Returns an iterator over all bonds starting at `source`.
"""
function bonds(l::Lattice{N}, site::Int) where N
    uc_idx = div(site-1, prod(l.Ls)) + 1
    (
        Bond{N}(site, _shift(l, site, b), b.label) 
            for b in l.unitcell.bonds if b.from == uc_idx
    )
end





"""
    positions(l::Lattice)

Returns an iterator containing positions of all sites on the lattice. 

If collected this iterator will return a multidimensional Array. The indices of 
this Array are [unitcell_sites, Bravais_x, Bravais_y, ...]. It can also be 
indexed with a linear site index.
"""
function positions(l::Lattice{1})
    (o + lattice_vectors(l)[1] * i for i in 1:l.Ls[1], o in l.unitcell.sites)
end
function positions(l::Lattice{2})
    origins = l.unitcell.sites
    v1, v2 = lattice_vectors(l)
    (o + v1 * i + v2 * j for i in 1:l.Ls[1], j in 1:l.Ls[2], o in origins)
end

# Arbitrary dimensions
function positions(l::Lattice{N}) where N
    # this does Bravais lattice positions + positions in unitcell
    (o + p for p in _positions(l, N), o in l.unitcell.sites)
end
function _positions(l::Lattice, N)
    v = lattice_vectors(l)[N]
    if N == 1
        return (v * i for i in 1:l.Ls[N])
    else
        return (p + v * i  for p in _positions(l, N-1), i in 1:l.Ls[N])
    end
end

"""
    position(l::Lattice, uc_index, idxs...)

Returns the position of a Lattice at a given set of lattice indices. The indices 
are sorted (basis index, x index, y index, z index)
"""
function position(l::Lattice, idxs...)
    return sum(lattice_vectors(l) .* idxs[1:end-1]) .+ l.unitcell.sites[idxs[end]]
end


"""
    reciprocal_vectors(l::Lattice)

Returns the reciprocal unit vectors for 2 or 3 dimensional lattice.
"""
function reciprocal_vectors(l::Lattice{2})
    v1, v2 = lattice_vectors(l)
    V = 2pi / abs(v1[2] * v2[1] - v1[1] * v2[2])
    r1 = V * [-v2[2], v2[1]] # R(90°) matrix applied 
    r2 = V * [-v1[2], v1[1]]
    return r1, r2
end

function reciprocal_vectors(l::Lattice{3})
    v1, v2, v3 = lattice_vectors(l)
    V = 2pi / dot(v1, cross(v2, v3))
    r1 = V * cross(v2, v3)
    r2 = V * cross(v3, v1)
    r3 = V * cross(v1, v2)
    return r1, r2, r3
end

function ReciprocalLattice(l::Lattice{D}, scale = 1.0) where {D}
    uc = UnitCell(
        "reciprocal " * l.unitcell.name,
        scale .* reciprocal_vectors(l),
        [zeros(D)],
        unique(map(b -> Bond(1, 1, b.uc_shift), l.unitcell.bonds))
    )
    Lattice(uc, l.Ls)
end


function _save(file::FileLike, entryname::String, l::Lattice)
    write(file, "$entryname/VERSION", 1)
    write(file, "$entryname/tag", "MonteCarloLattice")
    write(file, "$entryname/Ls", l.Ls)
    write(file, "$entryname/uc/name", l.unitcell.name)
    write(file, "$entryname/uc/lv", l.unitcell.lattice_vectors)
    write(file, "$entryname/uc/sites", l.unitcell.sites)
    serialized = map(b -> (b.from, b.to, b.uc_shift...,  b.label), l.unitcell.bonds)
    write(file, "$entryname/uc/bonds", serialized)
    return
end

function _load(data, ::Val{:MonteCarloLattice})
    version = data["VERSION"]
    if version == 0
        @warn("Loading a Lattice with index order (Basis, Bravais) as (Bravais, basis)")
    elseif version != 1
        @warn("Loading Version $(data["VERSION"]) as Version 1.")
    end
    Lattice(
        UnitCell(
            data["uc/name"],
            data["uc/lv"],
            data["uc/sites"],
            map(t -> Bond(t[1], t[2], t[3:end-1], t[end]), data["uc/bonds"])
        ), data["Ls"]
    )
end


################################################################################
### Bravais Lattice
################################################################################


"""
    Bravais(l::Lattice)

This wrapper tells functions to only consider the Bravais lattice, i.e. 
disregard sites within a unitcell.
"""
struct Bravais{N} <: AbstractLattice
    l::Lattice{N}
end

"""
    positions(l::Bravais)

Returns an iterator containing all site positions on the Bravais lattice. 
Explicitly this means ignoring offsets from sites within the unitcell.

If collected this iterator will return a multidimensional Array. The indices of 
this Array are [Bravais_x, Bravais_y, ...].
"""
positions(b::Bravais{1}) = (lattice_vectors(b.l)[1] * i for i in 1:b.l.Ls[1])
function positions(b::Bravais{2})
    v1, v2 = lattice_vectors(b.l)
    (v1 * i + v2 * j for i in 1:b.l.Ls[1], j in 1:b.l.Ls[2])
end

# Arbitrary dimensions
positions(b::Bravais{N}) where N = _positions(b.l, N)
lattice_vectors(b::Bravais) = lattice_vectors(b.l)
Base.size(b::Bravais) = size(b.l)
Base.length(b::Bravais) = prod(b.l.Ls)
Base.eachindex(b::Bravais) = 1:length(b)

function _ind2sub(B::Bravais, idx::Int)
    # this is kinda slow
    map(B.l.Ls) do L
        idx, x = fldmod1(idx, L)
        x
    end
end

function _sub2ind(B::Bravais, t::NTuple)
    # t was (b, x, y, z), now (x, y, z, b)
    idx = t[end] - 1
    for d in length(B.l.Ls)-1:-1:1
        idx = idx * B.l.Ls[d] + (t[d] - 1)
    end
    return idx + 1
end


################################################################################
### Open Boundary
################################################################################



# flat index shifted by bond
function _shift_Bravais_open(l::Lattice{N}, flat::Int, b::Bond{N}) where N
    # @boundscheck 1 <= flat <= length(l) # flat in bounds

    # flat_out = b.to
    # flat_fld = fld1(flat, length(l.unitcell.sites))
    # f = length(l.unitcell.sites)
    # for d in eachindex(l.Ls)
    #     flat_fld, t = fldmod1(flat_fld, l.Ls[d])
    #     # t = mod1(t + b.uc_shift[d], l.Ls[d])
    #     t = t + b.uc_shift[d]
    #     1 <= t <= l.Ls[d] || return Bond{N}(nothing)
    #     flat_out += f * (t - 1)
    #     f *= l.Ls[d]
    # end

    # return Bond{N}(flat + b.from - 1, flat_out, b.label)

    @boundscheck 1 <= flat <= length(l) # flat in bounds

    flat_out = 1
    flat_fld = flat
    f = 1
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        # t = mod1(t + b.uc_shift[d], l.Ls[d])
        t = t + b.uc_shift[d]
        1 <= t <= l.Ls[d] || return Bond{N}(nothing)
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    @boundscheck flat_fld == 0

    return Bond{N}(flat + (b.from - 1) * f, flat_out + (b.to - 1) * f, b.label)
end

struct OpenBondIterator{N}
    l::Lattice{N}
    directed::Bool
end

# TODO: Can we derive length from wrapping + lattice size?
Base.IteratorSize(::OpenBondIterator) = Base.SizeUnknown()
Base.eltype(::OpenBondIterator{N}) where N = Bond{N}

bonds_open(l::AbstractLattice, ::Val{true}) = bonds_open(l, true)
bonds_open(l::AbstractLattice, ::Val{false}) = bonds_open(l, false)
bonds_open(l::AbstractLattice, directed = false) = OpenBondIterator(l, directed)

function Base.iterate(iter::OpenBondIterator{N}, state = (1, 1)) where N
    flat, bond_idx = state
    flat > prod(iter.l.Ls) && return nothing
    uc = iter.l.unitcell

    if iter.directed
        bond = _shift_Bravais_open(iter.l, flat, uc.bonds[bond_idx])
        next_flat = flat + Int(bond_idx == length(uc.bonds))
        next_bond_idx = mod1(bond_idx+1, length(uc.bonds))
    else
        bond = _shift_Bravais_open(iter.l, flat, uc.bonds[uc._directed_indices[bond_idx]])
        next_flat = flat + Int(bond_idx == length(uc._directed_indices))
        next_bond_idx = mod1(bond_idx+1, length(uc._directed_indices))
    end

    if bond == Bond{N}(nothing)
        return iterate(iter, (next_flat, next_bond_idx))
    else
        return bond, (next_flat, next_bond_idx)
    end
end