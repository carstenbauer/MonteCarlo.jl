"""
    Bond(from::Int, to::Int, uc_shift::NTuple{N, Int}[, label::Int = 1])
    Bond{N}(from::Int, to::Int, label::Int)

Constructs a bond going from a site `from` to a site `to`.

In the context of a unit cell `from` and `to` represent sites in it. `uc_shift`
represents a translation to another unitcell, given by
`sum(uc_shift .* unitcell.lattice_vector)`. For example on a square lattice the 
shifts `(1, 0), (-1, 0), (0, 1), (0, -1)` describe bonds that connect to a 
neighboring unit cell.

In the context of a lattice (i.e. through `neighbors(l)`) `from` and `to` 
represent sites on the lattice and `uc_shift = ntuple(_ -> 0, N)` for all bonds.
"""
struct Bond{N}
    from::Int
    to::Int
    uc_shift::NTuple{N, Int}
    label::Int
end
Bond(from::Int, to::Int, uc_shift::NTuple) = Bond(from, to, uc_shift, 1)
Bond{N}(from::Int, to::Int, label::Int) where N = Bond(from, to, ntuple(i -> 0, N), label)

function _is_reversal(b1::Bond{N}, b2::Bond{N}) where N
    b1.from == b2.to && b1.to == b2.from && 
    all(((a, b),) -> a == -b, zip(b1.uc_shift, b2.uc_shift))
end
from(b::Bond)  = b.from
to(b::Bond)    = b.to
label(b::Bond) = b.label



"""
    UnitCell(
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
struct UnitCell{N}
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

        new{N}(lattice_vectors, sites, bonds, directed)
    end
end




struct Lattice{N} <: AbstractLattice
    unitcell::UnitCell{N}
    Ls::NTuple{N, Int}
    # TODO lattice_iterator_cache maybe?
end


Base.length(l::Lattice) = length(l.unitcell.sites) * prod(l.Ls)
Base.size(l::Lattice) = l.Ls
Base.eachindex(l::Lattice) = 1:length(l)
lattice_vectors(l::Lattice) = l.unitcell.lattice_vectors

Base.:(==)(a::Lattice, b::Lattice) = a.Ls == b.Ls && a.unitcell == b.unitcell
function Base.:(==)(a::UnitCell, b::UnitCell)
    a.lattice_vectors == b.lattice_vectors && a.sites == b.sites &&
    all(a.bonds .== b.bonds) && a._directed_indices == b._directed_indices
end
function Base.:(==)(a::Bond, b::Bond)
    a.from == b.from && a.to == b.to && a.label == b.label && a.uc_shift == b.uc_shift
end

# flat index to tuple
function _ind2sub(l::Lattice, idx::Int)
    # this is kinda slow
    map((length(l.unitcell.sites), l.Ls...)) do L
        idx, x = fldmod1(idx, L)
        x
    end
end

# tuple to flat index
function _sub2ind(l::Lattice, t::NTuple)
    # this is very fast
    idx = t[end] - 1
    for d in length(l.Ls)-1:-1:2
        idx = idx * l.Ls[d] + (t[d] - 1)
    end
    return idx * length(l.unitcell.sites) + t[1]
end

# flat index shifted by change in Bravais indices
function _shift(l::Lattice{N}, flat::Int, shift::NTuple{N, Int}) where N
    # tested
    flat_fld, t = fldmod1(flat, length(l.unitcell.sites))

    flat_out = t
    f = length(l.unitcell.sites)
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    return flat_out
end

# flat index shifted by bond
function _shift(l::Lattice{N}, flat::Int, b::Bond{N}) where N
    @boundscheck begin
        b1 = 1 <= flat <= length(l) # flat in bounds
        flat_fld, t = fldmod1(flat, length(l.unitcell.sites)) # b.from correct
        b2 = b.from == t
        b1 && b2
    end

    flat_out = b.to
    flat_fld = fld1(flat, length(l.unitcell.sites))
    f = length(l.unitcell.sites)
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + b.uc_shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    return flat_out
end

# flat index shifted by bond
function _shift_Bravais(l::Lattice{N}, flat::Int, b::Bond{N}) where N
    @boundscheck 1 <= flat <= length(l) # flat in bounds

    flat_out = b.to
    flat_fld = fld1(flat+1, length(l.unitcell.sites))
    f = length(l.unitcell.sites)
    for d in eachindex(l.Ls)
        flat_fld, t = fldmod1(flat_fld, l.Ls[d])
        t = mod1(t + b.uc_shift[d], l.Ls[d])
        flat_out += f * (t - 1)
        f *= l.Ls[d]
    end

    return Bond{N}(flat + b.from, flat_out, b.label)
end

# """
#     neighbors(l::AbstractLattice[, directed=Val(false)])

# Returns an iterator over bonds, given as tuples (source index, target index). If
# `directed = Val(true)` bonds are assumed to be directed, i.e. both
# `(1, 2)` and `(2, 1)` are included. If `directed = Val(false)` bonds are
# assumed to be undirected, i.e. `(1, 2)` and `(2, 1)` are assumed to be
# equivalent and only one of them will be included.
# """
neighbors(l::Lattice) = neighbors(l, Val(false))
function neighbors(l::Lattice{N}, directed::Val{true}) where N
    (
        _shift_Bravais(l, idx, b)
            for idx in 0:length(l.unitcell.sites):length(l)-1
            for b in l.unitcell.bonds
    )
end
function neighbors(l::Lattice{N}, directed::Val{false}) where {N}
    bonds = l.unitcell.bonds
    (
        _shift_Bravais(l, idx, bonds[j])
            for idx in 0:length(l.unitcell.sites):length(l)-1
            for j in l.unitcell._directed_indices
    )
end
function neighbors(l::Lattice{N}, site::Int) where N
    uc_idx = mod1(site, length(l.unitcell.sites))
    (
        Bond{N}(site, _shift(l, site, b), b.label) 
            for b in l.unitcell.bonds if b.from == uc_idx
    )
end


# """
#     positions(l::Lattice)

# Returns an iterator containing all site positions on the lattice. 

# If collected this iterator will return a multidimensional Array. The indices of 
# this Array are [unitcell_sites, Bravais_x, Bravais_y, ...].
# """
function positions(l::Lattice{1})
    (o + lattice_vectors(l)[1] * i for o in l.unitcell.sites, i in 1:l.Ls[1] )
end
function positions(l::Lattice{2})
    origins = l.unitcell.sites
    v1, v2 = lattice_vectors(l)
    (o + v1 * i + v2 * j for o in origins, i in 1:l.Ls[1], j in 1:l.Ls[2])
end

# Arbitrary dimensions
function positions(l::Lattice{N}) where N
    # this does Bravais lattice positions + positions in unitcell
    (o + p for o in l.unitcell.sites, p in _positions(l, N))
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
    reciprocal_vectors(l::Lattice)

Returns the reciprocal unit vectors for 2 or 3 Dimensional lattices
"""
function reciprocal_vectors(l::Lattice{2})
    v1, v2 = lattice_vectors(l)
    V = 2pi / abs(v1[2] * v2[1] - v1[1] * v2[2])
    r1 = V * cross([v1[1], v1[2], 0.0], [0,0,1])
    r2 = V * cross([v2[1], v2[2], 0.0], [0,0,1])
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


function _save(file::FileLike, entryname::String, l::Lattice)
    write(file, "$entryname/VERSION", 0)
    write(file, "$entryname/tag", "MonteCarloLattice")
    write(file, "$entryname/Ls", l.Ls)
    write(file, "$entryname/uc/lv", l.unitcell.lattice_vectors)
    write(file, "$entryname/uc/sites", l.unitcell.sites)
    serialized = map(b -> (b.from, b.to, b.uc_shift...,  b.label), l.unitcell.bonds)
    write(file, "$entryname/uc/bonds", serialized)
    return
end

function _load(data, ::Val{:MonteCarloLattice})
    data["VERSION"] == 0 || @warn("Loading Version $(data["VERSION"]) as Version 0.")
    Lattice(
        UnitCell(
            data["uc/lv"],
            data["uc/sites"],
            map(t -> Bond(t[1], t[2], t[3:end-1], t[end]), data["uc/bonds"])
        ), data["Ls"]
    )
end

################################################################################
### Lattice Constructors
################################################################################



function Chain(Lx)
    uc = UnitCell(
        (Float64[1],),
        [Float64[0]],
        [Bond(1, 1, ( 1,)), Bond(1, 1, (-1,))]
    )

    Lattice(uc, (Lx,))
end

function SquareLattice(Lx, Ly = Lx)
    uc = UnitCell(
        (Float64[1, 0], Float64[0, 1]),
        [Float64[0, 0]],
        [
            Bond(1, 1, ( 1,  0)),
            Bond(1, 1, ( 0,  1)),
            Bond(1, 1, (-1,  0)),
            Bond(1, 1, ( 0, -1))
        ]
    )

    Lattice(uc, (Lx, Ly))
end

function CubicLattice(Lx, Ly = Lx, Lz = Lx)
    uc = UnitCell(
        (Float64[1, 0, 0], Float64[0, 1, 0], Float64[0, 0, 1]),
        [Float64[0, 0, 0]],
        [
            Bond(1, 1, ( 1,  0,  0)),
            Bond(1, 1, ( 0,  1,  0)),
            Bond(1, 1, ( 0,  0,  1)),
            Bond(1, 1, (-1,  0,  0)),
            Bond(1, 1, ( 0, -1,  0)),
            Bond(1, 1, ( 0,  0, -1))
        ]
    )

    Lattice(uc, (Lx, Ly, Lz))
end

function Honeycomb(Lx, Ly = Lx)
    uc = UnitCell(
        (Float64[sqrt(3.0)/2, -0.5], Float64[sqrt(3.0)/2, +0.5]),
        [Float64[0.0, 0.0], Float64[1/sqrt(3.0), 0.0]],
        [
            Bond(1, 2, (0, 0)), Bond(1, 2, (-1, 0)), Bond(1, 2, (0, -1)),
            Bond(2, 1, (0, 0)), Bond(2, 1, ( 1, 0)), Bond(2, 1, (0,  1)),
        ]
    )

    Lattice(uc, (Lx, Ly))
end