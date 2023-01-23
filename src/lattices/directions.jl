"""
    directions(lattice)

Returns all (non-equivalent) directions in a given lattice. Note that you can
pass a `::Lattice` to consider all sites or a `::Bravais` to consider just 
Bravais lattice sites.

See also: [`directions_with_uc`](@ref)
"""
function directions(lattice::AbstractLattice, ϵ = 1e-6)
    _positions = collect(positions(lattice))
    wrap = generate_combinations(lattice)
    directions = Vector{Float64}[]
    sizehint!(directions, length(lattice))

    d = copy(first(_positions))
    new_d = copy(first(_positions))

    for p0 in _positions
        for p in _positions
            _apply_wrap!(d, p, p0, wrap, new_d, ϵ)

            # search for d in directions
            # if not present push it, otherwise continue with next iteration
            for dir in directions
                new_d .= dir .- d
                b = true
                for v in new_d
                    b = b && (abs(v) < ϵ)
                end
                if b # all_below(new_d, ϵ)
                    @goto loop_end
                end
            end

            push!(directions, copy(d))

            @label loop_end
        end
    end

    return sort!(directions, by = v -> directed_norm2(v, ϵ))
end

function _apply_wrap!(d, p, p0, wrap, new_d = similar(d), ϵ = 1e-6)
    d .= p .- p0 .+ wrap[1]
    n2 = directed_norm2(d, ϵ)

    for v in wrap[2:end]
        new_d .= p .- p0 .+ v
        new_n2 = directed_norm2(new_d, ϵ)
        if new_n2 + 100eps(n2) < n2
            d .= new_d
            n2 = new_n2
        end
    end

    return d
end


function directions(lattice::Bravais)
    _positions = collect(positions(lattice))
    return [p - _positions[1] for p in _positions[:]]
end


# """
#     directions_with_uc(lattice::Lattice)

# Returns a tuple `(trg_uc, src_uc, dir_vec)` for each (non-equivalent) direction 
# in the given lattice.

# See also: [`directions`](@ref)
# """
# function directions_with_uc(lattice::Lattice, ϵ = 1e-6)
#     _positions = collect(positions(lattice))
#     wrap = generate_combinations(lattice_vectors(lattice))
#     directions = Tuple{Int, Int, Vector{Float64}}[]
#     B = length(lattice.unitcell.sites)

#     for origin in 1:length(lattice)
#         src_uc = mod1(origin, B)
#         for (trg, p) in enumerate(_positions)
#             d = p .- _positions[origin] .+ wrap[1]
#             for v in wrap[2:end]
#                 new_d = p .- _positions[origin] .+ v
#                 if directed_norm(new_d, ϵ) + 100eps(Float64) < directed_norm(d, ϵ)
#                     d .= new_d
#                 end
#             end
#
#             # TODO this is not ok. The same direction can appear with multiple
#             # basis indices. (e.g. a lattice vector can start from any basis site)
#             idx = findfirst(dir -> isapprox(dir, d, atol=ϵ), directions)
#             if idx === nothing
#                 trg_uc = mod1(trg, B)
#                 push!(directions, (src_uc, trg_uc, d))
#             end
#         end
#     end

#     return sort!(directions, by = v -> directed_norm(v, ϵ))
# end


################################################################################
### Iterator methods
################################################################################

"""
    directions(::EachSitePairByDistance, ::Lattice)

Returns an array of directions matching the output shape of 
`EachSitePairByDistance`.

The indices of `EachSitePairByDistance` represent 
`(source basis, target basis, Bravais directional index)`. These are transformed 
into composite directions here, using the same indexing.
"""
function directions(l::Lattice, ::EachSitePairByDistance)
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for dir in Bravais_dirs, p1 in ps, p2 in ps]
end

"""
    directions(iter::EachLocalQuadByDistance, ::Lattice)
    directions(iter::EachLocalQuadBySyncedDistance, ::Lattice)

Returns an array of composite directions (basis + Bravais directions) like the 
method for `EachSitePairByDistance` and another vector of directions matching 
the directions given to the iterators.

To be more specific we use the sketch from `EachLocalQuadByDistance`:

```
          b1      b2
          ↓       ↓
src' --- src --- trg --- trg'
      ↑       ↑       ↑
    dir1     dir     dir2
      ↑               ↑
    idx1             idx2
```

The indices of the observable (as well as mean and standard error thereof) are 
`(b1, b2, dir, idx1, idx2)` which represent the basis, Bravais directions and 
indices into `iter.directions`. The first array returned here combines `b1`, 
`b2` and `dir` into composite directions, using the same indices as the first 
3 dimensions of the observable. The second array represents `dir1`/`dir2`, i.e. 
`directions(l)[iter.directions]`.
"""
function directions(l::Lattice, iter::EachLocalQuadBySyncedDistance)
    sub_dirs = directions(l)[iter.directions]
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for dir in Bravais_dirs, p1 in ps, p2 in ps], sub_dirs
end


function directions(l::Lattice, iter::EachLocalQuadByDistance)
    sub_dirs = directions(l)[last.(iter.directions)]
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for dir in Bravais_dirs, p1 in ps, p2 in ps], sub_dirs
end


################################################################################
### Other Utilities
################################################################################


"""
    nearest_neighbor_count(l::Lattice[, ϵ = 1e-6])

Returns the number of nearest neighbors on the lattice.
"""
function nearest_neighbor_count(l::Lattice, ϵ = 1e-6)
    lattice_directions = directions(l)
    dir_idxs = hopping_directions(l, ϵ, lattice_directions)
    distances = map(dir_idxs) do dir_idx
        r = lattice_directions[dir_idx]
        return dot(r, r)
    end

    min_dist = minimum(distances)
    return sum(d < min_dist + ϵ for d in distances)
end

"""
    hopping_directions(l::Lattice[, ϵ = 1e-6, lattice_directions = directions(l)])

Returns the indices of every hopping direction defined on the lattice. (This 
includes hoppings beyond NN if they are available.)
"""
function hopping_directions(l::Lattice{N}, ϵ = 1e-6, lattice_directions = directions(l)) where N
    uc = unitcell(l)
    r = Vector{Float64}(undef, N)

    valid_directions = map(uc.bonds) do bond
        p0 = uc.sites[from(bond)]
        p1 = uc.sites[to(bond)]
        shift = sum(bond.uc_shift .* uc.lattice_vectors)
        @. r = p1 - p0 + shift
        findfirst(eachindex(lattice_directions)) do idx
            isapprox(r, lattice_directions[idx], atol = ϵ)
        end
    end

    try
        return sort!(unique(valid_directions))
    catch e
        # The error caught here is `isless(::Nothing, ::Real) does not exist`.
        # `nothing` appears because there is no direction matching the bond 
        # direction for `findfirst` to find. With wrapping this can be 
        # circumvented, but that breaks assumptions in summations so it's better
        # to just error.
        error("The given lattice is too small to include all hopping directions.")
    end
end
