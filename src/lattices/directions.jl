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

    for origin in 1:length(lattice)
        for (trg, p) in enumerate(_positions)
            d = p .- _positions[origin] .+ wrap[1]
            for v in wrap[2:end]
                new_d = p .- _positions[origin] .+ v
                if directed_norm(new_d, ϵ) + 100eps(Float64) < directed_norm(d, ϵ)
                    d .= new_d
                end
            end

            idx = findfirst(dir -> isapprox(dir, d, atol=ϵ), directions)
            if idx === nothing
                push!(directions, d)
            end
        end
    end

    return sort!(directions, by = v -> directed_norm(v, ϵ))
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
function directions(::EachSitePairByDistance, l::Lattice)
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for p1 in ps, p2 in ps, dir in Bravais_dirs]
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
function directions(iter::EachLocalQuadBySyncedDistance, l::Lattice)
    sub_dirs = directions(l)[iter.directions]
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for p1 in ps, p2 in ps, dir in Bravais_dirs], sub_dirs
end


function directions(iter::EachLocalQuadByDistance, l::Lattice)
    sub_dirs = directions(l)[iter.directions]
    Bravais_dirs = directions(Bravais(l))
    ps = l.unitcell.sites
    return [p2 - p1 + dir for p1 in ps, p2 in ps, dir in Bravais_dirs], sub_dirs
end