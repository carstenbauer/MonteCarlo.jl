abstract type AbstractLatticeIterator end



################################################################################
### EachSite
################################################################################



"""
    EachSite(Nsites)
    EachSite(lattice)
    EachSite(mc, model)

Creates an iterator which iterates through every site of a given lattice.
"""
struct EachSite <: AbstractLatticeIterator
    N::Int64
end
EachSite(l::AbstractLattice) = EachSite(length(l))
EachSite(mc::MonteCarloFlavor, model::Model) = EachSite(lattice(model))
eachsite(args...) = EachSite(args...)

Base.iterate(iter::EachSite, i=1) = i ≤ iter.N ? (i, i+1) : nothing 
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
struct OnSite <: AbstractLatticeIterator
    N::Int64
end
OnSite(l::AbstractLattice) = OnSite(length(l))
OnSite(mc::MonteCarloFlavor, model::Model) = OnSite(lattice(model))
onsite(args...) = OnSite(args...)

Base.iterate(iter::OnSite, i=1) = i ≤ iter.N ? ((i, i), i+1) : nothing 
Base.length(iter::OnSite) = iter.N
Base.eltype(::OnSite) = Int64



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
struct EachSitePair <: AbstractLatticeIterator
    N::Int64
end
EachSitePair(l::AbstractLattice) = EachSitePair(length(l))
EachSitePair(mc::MonteCarloFlavor, model::Model) = EachSitePair(lattice(model))
eachsitepair(args...) = EachSitePair(args...)

function Base.iterate(iter::EachSitePair, i=1)
    if i ≤ iter.N^2
        return ((div(i-1, iter.N)+1, mod1(i, iter.N)), i+1)
    else
        return nothing 
    end
end
Base.length(iter::EachSitePair) = iter.N
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
struct EachSitePairByDistance <: AbstractLatticeIterator
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

function EachSitePairByDistance(lattice::AbstractLattice)
    _positions = positions(lattice)
    wrap = generate_combinations(lattice_vectors(lattice))
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
            # The rounding will allow us to use == here
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
    EachSitePairByDistance(length(lattice)^2, bonds[temp])
end
function EachSitePairByDistance(mc::MonteCarloFlavor, model::Model)
    EachSitePairByDistance(lattice(model))
end

function Base.iterate(iter::EachSitePairByDistance, state = (1, 1))
    dir, i = state
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
ndirections(iter::EachSitePairByDistance) = length(iter.pairs)
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



# TODO
# It would be better if this kept (dir, i, j)

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
struct EachLocalQuadByDistance{K} <: AbstractLatticeIterator
    N::Int64
    pairs_by_dir::Vector{Vector{Tuple{Int64, Int64}}}
    trg_from_src::Vector{Vector{Tuple{Int64, Int64}}}
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

    total_length = sum(length(x) for x in trg_from_src)^2
    EachLocalQuadByDistance{K}(total_length, pairs_by_dir.pairs, trg_from_src)
end
function EachLocalQuadByDistance{K}(mc::MonteCarloFlavor, model::Model) where {K}
    EachLocalQuadByDistance{K}(lattice(model))
end


function Base.iterate(iter::EachLocalQuadByDistance, state = (1, 1, 1, 1))
    dir12, idx, i, j = state
    if dir12 ≤ length(iter.pairs_by_dir)
        if idx ≤ length(iter.pairs_by_dir[dir12])
            src1, src2 = iter.pairs_by_dir[dir12][idx]
            if i ≤ length(iter.trg_from_src[src1])
                dir1, trg1 = iter.trg_from_src[src1][i]
                if j ≤ length(iter.trg_from_src[src2])
                    dir2, trg2 = iter.trg_from_src[src2][j]
                    return (
                        (dir12, dir1, dir2, src1, trg1, src2, trg2), 
                        (dir12, idx, i, j+1)
                    )
                else
                    return iterate(iter, (dir12, idx, i+1, 1))
                end
            else
                return iterate(iter, (dir12, idx+1, 1, 1))
            end

        else
            return iterate(iter, (dir12+1, 1, 1, 1))
        end
    else
        return nothing
    end
end
ndirections(iter::EachLocalQuadByDistance{K}) where {K} = (length(iter.pairs_by_dir), K, K)
Base.length(iter::EachLocalQuadByDistance) = iter.N
Base.eltype(::EachLocalQuadByDistance) = NTuple{7, Int64}



################################################################################
### Additonal stuff
################################################################################



# function directions(::EachSitePair, lattice::AbstractLattice)
#     pos = positions(lattice)
#     [p2 .- p1 for p2 in pos for p1 in pos]
# end


function directions(iter::EachSitePairByDistance, lattice::AbstractLattice)
    pos = MonteCarlo.positions(lattice)
    wrap = generate_combinations(latticeVectors(lattice.lattice))

    dirs = map(iter.pairs) do pairs
        src, trg = pairs[1]
        _d = pos[trg] - pos[src]
        # Find lowest distance w/ periodic bounds
        d = round.(_d .+ wrap[1], digits=6)
        for v in wrap[2:end]
            new_d = round.(_d .+ v, digits=6)
            if norm(new_d) < norm(d)
                d .= new_d
            end
        end
        d
    end
end