################################################################################
### Lattice Iteration Cache
################################################################################


#=
We may need various schemes to iterate through the lattice in a simulation. For
example, DQMC uses various schemes for measurements, e.g. reducing a sum over 
all site pairs to all unique distances, direct iteration or iterations over 
combinations of four sites.
These iterators typically share some mappings among them, such as mapping a 
pair of (src, trg) sites to distances. The lattice iteration caches' job is to 
avoid duplicating this data across multiple types by storing it in a central 
way. It also allows this data to be initialized "on demand" rather than by 
default which helps keep memory usage down when inspecting data.
=#

function LatticeCache()
    constructors = Dict{Symbol, Function}(
        :Bravais_dir2srctrg => l -> construct_dir2srctrg(Bravais(l)), # Vector{Vector{Int}}}
        :Bravais_srctrg2dir => l -> construct_srctrg2dir(Bravais(l)), # Matrix{Int}
        :Bravais_srcdir2trg => l -> construct_srcdir2trg(Bravais(l)), # Matrix{Int}

        :dir2srctrg => construct_dir2srctrg, # Vector{Vector{Tuple{Int, Int}}}
        :src2dirtrg => construct_src2dirtrg, # Vector{Vector{Tuple{Int, Int}}}
        :srctrg2dir => construct_srctrg2dir, # Matrix{Int}
        :srcdir2trg => construct_srcdir2trg, # Matrix{Int}
        :uc2bonddir => construct_uc2bonddir, # Tuple{Int, Vector{Vector{Pair{Int, Int}}}}
    )

    return LatticeCache(Dict{Symbol, Any}(), constructors)
end



register!(l::Lattice, key::Symbol, f::Function) = l.cache.constructors[key] = f

# for simplicity
function Base.getindex(l::Lattice, key::Symbol)
    if haskey(l.cache.cache, key)
        return l.cache.cache[key]
    else 
        return l.cache.cache[key] = l.cache.constructors[key](l)
    end
end

function Base.get!(l::Lattice, key::Symbol, f::Function)
    if haskey(l.cache.cache, key)
        return l.cache.cache[key]
    else 
        l.cache.constructors[key] = f
        return l.cache.cache[key] = f(l)
    end
end

Base.empty(l::Lattice) = empty!(l.cache.cache)

construct_dir2srctrg(l::AbstractLattice) =  _dir2srctrg(l)

function construct_src2dirtrg(l::AbstractLattice)
    dir2srctrg = _dir2srctrg(l)
    trg_from_src = [Tuple{Int64, Int64}[] for _ in eachindex(l)]
    for dir in eachindex(dir2srctrg)
        for (src, trg) in _int_or_pair_to_pair(dir2srctrg[dir])
            push!(trg_from_src[src], (dir, trg))
        end
    end
    return trg_from_src
end

function construct_srctrg2dir(l::AbstractLattice)
    dir2srctrg = _dir2srctrg(l)
    srctrg2dir = [-1 for _ in eachindex(l), __ in eachindex(l)]
    for dir in eachindex(dir2srctrg)
        for (src, trg) in _int_or_pair_to_pair(dir2srctrg[dir])
            srctrg2dir[src, trg] = dir
        end
    end
    return srctrg2dir
end

function construct_srcdir2trg(l::AbstractLattice)
    dir2srctrg = _dir2srctrg(l)
    srcdir2trg = zeros(Int, length(l), length(dir2srctrg))
    for dir in eachindex(dir2srctrg)
        for (src, trg) in _int_or_pair_to_pair(dir2srctrg[dir])
            srcdir2trg[src, dir] = trg
        end
    end
    return srcdir2trg
end

# TODO
function construct_uc2bonddir(l::Lattice)
    dir2srctrg = _dir2srctrg(l)
    B = length(unitcell(l))

    # Find directional indices along bonds per basis index
    _bonds = map(1:B) do basis
        [MonteCarlo._shift_Bravais(l, 1, b) for b in l.unitcell.bonds if from(b) == basis]
    end
    bonddir = map(_bonds) do bs
        map(bs) do b
            for dir in 2:length(dir2srctrg)
                idx = findfirst(isequal((b.from, b.to)), dir2srctrg[dir])
                if idx !== nothing
                    return dir
                end
            end
            error("Failed to find directional index connection $(b.from) -> $(b.to)")
        end
    end

    # create a mapping idx => dir where each dir has one index and indices go
    # from 1:N in steps of 1. 
    alldirs = sort!(unique(vcat(bonddir...)))
    mapping = zeros(Int, maximum(alldirs))
    i = 1
    for dir in alldirs
        mapping[dir] = i
        i += 1
    end

    bonddir = map(bonddir) do dirs
        [mapping[dir] => dir for dir in dirs]
    end

    return (i-1, bonddir)
end


# Math helpers

# For shifting sites across periodic bounds
function generate_combinations(l::AbstractLattice)
    vs = size(l) .* lattice_vectors(l)
    out = [zeros(length(vs[1]))]
    for v in vs
        out = vcat([e.-v for e in out], out, [e.+v for e in out])
    end
    out
end

# This shifts the norm of a vector slightly based on the angle to the x-axis.
# norm + ϵ * angle(v, e_x)
function directed_norm(v, ϵ)
    l = norm(v)
    if (length(v) == 2) && (l > ϵ)
        angle = acos(dot([1, 0], v) / l)
        v[2] < 0 && (angle = 2pi - angle)
        return l + ϵ * angle
    else
        return l
    end
end
function directed_norm2(v, ϵ)
    if length(v) == 2
        angle = atan(v[2], v[1])
        angle += (angle < 0.0) * 2pi
        return dot(v, v) * (1.0 + ϵ * angle)
    else
        return dot(v, v)
    end
end

function _dir2srctrg(l::AbstractLattice, ϵ = 1e-6)
    _positions = collect(positions(l))
    wrap = generate_combinations(l)
    directions = Vector{Float64}[]
    sizehint!(directions, length(l))
    # (src, trg), first index is dir, second index irrelevant
    bonds = [Tuple{Int64, Int64}[] for _ in eachindex(l)]

    p0 = copy(first(_positions))
    d = copy(first(_positions))
    new_d = copy(first(_positions))

    for origin in eachindex(l)
        p0 = _positions[origin]
        for (trg, p) in enumerate(_positions)
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

            # search for d in directions
            idx = 0
            found = false
            for (i, dir) in enumerate(directions)
                new_d .= dir .- d
                b = true
                for v in new_d
                    b = b && (abs(v) < ϵ)
                end
                if b
                    idx = i
                    found = true
                    break
                end
            end

            if !found
                push!(directions, copy(d))
                if length(bonds) < length(directions)
                    push!(bonds, Tuple{Int64, Int64}[])
                end
                push!(bonds[length(directions)], (origin, trg))
            else
                push!(bonds[idx], (origin, trg))
            end
        end
    end

    temp = sortperm(directions, by = v -> directed_norm2(v, ϵ))
    return bonds[temp]
end


function _dir2srctrg(B::Bravais)
    l = B.l
    N = length(B)
    output = [Vector{Int}(undef, N) for _ in 1:N]

    for flat_shift in 1:N
        shift = _ind2sub(B, flat_shift) .- 1
        for flat_src in 1:N
            src = _ind2sub(B, flat_src)
            trg = mod1.(src .+ shift, l.Ls)
            flat_trg = _sub2ind(B, trg)
            output[flat_shift][flat_src] = flat_trg
        end
    end

    return output
end

_int_or_pair_to_pair(v::Vector{<: Tuple}) = v
_int_or_pair_to_pair(v::Vector{<: Integer}) = enumerate(v)