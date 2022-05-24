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
    LatticeCache(
        LazyData{Vector{Vector{Int}}}(),
        LazyData{Matrix{Int}}(),
        LazyData{Vector{Vector{Tuple{Int, Int}}}}(),
        LazyData{Vector{Vector{Tuple{Int, Int}}}}(),
        LazyData{Matrix{Int}}()
    )
end

function init!(c::LatticeCache, l::Lattice)
    c.Bravais_dir2srctrg.constructor = () -> construct_dir2srctrg(Bravais(l))
    c.Bravais_srctrg2dir.constructor = () -> construct_srctrg2dir(Bravais(l))

    c.dir2srctrg.constructor = () -> construct_dir2srctrg(l)
    c.src2dirtrg.constructor = () -> construct_src2dirtrg(l)
    c.srctrg2dir.constructor = () -> construct_srctrg2dir(l)
    return
end

# for simplicity
Base.getindex(l::Lattice, key::Symbol) = value(getproperty(l.cache, key))


construct_dir2srctrg(l::AbstractLattice) =  _dir2srctrg(l)
function construct_dir2srctrg(b::Bravais)
    dir2srctrg = _dir2srctrg(b)
    N = length(b)

    if !(length(dir2srctrg) == N && all(vec -> length(vec) == N, dir2srctrg))
        throw(AssertionError(
            "There must be exactly one direction associated with each site " *
            "pair of a Bravais lattice. This is currently not the case, which " *
            "means there is a bug in `_dir2srctrg`. \n$dir2srctrg"
        ))
    end

    return map(dir2srctrg) do pairs_in_direction
        out = Vector{Int}(undef, N)
        for (src, trg) in pairs_in_direction
            out[src] = trg
        end
        return out
    end
end

function construct_src2dirtrg(l::AbstractLattice)
    dir2srctrg = _dir2srctrg(l)
    trg_from_src = [Tuple{Int64, Int64}[] for _ in eachindex(l)]
    for dir in eachindex(dir2srctrg)
        for (src, trg) in dir2srctrg[dir]
            push!(trg_from_src[src], (dir, trg))
        end
    end
    return trg_from_src
end

function construct_srctrg2dir(l::AbstractLattice)
    dir2srctrg = _dir2srctrg(l)
    srctrg2dir = [-1 for _ in eachindex(l), __ in eachindex(l)]
    for dir in eachindex(dir2srctrg)
        for (src, trg) in dir2srctrg[dir]
            srctrg2dir[src, trg] = dir
        end
    end
    return srctrg2dir
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
    if length(v) == 2 && l > ϵ
        angle = acos(dot([1, 0], v) / l)
        v[2] < 0 && (angle = 2pi - angle)
        return l + ϵ * angle
    else
        return l
    end
end


function _dir2srctrg(l::AbstractLattice, ϵ = 1e-6)
    _positions = collect(positions(l))
    wrap = generate_combinations(l)
    directions = Vector{Float64}[]
    # (src, trg), first index is dir, second index irrelevant
    bonds = [Tuple{Int64, Int64}[] for _ in eachindex(l)]

    for origin in eachindex(l)
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
    return bonds[temp]
end


# Directions - maybe this should be moved elsewhere
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

    return sort!(directions, by = v -> directed_norm(v, ϵ))
end


"""
    directions_with_uc(lattice::Lattice)

Returns a tuple `(trg_uc, src_uc, dir_vec)` for each (non-equivalent) direction 
in the given lattice.

See also: [`directions`](@ref)
"""
function directions_with_uc(lattice::Lattice, ϵ = 1e-6)
    _positions = collect(positions(lattice))
    wrap = generate_combinations(lattice_vectors(lattice))
    directions = Tuple{Int, Int, Vector{Float64}}[]
    B = length(lattice.unitcell.sites)

    for origin in 1:length(lattice)
        src_uc = mod1(origin, B)
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
                trg_uc = mod1(trg, B)
                push!(directions, (src_uc, trg_uc, d))
            end
        end
    end

    return sort!(directions, by = v -> directed_norm(v, ϵ))
end
