reciprocal_distretization(x, L) = reciprocal_distretization(lattice(x), L)
function reciprocal_distretization(lattice::AbstractLattice, L)
    r1, r2 = reciprocal_vectors(lattice, L)
    reciprocal_distretization(r1, r2, L)
end
function reciprocal_distretization(v1, v2, L)
    pgram = map(((i, j) for i in 0:L-1 for j in 0:L-1)) do (i, j)
        out = [0., 0.]
        temp = [0., 0.]
        min = Inf
        for n in -1:1, m in -1:1
            temp .= v1 * (n + i/L) .+ v2 * (m + j/L)
            if norm(temp) < min
                min = norm(temp)
                out .= temp
            end
        end
        out
    end
    return pgram
end


structure_factor(args...) = fourier_transform(args...)
function fourier_transform(mc::DQMC, key::Symbol, args...)
    fourier_transform(mc, mc[key], args...)
end
function fourier_transform(mc::DQMC, m::DQMCMeasurement, L::Integer, args...)
    qs = reciprocal_distretization(mc, L)
    return qs, fourier_transform(mc, qs, m, args...)
end
function fourier_transform(mc::DQMC, qs::Vector, m::DQMCMeasurement, args...)
    dirs = directions(mc)
    values = mean(m)
    fourier_transform(qs, dirs, values, args...)
end

function fourier_transform(qs::Vector, dirs::Vector, values::Vector)
    @boundscheck length(dirs) == length(qs) == length(values)
    map(qs) do q
        sum(cis(dot(q, v)) * o for (v, o) in zip(dirs, values))
    end
end

function fourier_transform(qs::Vector, dirs::Vector, values::Matrix, weights::Vector)
    @boundscheck begin
        length(weights) ≤ size(values, 2) &&
        length(dirs) == length(qs) &&
        length(dirs) == size(values, 1)
    end
    map(qs) do q
        out = zero(ComplexF64)
        for i in eachindex(dirs)
            temp = zero(ComplexF64)
            for j in eachindex(weights)
                temp += weights[j] * values[i, j]
            end
            out += cis(dot(q, dirs[i])) * temp
        end
        out
    end
end

function fourier_transform(qs::Vector, dirs::Vector, values::Array{T, 3}, weights::Vector) where {T}
    @boundscheck begin
        length(weights) ≤ size(values, 2) &&
        length(dirs) == length(qs) &&
        length(dirs) == size(values, 1)
    end
    map(qs) do q
        out = zero(ComplexF64)
        @turbo for i in eachindex(dirs)
            temp = zero(ComplexF64)
            for j in eachindex(weights), k in eachindex(weights)
                temp += weights[j] * weights[k] * values[i, j, k]
            end
            out += cis(dot(q, dirs[i])) * temp
        end
        out
    end
end



uniform_fourier(mc::DQMC, key::Symbol) = uniform_fourier(mc[key])
uniform_fourier(m::DQMCMeasurement) = sum(mean(m))



apply_symmetry(mc::DQMC, key::Symbol, weights=(1)) = apply_symmetry(mc[key], weights)
apply_symmetry(m::DQMCMeasurement, weights=(1)) = apply_symmetry(mean(m), weights)
function apply_symmetry(data::AbstractArray{T, 3}, weights=(1)) where {T}
    out = zeros(T, size(data, 1))
    @turbo for i in eachindex(out)
        for j in eachindex(weights), k in eachindex(weights)
            out[i] += weights[j] * weights[k] * data[i,j,k]
        end
    end
    out
end
function apply_symmetry(data::AbstractArray{T, 1}, weights=(1)) where {T}
    out = zero(T)
    @turbo for i in eachindex(weights)
        out += weights[i] * data[i]
    end
    out
end
function apply_symmetry(data::AbstractArray{T, 2}, weights=(1)) where {T}
    out = zeros(T, size(data, 1))
    @turbo for i in eachindex(out)
        for j in eachindex(weights)
            out[i] += weights[j] * data[i,j]
        end
    end
    out
end


################################################################################
### Superfluid Stiffness
################################################################################



function superfluid_stiffness(
        mc::DQMC, G::DQMCMeasurement, ccs::DQMCMeasurement; 
        shift_dir = [1., 0.], calculate_error = false
    )
    # find all hopping directions (skipping on-site)
    valid_directions = hopping_directions(mc, mc.model)

    # reduce to directions that have a positive component in shift_dir
    dirs = directions(lattice(mc))
    filter!(idx -> dot(dirs[idx], shift_dir) > 0, valid_directions)

    # Reduce to valid (ccs_idx, dir_idx) pairs (we may have more or less in 
    # the measurement)
    ind_dir = Pair{Int, Int}[]
    for (i, j) in enumerate(ccs.lattice_iterator.directions)
        if j in valid_directions
            push!(ind_dir, i => j)
        end
    end
    if length(ind_dir) != length(valid_directions)
        @warn("Missing directions in Superfluid Stiffness")
    end

    Kx  = dia_K_x(mc, mean(G), last.(ind_dir))
    Λxx = para_ccc(mc, mean(ccs), ind_dir)

    if calculate_error
        @warn "Errors not implemented for sfs yet"
        return 0.25 * (-Kx - Λxx), 0.0
    else
        return 0.25 * (-Kx - Λxx)
    end
end


function dia_K_x(mc, G, idxs)
    T = Matrix(MonteCarlo.hopping_matrix(mc, mc.model))

    push!(mc.lattice_iterator_cache, MonteCarlo.Dir2SrcTrg(), lattice(mc))
    dir2srctrg = mc.lattice_iterator_cache[MonteCarlo.Dir2SrcTrg()]
    N = length(lattice(mc))
    
    Kx = ComplexF64(0)
    flv = max(nflavors(field(mc)), nflavors(model(mc))) 
    if     flv == 1; f = 2.0
    elseif flv == 2; f = 1.0
    else error("The diamagnetic contribution to the superfluid density has no implementation for $flv flavors")
    end
    
    for shift in 0 : N : flv*N - 1
        for i in idxs
            for (src, trg) in dir2srctrg[i]
                # c_j^† c_i = δ_ij - G[i, j], but δ always 0 cause no onsite
                Kx -= f * T[trg+shift, src+shift] * G[src+shift, trg+shift]
                # reverse directions cause we filter those out beforehand
                Kx -= f * T[src+shift, trg+shift] * G[trg+shift, src+shift]
            end
        end
    end
    Kx /= N
end

# uses directions with filter > 0
function para_ccc(mc, ccs, ind_dir)
    #dq = [2pi / size(lattice(mc))[2], 0.0]
    dq = [0.0, 0.0]
    dirs = directions(lattice(mc))
    Λxx = ComplexF64(0)

    for (i, dir) in enumerate(dirs)
        for (j, jdir) in ind_dir, (k, kdir) in ind_dir
            # result += ccs[i, j] * (cis(dot(dir, long)) - cis(dot(dir, trans)))
            Λxx += ccs[i, j, k] * cis(-dot(dir + 0.5(dirs[jdir] .- dirs[kdir]), dq))
        end
    end
    
    Λxx
end
