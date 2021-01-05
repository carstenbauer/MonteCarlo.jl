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
        @avx for i in eachindex(dirs)
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
    @avx for i in eachindex(out)
        for j in eachindex(weights), k in eachindex(weights)
            out[i] += weights[j] * weights[k] * data[i,j,k]
        end
    end
    out
end
function apply_symmetry(data::AbstractArray{T, 1}, weights=(1)) where {T}
    out = zero(T)
    @avx for i in eachindex(weights)
        out += weights[i] * data[i]
    end
    out
end
function apply_symmetry(data::AbstractArray{T, 2}, weights=(1)) where {T}
    out = zeros(T, size(data, 1))
    @avx for i in eachindex(out)
        for j in eachindex(weights)
            out[i] += weights[j] * data[i,j]
        end
    end
    out
end



# cc2superfluid density
superfluid_density(mc, key::Symbol, L = lattice(mc).L) = superfluid_density(mc, mc[key], L)
function superfluid_density(mc, m::DQMCMeasurement, L=lattice(mc).L)
    dirs = directions(mc)
    qx, qy = reciprocal_vectors(lattice(mc), L)
    superfluid_density(mean(m), dirs, qx, qy)
end
function superfluid_density(data::Array{T, 2}, dirs, qx, qy, skip_zero_distance=true) where {T}
    output = ComplexF64(0)
    for i in axes(data, 1)
        for j in 1+skip_zero_distance:size(data, 2)
            output += (cis(dot(qy, dirs[j])) - cis(dot(qx, dirs[j]))) * data[i, j]
        end
    end
    output
end