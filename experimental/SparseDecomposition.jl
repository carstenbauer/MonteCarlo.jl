#=
Based on
https://mathoverflow.net/questions/258283/decomposing-a-matrix-into-a-product-of-sparse-matrices

The idea is that each step in a Gaussian elimination can be represented as a 
matrix product M' * E = M where E = I + v * (e_i ⊗ e_j), i.e. an identity 
matrix with one other nonzero element. Going through the Gaussian elimination
will eventually yield a Diagonal Matrix, such that M = D ⋅ E_N ⋯ E_1

This decomposition is similar to a checkerboard decomposition in that it 
decomposes an O(N³) matrix product into many simpler O(N) matrix products 
(in place they are essentially row + const * row). The decomposition is exact, 
lattice indepent, but creates just a little under N^2 terms making it hardly any
better than the matrix multiplication it replaces.
=#


# This combines multiple E matrices with the same index j. These matrices
# commute and can be applied simulateneously.
struct SparseGaussian{T} <: AbstractMatrix{T}
    vals::Vector{T}
    is::Vector{Int}
    j::Int
    N::Int
end

function Base.Matrix(sg::SparseGaussian{T}) where {T}
    M = Matrix{T}(I, sg.N, sg.N)
    for (i, v) in zip(sg.is, sg.vals)
        M[i, sg.j] = v
    end
    return M
end

Base.size(sg::SparseGaussian) = (sg.N, sg.N)

struct GaussianDecomposition{T} <: AbstractMatrix{T}
    parts::Vector{SparseGaussian{T}}
    diag::Diagonal{Float64}

end

Base.size(gd::GaussianDecomposition) = size(gd.diag)

function GaussianDecomposition(M::Matrix{T}, ϵ = 1e-12) where T
    groups, R = gaussian_elimination(M, ϵ)
    D = Diagonal(diag(R))
    parts = SparseGaussian{T}[]

    # group blocks with same j index together
    idx = groups[1][2]
    vals = T[]
    is = Int[]
    for (i, j, val) in groups
        if j != idx
            push!(parts, SparseGaussian(vals, is, idx, size(M, 1)))
            vals = T[]
            is = Int[]
            idx = j
        else
            push!(vals, val)
            push!(is, i)
        end
    end
    push!(parts, SparseGaussian(vals, is, idx, size(M, 1)))

    return GaussianDecomposition(parts, D)
end 


function gaussian_elimination(M::Matrix{T}, ϵ = 1e-12) where T
    # R ⋅ E_K ⋯ E_1 = M (up to ϵ truncation)
    # E matrix as (index, index, value)
    Es = Tuple{Int, Int, T}[]
    R = copy(M)

    for j in axes(R, 2)
        for i in axes(R, 1)
            i == j && continue
            R[i, j] == 0.0 && continue

            # Representative of E = I - R[i, j] / R[j, j]
            c = - R[i, j] / R[j, j]

            # R = E * R
            # R[k, l] = E[k, m] R[m, l]
            # R[k, l] = R[k, l] - R[i, j] / R[j, j] R[j, l]
            # R[k, l] = R[k, l] + c R[j, l] 
            for l in axes(R, 2)
                x = R[i, l] + c * R[j, l]
                R[i, l] = ifelse(abs(x) < 100max(ϵ, eps(R[i, l])), 0.0, x)
            end

            push!(Es, (i, j, c))
        end
    end

    return Es, R
end

import MonteCarlo: vmul!, @turbo, lvmul!

# slow
function vmul!(trg::Matrix, E::SparseGaussian, src::Matrix)
    copyto!(trg, src)
    j = E.j
    @inbounds for k in axes(src, 2)
        @simd for idx in eachindex(E.is)
            i = E.is[idx]
            val = E.vals[idx]
            # trg[i, k] = E[i, j] * src[j, k]
            trg[i, k] = muladd(val, src[j, k], trg[i, k])
        end
    end
    return trg
end

# fast?
function vmul!(trg::Matrix, src::Matrix, E::SparseGaussian)
    copyto!(trg, src)
    j = E.j
    @inbounds for (i, val) in zip(E.is, E.vals)
        @turbo for k in axes(src, 2)
            # trg[k, j] = src[k, i] * E[i, j]
            trg[k, j] += src[k, i] * val
        end
    end
    return trg
end

function vmul!(trg::Matrix{T}, gd::GaussianDecomposition{T}, src::Matrix{T}, tmp::Matrix{T}) where T
    # T = D ⋅ E_K ⋯ E_1
    # trg = T ⋅ src = D ⋅ E_K ⋯ E_1 ⋅ src

    # N parts mults, diag mult inline
    if iseven(length(gd.parts))
        tmp_trg = trg
        tmp_src = tmp
    else # odd - same tmp_trg as above after vmul!
        tmp_trg = tmp
        tmp_src = trg
    end

    vmul!(tmp_src, gd.parts[1], src)
    
    @inbounds for i in 2:length(gd.parts)
        vmul!(tmp_trg, gd.parts[i], tmp_src)
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end

    lvmul!(gd.diag, tmp_src)
    @assert tmp_src === trg

    return trg
end
