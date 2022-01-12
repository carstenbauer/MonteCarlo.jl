"""
    vsubkron!(G, L, R)

Calculates `G[i, j] = G[i, j] - (∑ₘ L[i, m] R[j, m])` where `m` can be omitted,
meaning L and R can be Vectors. Used for local updates.
"""
function vsubkron!(G::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    @turbo for k in eachindex(L), l in eachindex(R)
        G[k, l] -= L[k] * R[l]
    end  
end

function vsubkron!(G::Matrix{Float64}, L::Matrix{Float64}, R::Matrix{Float64})
    @turbo for k in axes(L, 1), l in axes(R, 1), m in axes(L, 2)
        G[k, l] -= L[k, m] * R[l, m]
    end  
end

function vsubkron!(G::CMat64, L::CVec64, R::CVec64)
    @turbo for k in eachindex(L), l in eachindex(R)
        G.re[k, l] -= L.re[k] * R.re[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.re[k, l] += L.im[k] * R.im[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.im[k, l] -= L.im[k] * R.re[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.im[k, l] -= L.re[k] * R.im[l]
    end
end

function vsubkron!(G::BlockDiagonal, L::Tuple, R::Tuple)
    @inbounds for b in eachindex(G.blocks)
        vsubkron!(G.blocks[b], L[b], R[b])
    end
end



"""
    vsub!(trg, ::UniformScaling, src, i, N)

Calculates `trg[m, n] = I[m, (i:N:end)[n]] - src[m, (i:N:end)[n]]` where 
`n = eachindex(i:N:end)`. 
"""
function vsub!(trg::FVec64, ::UniformScaling, src::FMat64, i::Int, N::Int)
    @turbo for j in eachindex(trg)
        trg[j] = - src[j, i]
    end
    @inbounds trg[i] += 1.0
    nothing
end

function vsub!(trg::CVec64, ::UniformScaling, src::CMat64, i::Int, N::Int)
    @turbo for j in eachindex(trg.re)
        trg.re[j] = - src.re[j, i]
    end
    @inbounds trg.re[i] += 1.0
    @turbo for j in eachindex(trg.im)
        trg.im[j] = - src.im[j, i]
    end
    nothing
end

function vsub!(trg::FMat64, ::UniformScaling, src::FMat64, i::Int, N::Int)
    @inbounds for (k, l) in enumerate(i:N:size(src, 2))
        @turbo for j in axes(trg, 1)
            trg[j, k] = - src[j, l]
        end
        trg[l, k] += 1.0
    end
    nothing
end

function vsub!(trg::Tuple, ::UniformScaling, src::BlockDiagonal, i::Int, N::Int)
    @inbounds for b in eachindex(trg)
        vsub!(trg[b], I, src.blocks[b], i, N)
    end
    nothing
end



"""
    vmul!(trg, M, src, i, N)

Calculates `trg[m, n] = ∑ₖ M[n, k] * src[(i:N:end)[k], m]` where 
`n, k = eachindex(i:N:end)`. 
"""
function vmul!(trg::FVec64, M::Float64, src::FMat64, i::Int, N::Int)
    @turbo for j in eachindex(trg)
        trg[j] = M * src[i, j]
    end
    nothing
end

function vmuladd!(trg::FVec64, M::Float64, src::FMat64, i::Int, N::Int)
    @turbo for j in eachindex(trg)
        trg[j] += M * src[i, j]
    end
    nothing
end

function vmul!(trg::CVec64, M::ComplexF64, src::CMat64, i::Int, N::Int)
    vmul!(trg.re, real(M), src.re, i, N)
    vmuladd!(trg.re, -imag(M), src.im, i, N)
    vmul!(trg.im, real(M), src.im, i, N)
    vmuladd!(trg.im, imag(M), src.re, i, N)
    nothing
end

function vmul!(trg::FMat64, M::FMat64, src::FMat64, i::Int, N::Int)
    r = i:N:size(src, 1)
    @inbounds for i in axes(trg, 1), j in eachindex(r)
        tij = 0.0
        for (k, l) in enumerate(r)
            tij += M[j, k] * src[l, i]
        end
        trg[i, j] = tij
    end
    nothing
end

function vmul!(trg::NTuple, M::Union{FVec64, CVec64}, src::BlockDiagonal, i::Int, N::Int)
    @inbounds for b in eachindex(trg)
        vmul!(trg[b], M[b], src.blocks[b], i, N)
    end
    nothing
end