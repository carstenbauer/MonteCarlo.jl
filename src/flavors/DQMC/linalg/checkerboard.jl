# Representing M = I + P where P[i, j] != 0 => P[i, !j] = P[j, :] = 0
# These could actually be StaticArrays I think. Or tuples I guess
struct SparseCBMatrix{T} <: AbstractMatrix{T}
    vals::Vector{T}
    # M[i, j]
    is::Vector{Int}
    js::Vector{Int}
end

struct CheckerboardDecomposed{T} <: AbstractMatrix{T}
    diag::Diagonal{T, Vector{T}}
    parts::Vector{SparseCBMatrix{T}}

    # need this to identify squared vs non-squared the way Carsten implemented it
    squared::Bool
end


################################################################################
### Constructor
################################################################################


function CheckerboardDecomposed(B::BlockDiagonal, lattice, transform, is_squared)
    return BlockDiagonal(map(
        b -> CheckerboardDecomposed(b, lattice, transform, is_squared), B.blocks
    ))
end
# function CheckerboardDecomposed(C::CMat64, lattice, transform, is_squared)
#     return CMat64( # TODO
#         CheckerboardDecomposed(C.re, lattice, transform, is_squared),
#         CheckerboardDecomposed(C.im, lattice, transform, is_squared)
#     )
# end
function CheckerboardDecomposed(M::Matrix, lattice, transform, is_squared)

    N = length(lattice)
    flv = div(size(M, 1), N)
    checkerboard, groups, n_grps = build_checkerboard(lattice)

    D = Diagonal(exp.(transform.(diag(M))))
    parts = SparseCBMatrix{eltype(M)}[]

    for group in groups
        # through the magic of math and sparsity, exp(temp) = I + temp
        # So we introduce a type for it
        vals = Vector{eltype(M)}(undef, flv*flv*length(group))
        is = Vector{Int}(undef, flv*flv*length(group))
        js = Vector{Int}(undef, flv*flv*length(group))
        idx = 1

        for id in group
            @views src, trg = checkerboard[1:2, id]

            for f1 in 0:N:(flv-1)*N, f2 in 0:N:(flv-1)*N
                # transform should now just do -0.5 * dtau * temp
                vals[idx] = transform(M[trg+f1, src+f2]) 
                is[idx] = src
                js[idx] = trg
                idx += 1
            end
        end

        # hopefully this helps with cache coherence?
        _sort_by_ij!(vals, is, js)

        push!(parts, SparseCBMatrix(vals, is, js))
    end

    # squared Checkerboard matrices come together as Tn ... T1 T1 ... eTn
    # Don't ask me why though
    if is_squared
        parts[1] = parts[1] * parts[1]
        D = D * D
    end

    return CheckerboardDecomposed(D, parts, is_squared)
end

function _sort_by_ij!(vals, is, js)
    perm = sortperm(js)
    permute!(vals, perm)
    permute!(is, perm)
    permute!(js, perm)

    perm = sortperm(is)
    permute!(vals, perm)
    permute!(is, perm)
    permute!(js, perm)
    return
end

rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)

# Note
# getindex and Matrix() are not well defined... Depending on the transform we 
# used we may need to add or multiple the internal matrices

# not for performant code
function Base.:(*)(A::SparseCBMatrix{T}, B::SparseCBMatrix{T}) where T
    # (I * P_1) (I + P_2) = I + P_1 + P_2 + P_1 * P_2

    # I + P_1
    vals = copy(A.vals)
    is = copy(A.is)
    js = copy(A.js)

    for k in eachindex(B.vals)
        # + P_2
        idx = findfirst(isequal((B.is[k], B.js[k])), collect(zip(is, js)))
        if idx === nothing
            push!(vals, B.vals[k])
            push!(is, B.is[k])
            push!(js, B.js[k])
        else
            vals[idx] += B.vals[k]
        end

        # + P_1 * P_2
        for m in eachindex(A.vals)
            # A[i, j] B[j, k]
            if A.js[m] == B.is[k]
                i = A.is[m]
                j = B.js[k]
                idx = findfirst(isequal((i, j)), collect(zip(is, js)))
                if idx === nothing
                    push!(vals, A.vals[m] * B.vals[k])
                    push!(is, i)
                    push!(js, j)
                else
                    vals[idx] += A.vals[m] * B.vals[k]
                end
            end
        end
    end

    _sort_by_ij!(vals, is, js)

    return SparseCBMatrix(vals, is, js)
end

function vmul!(trg::Matrix{T}, S::SparseCBMatrix{T}, D::Diagonal{T}) where T
    # could also try dense ij as in j = ij[i] with zeros in vals?
    # O(N^2 + N/2 + N)
    for i in eachindex(trg)
        trg[i] = zero(T)
    end
    # P * D
    for k in eachindex(S.is)
        i = S.is[k]; j = S.js[k]
        trg[i, j] = S.vals[k] * D.diag[j]
    end
    # I * D
    for i in axes(trg, 1)
        trg[i, i] = D.diag[i]
    end
    return trg
end

function vmul!(trg::Matrix{T}, S::SparseCBMatrix{T}, M::Matrix{T}) where T
    # O(N^2 + N^2/2)
    # I * M
    copyto!(trg, M)
    # P * M
    lvmul!(S, trg)
    return trg
end

function lvmul!(S::SparseCBMatrix{T}, M::Matrix{T}) where T
    # O(N^2/2)
    # I * M done automatically
    # P * M
    @inbounds @fastmath for k in axes(M, 2)
        @simd for n in eachindex(S.is)
            i = S.is[n]; j = S.js[n]
            M[i, k] = muladd(S.vals[n], M[j, k], M[i, k])
        end
    end
    return M
end

function lvmul!(S::Adjoint{T, SparseCBMatrix{T}}, M::Matrix{T}) where T
    # O(N^2/2)
    # I * M done automatically
    # P * M
    @inbounds for k in axes(M, 2)
        @simd for n in eachindex(S.parent.is)
            j = S.parent.is[n]; i = S.parent.js[n] # transpose
            M[i, k] = muladd(conj(S.parent.vals[n]), M[j, k], M[i, k])
        end
    end
    return M
end

function vmul!(trg::Matrix{T}, M::Matrix{T}, S::SparseCBMatrix{T}) where T
    copyto!(trg, M)
    rvmul!(trg, S)
    return trg
end

function rvmul!(M::Matrix{T}, S::SparseCBMatrix{T}) where T
    @inbounds for n in eachindex(S.is)
        j = S.is[n]; k = S.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            M[i, k] = muladd(M[i, j], S.vals[n], M[i, k])
        end
    end
    return M
end



# Note
# These are rather specialized in what they calculate.
# With squared = false they calculate P1 P2 ⋯ PN * D
# With squared = true: PN ⋯ P1 ⋯ PN * D (where P1 and D is already squared beforehand)
function vmul!(trg::Matrix{T}, cb::CheckerboardDecomposed{T}, src::Matrix{T}) where T
    # P ⋯ P D M
    vmul!(trg, cb.diag, src)

    for P in reverse(cb.parts)
        lvmul!(P, trg)
    end

    # This if should be resolved at compile time I think...
    if cb.squared
        @inbounds for i in 2:length(cb.parts)
            lvmul!(cb.parts[i], trg)
        end
    end
    
    return trg
end

function vmul!(trg::Matrix{T}, src::Matrix{T}, cb::CheckerboardDecomposed{T}) where T
    copyto!(trg, src)

    for P in reverse(cb.parts)
        rvmul!(trg, P)
    end
    
    if cb.squared
        @inbounds for i in 2:length(cb.parts)
            rvmul!(trg, cb.parts[i])
        end
    end

    rvmul!(trg, cb.diag)

    return trg
end

function vmul!(trg::Matrix{T}, cb::Adjoint{T, <: CheckerboardDecomposed}, src::Matrix{T}) where {T <: Real}
    # D' P' ⋯ P' M
    vmul!(trg, adjoint(cb.parent.diag), src)

    for P in reverse(cb.parent.parts)
        lvmul!(adjoint(P), trg)
    end

    # This if should be resolved at compile time I think...
    if cb.parent.squared
        @inbounds for i in 2:length(cb.parent.parts)
            lvmul!(adjoint(cb.parent.parts[i]), trg)
        end
    end
    
    return trg
end