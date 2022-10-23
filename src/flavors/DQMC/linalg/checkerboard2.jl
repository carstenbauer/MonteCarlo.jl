# Second version that keeps around 
# src -> trg and trg -> src in the SparseCBMatrix

# Interpretation changes here so we adjust type names
struct SparseCBMatrix2{T} <: AbstractMatrix{T}
    vals::Vector{T}
    # M[i, j]
    is::Vector{Int}
    js::Vector{Int}
end

# printing
function Base.display(x::SparseCBMatrix2{T}) where T
    println(stdout, "SparseCBMatrix2{$T}:")
    for (i, j, v) in zip(x.is, x.js, x.vals)
        println(stdout, "    [$i, $j], [$j, $i] -> $v")
    end
    return
end

struct CheckerboardDecomposed2{T} <: AbstractMatrix{T}
    diag::Diagonal{T, Vector{T}}
    parts::Vector{SparseCBMatrix2{T}}

    # need this to identify squared vs non-squared the way Carsten implemented it
    squared::Bool
end

# printing
function Base.display(x::CheckerboardDecomposed2{T}) where T
    println(stdout, "CheckerboardDecomposed2{$T} with 1+$(length(x.parts)) parts representing:")
    display(Matrix(x))
    return
end

function Base.Matrix(x::CheckerboardDecomposed2{T}) where T
    N = length(x.diag.diag)
    output = Matrix{T}(undef, N, N)
    id = Matrix{T}(I, N, N)
    tmp = similar(output)
    vmul!(output, id, x, tmp)
    return output
end


function CheckerboardDecomposed2(M::Matrix, lattice, factor, is_squared)
    N = length(lattice)
    @assert N > 2 "Checkerboard decompositions don't work on L = 2 lattices. Choose a larger one!"
    @assert N % 2 == 0 "Checkerboard decomposition can currently only handle lattices with even system size L."

    flv = div(size(M, 1), N)
    checkerboard, groups, n_grps = build_checkerboard2(lattice)

    # This version assumes src -> trg to imply trg -> src to exist in T
    # exp(T) then becomes cosh(abs(T[src, trg])) on the diagonal
    # and sinh(abs(T[src, trg])) * (cos(f) + isin(f)) on the src/trg spaces
    # to avoid some multiplication we pull out the cosh (diag just becomes copy)

    D = Diagonal(exp.(factor * diag(M)))
    x = ones(Float64, N*flv*flv)
    parts = SparseCBMatrix2{eltype(M)}[]

    for group in groups
        # through the magic of math and sparsity, exp(temp) = I + temp
        # So we introduce a type for it
        vals = Vector{eltype(M)}(undef, flv*flv*length(group))
        is = Vector{Int}(undef, flv*flv*length(group))
        js = Vector{Int}(undef, flv*flv*length(group))
        idx = 1

        for id in group
            src, trg = checkerboard[1:2, id]

            for f1 in 0:N:(flv-1)*N, f2 in 0:N:(flv-1)*N
                # transform should now just do -0.5 * dtau * temp
                v = factor * M[src+f1, trg+f2]
                # pre = abs(v)
                # _re = real(v/pre)
                # _im = imag(v/pre)
                c = cosh(v) # or abs
                x[src + f1] *= c
                x[trg + f2] *= c
                vals[idx] = tanh(v)
                is[idx] = src + f1
                js[idx] = trg + f2
                idx += 1
            end
        end

        # hopefully this helps with cache coherence?
        _sort_by_ij!(vals, is, js)

        push!(parts, SparseCBMatrix2(vals, is, js))
    end

    # squared Checkerboard matrices come together as Tn ... T1 T1 ... eTn
    # Don't ask me why though
    # yea it's not... right... is it?
    if is_squared
        # Is this correct? maybe...?
        # P = parts[1]
        # for n in eachindex(P.vals)
        #     x[P.is[n]] *= (1 + P.vals[n] * P.vals[n])
        #     x[P.js[n]] *= (1 + P.vals[n] * P.vals[n])
        # end
        # parts[1].vals .*= 2

        D.diag .*= x
        D = D * D
    else
        D.diag .*= x
    end

    # TODO: 
    # this only works with constant diagonal part
    # otherwise we need to be careful about multiplication order of D
    # i.e. for inverted matrices the order needs to swap
    # I have also not checked if the current order is correct...
    if factor > 0.0 # inverted
        reverse!(parts)
    end
    

    return CheckerboardDecomposed2(D, parts, is_squared)
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
# TODO does this just still work?
function _square(A::SparseCBMatrix2{T}, N::Int) where T
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

    return SparseCBMatrix2(vals, is, js)
end

function vmul!(output::Matrix{T}, S::SparseCBMatrix2{T}, M::Matrix{T}) where T
    # O(N^2 + N^2)
    # I * M
    copyto!(output, M)

    # O(N^2)
    # P * M
    @inbounds @fastmath for k in axes(M, 2)
        @simd for n in eachindex(S.is)
            i = S.is[n]; j = S.js[n]
            output[i, k] = muladd(S.vals[n], M[j, k], output[i, k])
            output[j, k] = muladd(conj(S.vals[n]), M[i, k], output[j, k])
        end
    end
    return output
end


function vmul!(output::Matrix{T}, M::Matrix{T}, S::SparseCBMatrix2{T}) where T
    copyto!(output, M)

    @inbounds for n in eachindex(S.is)
        j = S.is[n]; k = S.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], S.vals[n], output[i, k])
            output[i, j] = muladd(M[i, k], conj(S.vals[n]), output[i, j])
        end
    end

    return M
end

function vmulc!(output::Matrix{T}, M::Matrix{T}, S::SparseCBMatrix2{T}) where T
    copyto!(output, M)

    @inbounds for n in eachindex(S.is)
        j = S.is[n]; k = S.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], conj(S.vals[n]), output[i, k])
            output[i, j] = muladd(M[i, k], S.vals[n], output[i, j])
        end
    end

    return M
end

function vmul!(output::Matrix{T}, M::Matrix{T}, S::Transpose{T, SparseCBMatrix2{T}}) where T
    copyto!(output, M)

    @inbounds for n in eachindex(S.parent.is)
        k = S.parent.is[n]; j = S.parent.js[n]
        @turbo for i in axes(M, 1) # fast loop :)
            output[i, k] = muladd(M[i, j], S.parent.vals[n], output[i, k])
            output[i, j] = muladd(M[i, k], conj(S.parent.vals[n]), output[i, j])
        end
    end

    return M
end




function vmul!(trg::Matrix{T}, src::Matrix{T}, cb::CheckerboardDecomposed2{T}, tmp::Matrix{T}) where T
    # M P1 ⋯ PN P1 ⋯ PN D
    if cb.squared
        # 2N parts mults, diag mult inline
        vmul!(tmp, src, cb.parts[1])
        tmp_trg = trg
        tmp_src = tmp
    else
        # N parts mults, diag mult inline
        if iseven(length(cb.parts))
            tmp_trg = trg
            tmp_src = tmp
        else # odd - same tmp_trg as above after vmul!
            tmp_trg = tmp
            tmp_src = trg
        end

        vmul!(tmp_src, src, cb.parts[1])
    end
    
    @inbounds for i in 2:length(cb.parts)
        vmul!(tmp_trg, tmp_src, cb.parts[i])
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end
    
    if cb.squared
        @inbounds for P in cb.parts
            vmul!(tmp_trg, tmp_src, P)
            x = tmp_trg
            tmp_trg = tmp_src
            tmp_src = x
        end
    end

    rvmul!(tmp_src, cb.diag)
    @assert tmp_src === trg

    return trg
end

# function vmul!(trg::Matrix{T}, cb::Adjoint{T, <: CheckerboardDecomposed2}, src::Matrix{T}, tmp::MAtrix{T}) where {T <: Real}
#     # D' P' ⋯ P' M
#     copyto!(trg, src)

#     for P in reverse(cb.parent.parts)
#         lvmul!(adjoint(P), trg)
#     end

#     # This if should be resolved at compile time I think...
#     if cb.parent.squared
#         @inbounds for i in 2:length(cb.parent.parts)
#             lvmul!(adjoint(cb.parent.parts[i]), trg)
#         end
#     end

#     lvmul!(adjoint(cb.parent.diag), trg)
    
#     return trg
# end


# What's better than A*B? That'S right, it's (B^T A^T)^T because 
# LoopVectorization is just that good

vmul!(trg, A, B, tmp) = vmul!(trg, A, B)
function vmul!(trg::Matrix{T}, cb::CheckerboardDecomposed2{T}, src::Matrix{T}, tmp::Matrix{T}) where T
    # P1 ⋯ PN P1 ⋯ PN D M = ((D M)^T P1^T ⋯ PN^T P1^T ⋯ PN^T)^T
    if cb.squared
        # 2N part products 
        vmul!(trg, cb.diag, src)
        tmp_trg = tmp
        tmp_src = trg
    elseif iseven(length(cb.diag))
        vmul!(trg, cb.diag, src)
        tmp_trg = tmp
        tmp_src = trg
    else
        vmul!(tmp, cb.diag, src)
        tmp_trg = trg
        tmp_src = tmp
    end

    # transpose
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    x = tmp_trg
    tmp_trg = tmp_src
    tmp_src = x

    for P in cb.parts
        vmul!(tmp_trg, tmp_src, transpose(P))
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end

    # This if should be resolved at compile time I think...
    if cb.squared
        @inbounds for P in cb.parts
            vmul!(tmp_trg, tmp_src, transpose(P))
            x = tmp_trg
            tmp_trg = tmp_src
            tmp_src = x
        end
    end
    
    # tranpose
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    @assert tmp_trg === trg

    return trg
end

function vmul!(trg::Matrix{T}, cb::Adjoint{T, <: CheckerboardDecomposed2}, src::Matrix{T}, tmp::Matrix{T}) where {T <: Real}
    # D' PN' ⋯ P1' ⋯ PBN' M = D' (M^T conj(PN) ⋯ conj(P1) ⋯ conj(PN) )^T
    # (P1 ⋯ PN P1 ⋯ PN D)' M = D' PN' ⋯ P1^T PN' ⋯ P1' M
    #                        = D' (M^T P1* ⋯ PN* P1* ⋯ PN*)^T

    if cb.parent.squared
        # 2N part products + diag product + 2 transpose
        tmp_trg = trg
        tmp_src = tmp
    elseif iseven(length(cb.parent.diag))
        tmp_trg = trg
        tmp_src = tmp
    else
        tmp_trg = tmp
        tmp_src = trg
    end

    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = src[j, i]
    end

    x = tmp_trg
    tmp_trg = tmp_src
    tmp_src = x

    for P in cb.parent.parts
        vmulc!(tmp_trg, tmp_src, P)
        x = tmp_trg
        tmp_trg = tmp_src
        tmp_src = x
    end

    # This if should be resolved at compile time I think...
    if cb.parent.squared
        @inbounds for P in cb.parent.parts
            vmulc!(tmp_trg, tmp_src, P)
            x = tmp_trg
            tmp_trg = tmp_src
            tmp_src = x
        end
    end
    
    @turbo for i in axes(trg, 1), j in axes(trg, 2)
        tmp_trg[i, j] = tmp_src[j, i]
    end

    vmul!(tmp_src, adjoint(cb.parent.diag), tmp_trg)

    @assert tmp_src === trg
    
    return trg
end
