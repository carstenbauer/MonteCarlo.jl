# Note: (general)
# This is not optimized at all. This first version is straight up just what 
# Carsten had originally, moved into a type


struct CheckerboardDecomposed{T} <: AbstractMatrix{T}
    diag::Diagonal{T, Vector{T}}
    parts::Vector{SparseMatrixCSC{T, Int64}}

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
    D = Diagonal(transform.(diag(M)))

    N = length(lattice)
    flv = div(size(M, 1), N)
    checkerboard, groups, n_grps = build_checkerboard(lattice)

    T0 = zero(eltype(M))
    temp = Matrix{eltype(M)}(undef, size(M))
    parts = SparseMatrixCSC{eltype(M), Int64}[]

    for group in groups
        # write hoppings from group to temp
        temp .= T0
        for id in group
            @views src, trg = checkerboard[1:2, id]

            for f1 in 0:N:(flv-1)*N, f2 in 0:N:(flv-1)*N
                temp[trg+f1, src+f2] = M[trg+f1, src+f2]
            end
        end

        # apply transformations, e.g. exp(-0.5 * dtau * temp), and add to parts
        T_group = transform(temp)
        rem_eff_zeros!(T_group)
        push!(parts, T_group)
    end

    # squared Checkerboard matrices come together as Tn ... T1 T1 ... eTn
    # Don't ask me why though
    if is_squared
        parts[1] = parts[1] * parts[1]
        D = D * D
    end

    return CheckerboardDecomposed(D, parts, is_squared)
end

rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e)<1e-15 ? zero(e) : e,X,X)

# Note
# getindex and Matrix() are not well defined... Depending on the transform we 
# used we may need to add or multiple the internal matrices

# Note
# These are rather specialized in what they calculate.
# With squared = false they calculate P1 P2 ⋯ PN * D
# With squared = true: PN ⋯ P1 ⋯ PN * D (where P1 and D is already squared beforehand)
function vmul!(trg::Matrix{T}, cb::CheckerboardDecomposed{T}, src::Matrix{T}) where T
    # vmulskip! always swaps outputs, so we have a total double swap here
    mul!(trg, cb.diag, src)

    tmp_src = trg
    tmp_trg = src

    # N operations
    for P in reverse(cb.parts)
        mul!(tmp_trg, P, tmp_src)
        x = tmp_src 
        tmp_src = tmp_trg
        tmp_trg = x
    end

    # This if should be resolved at compile time I think...
    if cb.squared || isodd(length(cb.parts))
        # N-1 multiplications -> 2N-1 total -> tmp_trg == trg
        for i in 2:length(cb.parts)
            P = cb.parts[i]
            mul!(tmp_trg, P, tmp_src)
            x = tmp_src 
            tmp_src = tmp_trg
            tmp_trg = x
        end

        # total even number of multiplications, fix target
        copyto!(trg, src)
    end
    
    return trg
end

function vmul!(trg::Matrix{T}, src::Matrix{T}, cb::CheckerboardDecomposed{T}) where T
    tmp_src = src
    tmp_trg = trg
    
    for P in reverse(cb.parts)
        mul!(tmp_trg, tmp_src, P)
        x = tmp_src 
        tmp_src = tmp_trg
        tmp_trg = x
    end
    
    if cb.squared || isodd(length(cb.parts))
        for i in 2:length(cb.parts)
            P = cb.parts[i]
            mul!(tmp_trg, tmp_src, P)
            x = tmp_src 
            tmp_src = tmp_trg
            tmp_trg = x
        end

        # one more multiplication to do so we're currently on trg but need to be
        # on src to end at trg
        copyto!(src, trg)
    end

    mul!(trg, src, cb.diag)

    return trg
end