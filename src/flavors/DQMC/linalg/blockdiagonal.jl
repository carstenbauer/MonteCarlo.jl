################################################################################
### Type and general utility
################################################################################

"""
    BlockDiagonal(blocks...)
    BlockDiagonal{eltype, N_blocks, MatrixType}(init, n, m)

Creates a block diagonal matrix from a collection of blocks or an initializer, 
i.e. `undef` or `I`. In the latter case `n, m` specifies the size of the full 
block diagonal matrix.

Notes:
- `vmul!` and friends are implemented for real and complex `BlockDiagonal`
- some base methods are implemented - `Matrix`, `getidnex`, `setindex`, `copyto!`,
`size`, `copy`, `log`, `exp`, `det`, `*`
- BlockDiagonal of 2 equally sized square matrices is heavily tested and used
- BlockDiagonal of more equally sized square matrices should work
- irregularly sized and non-square matrices are not supported
- methods generally assume correct size and use @inbounds for performance.
"""
struct BlockDiagonal{T, N, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::NTuple{N, AT}

    function BlockDiagonal(blocks::NTuple{N, AT}) where {T, N, AT <: AbstractMatrix{T}}
        n, m = size(first(blocks))
        n == m || throw(ErrorException("Each block must be square."))
        all(b -> size(b) == (n, n), blocks) || throw(ErrorException(
            "All Blocks must have the same size."
        ))
        # One matrix with zeros is faster than multiple 3x3 matrices on my
        # machine (avx2). This will most likely depend on AVX version...
        # if n < 4
        #     full = zeros(T, n*N, n*N)
        #     for i in 1:N
        #         @views copyto!(full[(i-1)*n+1 : i*n, (i-1)*n+1 : i*n], blocks[i])
        #     end
        #     # new{T, 1, AT}((full,))
        #     # Is this bad? - Yes, because the rest of the code doesn't know about Matrix returns
        #     return full
        # else
            return new{T, N, AT}(blocks)
        # end
    end
end

BlockDiagonal(blocks::AT...) where {T, AT <: AbstractMatrix{T}} = BlockDiagonal(blocks)
function BlockDiagonal{T, N, AT}(::UndefInitializer, n, m) where {T, N, AT}
    n == m || throw(ErrorException("Each block must be square."))
    (n, m) = divrem(n, N)
    m == 0 || throw(ErrorException(
        "The size of the Matrix is incompatible with the blocking ($n % $N != 0)"
    ))
    BlockDiagonal([AT(undef, n, n) for _ in 1:N]...)
end
function BlockDiagonal{T, N, AT}(::UniformScaling, n, m) where {T, N, AT}
    n == m || throw(ErrorException("Each block must be square."))
    (n, m) = divrem(n, N)
    m == 0 || throw(ErrorException(
        "The size of the Matrix is incompatible with the blocking ($n % $N != 0)"
    ))
    BlockDiagonal([AT(I, n, n) for _ in 1:N]...)
end

# Shorthand for typing functions
const BD{N} = BlockDiagonal{<: Number, N}

function Base.Matrix(B::BlockDiagonal{T, N}) where {T, N}
    n = size(B.blocks[1], 1)
    output = zeros(T, N * n, N * n)
    copyto!(output, B)
    output
end

function Base.checkbounds(B::BlockDiagonal, i::Integer, j::Integer)
    N = M = 0
    @inbounds for b in B.blocks
        N += size(b, 1)
        M += size(b, 2)
    end
    (1 ≤ i ≤ N) && (1 ≤ j ≤ M)
end

function Base.getindex(B::BlockDiagonal{T, N}, i, j) where {T, N}
    @boundscheck checkbounds(B, i, j)
    @inbounds n = size(B.blocks[1], 1)

    b, k = divrem(i-1, n)
    _b, l = divrem(j-1, n)
    
    if b != _b
        return zero(T)
    else
        @inbounds return B.blocks[b+1][k+1, l+1]
    end
end

function Base.setindex!(B::BlockDiagonal, val, i, j)
    i0 = j0 = 0
    for block in B.blocks
        i1 = size(block, 1) + i0
        j1 = size(block, 2) + j0
        if (i0 < i <= i1) && (j0 < j <= j1)
            @inbounds block[i - i0, j - j0] = val
            return B
        end
        i0 = i1; j0 = j1
    end
    throw(BoundsError(B, (i, j)))
end

function Base.copyto!(B1::BD{N}, B2::BD{N}) where {N}
    @inbounds for i in 1:N
        copyto!(B1.blocks[i], B2.blocks[i])
    end
    return B1
end
function Base.copyto!(B::BlockDiagonal, ::UniformScaling)
    @inbounds for block in B.blocks
        copyto!(block, I)
    end
    return B
end
function Base.copyto!(B::BlockDiagonal, D::Diagonal)
    n = 0
    @inbounds for block in B.blocks
        s = 1 + n
        n += size(block, 1)
        @views copyto!(block, Diagonal(D.diag[s:n]))
    end
    return B
end

function Base.copyto!(output::Matrix, B::BlockDiagonal{T, N}) where {T, N}
    n = size(B.blocks[1], 1)
    for i in 1:N
        @views copyto!(output[(i-1)*n+1 : i*n, (i-1)*n+1 : i*n], B.blocks[i])
    end
    output
end

# for Base.show
Base.size(B::BlockDiagonal) = mapreduce(b -> size(b), (a, b) -> a .+ b, B.blocks)

# Needed for tests
Base.copy(B::BlockDiagonal) = deepcopy(B)

# Needed for stack building (shouldn't be performance critical)
function Base.:(*)(x::Number, B::BlockDiagonal)
    BlockDiagonal(map(block -> x * block, B.blocks)...)
end
Base.:(*)(B::BlockDiagonal, x::Number) = x * B
function Base.:(*)(B1::BlockDiagonal{T1, N}, B2::BlockDiagonal{T2, N}) where {T1, T2, N}
    BlockDiagonal(map(*, B1.blocks, B2.blocks)...)
end

Base.exp(B::BlockDiagonal) = BlockDiagonal(map(block -> exp(block), B.blocks))
function fallback_exp(B::BlockDiagonal)
    BlockDiagonal(map(block -> fallback_exp(block), B.blocks))
end

# This takes super long to compile...? 
# Dot syntax is another 6x slower
# function Base.log(B::BlockDiagonal)
#     BlockDiagonal(map(block -> log(block), B.blocks))
# end

# function LinearAlgebra.det(B::BlockDiagonal{T}) where {T}
#     output = T(1)
#     for b in B.blocks
#         output *= det(b)
#     end
#     output
# end


# I thought this would be needed for greens(k, l), but it's not?
function LinearAlgebra.transpose!(A::BlockDiagonal{T, N}, B::BlockDiagonal{T, N}) where {T, N}
    @inbounds for i in 1:N
        transpose!(A.blocks[i], B.blocks[i])
    end
    A
end
# This however is used for greens(k, l)
function LinearAlgebra.rmul!(B::BlockDiagonal, f::Number)
    @inbounds for block in B.blocks
        rmul!(block, f)
    end
    B
end


################################################################################
### AVX-powered matrix multiplications
################################################################################


function vmul!(C::BD{N}, A::BD{N}, B::BD{N}) where {N}
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A.blocks[i], B.blocks[i])
    end
    nothing
end

function vmul!(C::BD{N}, A::BD{N}, B::Diagonal) where {N}
    n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A.blocks[i], B, (i-1)*n+1 : i*n)
    end
    nothing
end
function vmul!(C::BD{N}, A::Diagonal, B::BD{N}) where {N}
    # Assuming correct size
    n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A, B.blocks[i], (i-1)*n+1 : i*n)
    end
    nothing
end
function vmul!(C::BD{N}, A::Adjoint, B::Diagonal) where {N}
    n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        vmul!(C.blocks[i], adjoint(A.parent.blocks[i]), B, (i-1)*n+1 : i*n)
    end
    nothing
end
function vmul!(C::BD{N}, A::Diagonal, B::Adjoint) where {N}
    n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A, adjoint(B.parent.blocks[i]), (i-1)*n+1 : i*n)
    end
    nothing
end

function vmul!(C::BD{N}, A::BD{N}, B::Adjoint) where {N}
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
    end
end
function vmul!(C::BD{N}, A::Adjoint, B::BD{N}) where {N}
    @inbounds for i in 1:N
        vmul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
    end
end
function vmul!(C::BD{N}, A::Adjoint, B::Adjoint) where {N}
    @inbounds for i in 1:N
        vmul!(C.blocks[i], adjoint(A.parent.blocks[i]), adjoint(B.parent.blocks[i]))
    end
end
function rvmul!(A::BD{N}, B::Diagonal) where {N}
    n = size(A.blocks[1], 1)
    @inbounds for i in 1:N
        @views rvmul!(A.blocks[i], Diagonal(B.diag[(i-1)*n+1 : i*n]))
    end
end
function lvmul!(A::Diagonal, B::BD{N}) where {N}
    n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        @views lvmul!(Diagonal(A.diag[(i-1)*n+1 : i*n]), B.blocks[i])
    end
end

# used in greens(k, l)
function rvadd!(A::BD{N}, B::BD{N}) where {N}
    @inbounds for i in 1:N
        rvadd!(A.blocks[i], B.blocks[i])
    end
end
# used in equal time greens
function rvadd!(B::BD{N}, D::Diagonal) where {N}
    # Assuming correct size
    n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        @views rvadd!(B.blocks[i], Diagonal(D.diag[(i-1)*n+1 : i*n]))
    end
end
# used in CombinedGreensIterator
function vsub!(O::BD{N}, A::BD{N}, ::UniformScaling) where {N}
    @inbounds for i in 1:N
        vsub!(O.blocks[i], A.blocks[i], I)
    end
end



function rdivp!(A::BD, T::BD, O::BD, pivot) where {ET<:Real, N, BD <: BlockDiagonal{ET, N}}
    # assume Diagonal is ±1!
    @inbounds begin
        n = size(A.blocks[1], 1)

        # Apply pivot
        for b in 1:N
            o = O.blocks[b]
            a = A.blocks[b]
            t = T.blocks[b]
            offset = (b-1)*n

            for j in 1:n
                p = pivot[j + offset]
                @turbo for i in 1:n
                    o[i, j] = a[i, p]
                end
            end

            # do the rdiv
            # @turbo will segfault on `k in 1:0`, so pull out first loop 
            invt11 = 1.0 / t[1, 1]
            @turbo for i in 1:n
                a[i, 1] = o[i, 1] * invt11
            end
            for j in 2:n
                invtjj = 1.0 / t[j, j]
                @turbo for i in 1:n
                    x = o[i, j]
                    for k in 1:j-1
                        x -= a[i, k] * t[k, j]
                    end
                    a[i, j] = x * invtjj
                end
            end
        end
    end
    A
end




################################################################################
### Fallbacks
################################################################################



function rdivp!(A::BD, T::BD, O::BD, pivot) where {ET, N, BD <: BlockDiagonal{ET, N}}
    # Assuming correct size
    n = size(A.blocks[1], 1)
    @inbounds for i in 1:N
        @views rdivp!(A.blocks[i], T.blocks[i], O.blocks[i], pivot[(i-1)*n+1 : i*n])
    end
    A
end



################################################################################
### UDT
################################################################################



function udt_AVX_pivot!(
        U::BlockDiagonal{Float64, N, AT}, 
        D::AbstractArray{Float64, 1}, 
        input::BlockDiagonal{Float64, N, AT},
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{Float64, 1} = Vector{Float64}(undef, length(D)),
        apply_pivot::Val = Val(true)
    ) where {N, AT <: AbstractMatrix{Float64}}
    # Assuming correct size
    n = 0
    @inbounds for i in 1:N
        s = 1+n
        n += size(U.blocks[i], 1)
        @views udt_AVX_pivot!(
            U.blocks[i], D[s:n], input.blocks[i],
            pivot[s:n], temp[s:n], apply_pivot
        )
    end
end
function udt_AVX_pivot!(
        U::BlockDiagonal{ComplexF64, N, AT}, 
        D::AbstractArray{Float64, 1}, 
        input::BlockDiagonal{ComplexF64, N, AT},
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{ComplexF64, 1} = Vector{ComplexF64}(undef, length(D)),
        apply_pivot::Val = Val(true)
    ) where {N, AT <: AbstractMatrix{ComplexF64}}
    # Assuming correct size
    n = 0
    @inbounds for i in 1:N
        s = 1+n
        n += size(U.blocks[i], 1)
        @views udt_AVX_pivot!(
            U.blocks[i], D[s:n], input.blocks[i],
            pivot[s:n], temp[s:n], apply_pivot
        )
    end
end