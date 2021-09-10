# TODO
# - find need allocations in attractive case
# - decimate allocations here
# - I guess maybe it's additional Diagonal's etc?



################################################################################
### Type and general utility
################################################################################


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

function Base.Matrix(B::BlockDiagonal{T, N}) where {T, N}
    n = size(B.blocks[1], 1)
    M = N * n
    output = zeros(T, M, M)
    for i in 1:N
        @views copyto!(output[(i-1)*n+1 : i*n, (i-1)*n+1 : i*n], B.blocks[i])
    end
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

function Base.setindex!(B::BlockDiagonal{T, N}, val, i, j) where {T, N}
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

function Base.copyto!(B1::BlockDiagonal{T, N}, B2::BlockDiagonal{T, N}) where {T, N}
    @inbounds for i in 1:N
        copyto!(B1.blocks[i], B2.blocks[i])
    end
    return B1
end
function Base.copyto!(B::BlockDiagonal{T, N}, ::UniformScaling) where {T, N}
    @inbounds for i in 1:N
        copyto!(B.blocks[i], I)
    end
    return B
end
function Base.copyto!(B::BlockDiagonal{T, N}, D::Diagonal{T}) where {T, N}
    @inbounds n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        b = B.blocks[i]
        offset = (i-1)*n
        @turbo for j in 1:n, k in 1:n
            b[j, k] = ifelse(j == k, D.diag[offset+j], zero(T))
        end
    end
    return B
end


# for Base.show
function Base.size(B::BlockDiagonal{T, N}) where {T, N}
    @inbounds n = size(B.blocks[1], 1)
    (n*N, n*N)
end

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

function Base.exp(B::BlockDiagonal)
    BlockDiagonal(map(block -> exp(block), B.blocks)...)
end

function Base.log(B::BlockDiagonal)
    BlockDiagonal(map(block -> log(block), B.blocks)...)
end

function LinearAlgebra.det(B::BlockDiagonal{T}) where {T}
    output = T(1)
    for b in B.blocks
        output *= det(b)
    end
    output
end


# I thought this would be needed for greens(k, l), but it's not?
function LinearAlgebra.transpose!(A::BlockDiagonal{T, N}, B::BlockDiagonal{T, N}) where {T, N}
    @inbounds for i in 1:N
        transpose!(A.blocks[i], B.blocks[i])
    end
    A
end
# This however is used for greens(k, l)
function LinearAlgebra.rmul!(B::BlockDiagonal{T, N}, f::Number) where {T, N}
    @inbounds for i in 1:N
        rmul!(B.blocks[i], f)
    end
    B
end


################################################################################
### AVX-powered matrix multiplications
################################################################################



function vmul!(C::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, B::BlockDiagonal{T, N}) where {T, N}
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A.blocks[i], B.blocks[i])
    end
end
function vmul!(C::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, B::Diagonal{T}) where {T<:Real, N}
    # Assuming correct size
    @inbounds n = size(A.blocks[1], 1)
    @inbounds for i in 1:N
        c = C.blocks[i]
        a = A.blocks[i]
        @turbo for k in 1:n, l in 1:n
            c[k,l] = a[k,l] * B.diag[(i-1)*n + l]
        end
    end
end
function vmul!(C::BlockDiagonal{T, N}, A::Diagonal{T}, B::BlockDiagonal{T, N}) where {T<:Real, N}
    # Assuming correct size
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        c = C.blocks[i]
        b = B.blocks[i]
        @turbo for k in 1:n, l in 1:n
            c[k,l] = A.diag[(i-1)*n + k] * b[k,l]
        end
    end
end
function vmul!(C::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, X::Adjoint{T}) where {T<:Real, N}
    B = X.parent
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        c = C.blocks[i]
        @turbo for k in 1:n, l in 1:n
            Ckl = zero(eltype(c))
            for m in 1:n
                Ckl += a[k,m] * b[l, m]
            end
            c[k,l] = Ckl
        end
    end
end
function vmul!(C::BlockDiagonal{T, N}, X::Adjoint{T}, B::BlockDiagonal{T, N}) where {T<:Real, N}
    A = X.parent
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        c = C.blocks[i]
        @turbo for k in 1:n, l in 1:n
            Ckl = zero(eltype(c))
            for m in 1:n
                Ckl += a[m,k] * b[m,l]
            end
            c[k,l] = Ckl
        end
    end
end
function vmul!(C::BD, X1::Adjoint{T, BD}, X2::Adjoint{T, BD}) where {T <: Real, N, BD <: BlockDiagonal{T, N}}
    A = X1.parent
    B = X2.parent
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        c = C.blocks[i]
        @turbo for k in 1:n, l in 1:n
            Ckl = zero(eltype(c))
            for m in 1:n
                Ckl += a[m,k] * b[l, m]
            end
            c[k,l] = Ckl
        end
    end
end
function rvmul!(A::BlockDiagonal{T, N}, B::Diagonal) where {T, N}
    # Assuming correct size
    @inbounds n = size(A.blocks[1], 1)
    @inbounds for i in 1:N
        @views rvmul!(A.blocks[i], Diagonal(B.diag[(i-1)*n+1 : i*n]))
    end
end
function lvmul!(A::Diagonal, B::BlockDiagonal{T, N}) where {T, N}
    # Assuming correct size
    @inbounds n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        @views lvmul!(Diagonal(A.diag[(i-1)*n+1 : i*n]), B.blocks[i])
    end
end

# used in greens(k, l)
function rvadd!(A::BlockDiagonal{T, N}, B::BlockDiagonal{T, N}) where {T <: Real, N}
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        @turbo for j in axes(a, 1), k in axes(a, 2)
            a[j, k] = a[j, k] + b[j, k]
        end
    end
end
# used in equal time greens
function rvadd!(B::BlockDiagonal{T, N}, D::Diagonal{T}) where {T<:Real, N}
    # Assuming correct size
    @inbounds n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        a = B.blocks[i]
        offset = (i-1) * n
        @turbo for j in 1:n
            a[j, j] = a[j, j] + D.diag[j+offset]
        end
    end
end
# used in CombinedGreensIterator
function vsub!(O::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, ::UniformScaling) where {T<:Real, N}
    @inbounds n = size(O.blocks[1], 1)
    T1 = one(T)
    @inbounds for i in 1:N
        a = A.blocks[i]
        o = O.blocks[i]
        @turbo for j in 1:n, k in 1:n
            o[j, k] = a[j, k]
        end
        @turbo for j in 1:n
            o[j, j] -= T1
        end
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



function vmul!(C::BlockDiagonal{T, N}, A::Diagonal, B::BlockDiagonal{T, N}) where {T, N}
    # Assuming correct size
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        @views vmul!(C.blocks[i], Diagonal(A.diag[(i-1)*n+1 : i*n]), B.blocks[i])
    end
end
function vmul!(C::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, B::Diagonal) where {T, N}
    # Assuming correct size
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        @views vmul!(C.blocks[i], A.blocks[i], Diagonal(B.diag[(i-1)*n+1 : i*n]))
    end
end
function vmul!(C::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, X::Adjoint) where {T, N}
    B = X.parent
    @inbounds for i in 1:N
        vmul!(C.blocks[i], A.blocks[i], adjoint(B.blocks[i]))
    end
end
function vmul!(C::BlockDiagonal{T, N}, X::Adjoint, B::BlockDiagonal{T, N}) where {T, N}
    A = X.parent
    @inbounds for i in 1:N
        vmul!(C.blocks[i], adjoint(A.blocks[i]), B.blocks[i])
    end
end
function vmul!(C::BD, X1::Adjoint{T, BD}, X2::Adjoint{T, BD}) where {T, N, BD <: BlockDiagonal{T, N}}
    A = X1.parent
    B = X2.parent
    @inbounds n = size(C.blocks[1], 1)
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        c = C.blocks[i]
        @inbounds for k in 1:n, l in 1:n
            Ckl = zero(eltype(c))
            for m in 1:n
                Ckl += conj(a[m,k]) * conj(b[l, m])
            end
            c[k,l] = Ckl
        end
    end
end
function rvadd!(B::BlockDiagonal{T, N}, D::Diagonal) where {T, N}
    # Assuming correct size
    @inbounds n = size(B.blocks[1], 1)
    @inbounds for i in 1:N
        @views rvadd!(B.blocks[i], Diagonal(D.diag[(i-1)*n+1 : i*n]))
    end
end
function rvadd!(A::BlockDiagonal{T, N}, B::BlockDiagonal{T, N}) where {T, N}
    @inbounds for i in 1:N
        a = A.blocks[i]
        b = B.blocks[i]
        rvadd!(a, b)
    end
end
function rdivp!(A::BD, T::BD, O::BD, pivot) where {ET, N, BD <: BlockDiagonal{ET, N}}
    # Assuming correct size
    @inbounds n = size(T.blocks[1], 1)
    @inbounds for i in 1:N
        @views rdivp!(A.blocks[i], T.blocks[i], O.blocks[i], pivot[(i-1)*n+1 : i*n])
    end
    A
end
function vsub!(O::BlockDiagonal{T, N}, A::BlockDiagonal{T, N}, ::UniformScaling) where {T, N}
    @inbounds n = size(O.blocks[1], 1)
    T1 = one(T)
    @inbounds for i in 1:N
        a = A.blocks[i]
        o = O.blocks[i]
        for j in 1:n, k in 1:n
            o[j, k] = a[j, k]
        end
        for j in 1:n
            o[j, j] -= T1
        end
    end
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
    @inbounds n = size(input.blocks[1], 1)
    @inbounds for i in 1:N
        @views udt_AVX_pivot!(
            U.blocks[i], D[(i-1)*n+1 : i*n], input.blocks[i],
            pivot[(i-1)*n+1 : i*n], temp[(i-1)*n+1 : i*n], apply_pivot
        )
        # pivot[(i-1)*n+1 : i*n] .+= (i-1)*n
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
    @inbounds n = size(input.blocks[1], 1)
    @inbounds for i in 1:N
        @views udt_AVX_pivot!(
            U.blocks[i], D[(i-1)*n+1 : i*n], input.blocks[i],
            pivot[(i-1)*n+1 : i*n], temp[(i-1)*n+1 : i*n], apply_pivot
        )
        # pivot[(i-1)*n+1 : i*n] .+= (i-1)*n
    end
end