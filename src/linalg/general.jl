################################################################################
### AVX-powered matrix multiplications for real matrices
################################################################################



function vmul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end
function vmul!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A[m,n] * B[n,n]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}) where {T <: Real}
    @avx for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = A[m,m] * B[m,n]
    end
end
function vmul!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}) where {T <: Real}
    B = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * conj(B[n, k])
        end
        C[m,n] = Cmn
    end
end
function vmul!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}) where {T <: Real}
    A = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += conj(A[k,m]) * B[k,n]
        end
        C[m,n] = Cmn
    end
end
function rvmul!(A::Matrix{T}, B::Diagonal{T}) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        A[m,n] = A[m,n] * B[n,n]
    end
end
function lvmul!(A::Diagonal{T}, B::Matrix{T}) where {T <: Real}
    @avx for m in 1:size(B, 1), n in 1:size(B, 2)
        B[m,n] = A.diag[m] * B[m, n]
    end
end
function rvadd!(A::Matrix{T}, D::Diagonal{T}) where {T <: Real}
    @avx for i in axes(A, 1)
        A[i, i] = A[i, i] + D.diag[i]
    end
end
function rvadd!(A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @avx for i in axes(A, 1), j in axes(A, 2)
        A[i, j] = A[i, j] + B[i, j]
    end
end


"""
    rdivp!(A, T, O, pivot)

Computes `A * T^-1` where `T` is an upper triangular matrix which should be 
pivoted according to a pivoting vector `pivot`. `A`, `T` and `O` should all be
square matrices of the same size. The result will be written to `A` without 
changing `T`.

This function is written to work with (@ref)[`udt_AVX_pivot!`].
"""
function rdivp!(A, T, O, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            @avx for i in 1:N
                O[i, j] = A[i, p]
            end
        end

        # do the rdiv
        # @avx will segfault on `k in 1:0`, so pull out first loop 
        @avx for i in 1:N
            A[i, 1] = O[i, 1] / T[1, 1]
        end
        for j in 2:N
            @avx for i in 1:N
                x = O[i, j]
                for k in 1:j-1
                    x -= A[i, k] * T[k, j]
                end
                A[i, j] = x / T[j, j]
            end
        end
    end
    A
end



################################################################################
### Fallbacks for complex matrices
################################################################################



vmul!(C, A, B) = mul!(C, A, B)
rvmul!(A, B) = rmul!(A, B)
lvmul!(A, B) = lmul!(A, B)
rvadd!(A, B) = A .= A .+ B


function rdivp!(A::Matrix{<: Complex}, T::Matrix{<: Complex}, O::Matrix{<: Complex}, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            for i in 1:N
                O[i, j] = A[i, p]
            end
        end

        # do the rdiv
        # @avx will segfault on `k in 1:0`, so pull out first loop 
        for j in 1:N
            for i in 1:N
                x = O[i, j]
                @simd for k in 1:j-1
                    x -= A[i, k] * T[k, j]
                end
                A[i, j] = x / T[j, j]
            end
        end
    end
    A
end