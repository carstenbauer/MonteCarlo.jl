################################################################################
### UDT Decomposition (no pivoting)
################################################################################

# fully in-place
# larger error than pivoted

@inline function reflector!(x::AbstractVector)
    n = length(x)
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        @avx for i = 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ξ1/normu) #zero(ξ1) ?
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        @avx for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end

# Needed for pivoted as well
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::StridedMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        @avx for i = 2:m
            vAj += conj(x[i]) * A[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        A[1, j] -= vAj
        @avx for i = 2:m
            A[i, j] -= x[i]*vAj
        end
    end
    return A
end

"""
    udt_AVX!(U::Matrix, D::Vector, T::Matrix)

In-place calculation of a UDT decomposition. The matrix `T` is simultaniously
the input matrix that is decomposed and an output matrix.

This assumes correctly sized square matrices as inputs.
"""
function udt_AVX!(U::AbstractArray{C, 2}, D::AbstractArray{C, 1}, input::AbstractArray{C, 2}) where {C<:Number}
    # Assumptions:
    # - all matrices same size
    # - input can be fucked up (input becomes T)

    @bm "Compute QR decomposition" begin
        n, _ = size(input)
        @inbounds D[n] = zero(C)
        @inbounds for k = 1:(n - 1 + !(C<:Real))
            x = LinearAlgebra.view(input, k:n, k)
            τk = reflector!(x)
            D[k] = τk
            reflectorApply!(x, τk, view(input, k:n, k + 1:n))
        end
    end

    @bm "Calculate Q" begin
        copyto!(U, I)
        @inbounds begin
            U[n, n] -= D[n]
            for k = n-1:-1:1
                for j = k:n
                    vBj = U[k,j]
                    @avx for i = k+1:n
                        vBj += conj(input[i,k]) * U[i,j]
                    end
                    vBj = D[k]*vBj
                    U[k,j] -= vBj
                    @avx for i = k+1:n
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        # U done
    end

    @bm "Calculate R" begin
        @inbounds for j in 1:n-1
            @avx for i in max(1, j + 1):n
                input[i,j] = zero(input[i,j])
            end
        end
    end

    @bm "Calculate D" begin
        @inbounds for i in 1:n
            D[i] = abs(real(input[i, i]))
        end
    end

    @bm "Calculate T" begin
        @avx for i in 1:n
            d = 1.0 / D[i]
            for j in 1:n
                input[i, j] = d * input[i, j]
            end
        end
    end

    nothing
end


################################################################################
### UDT Decomposition (with pivoting)
################################################################################


@inline function reflector!(x::Matrix, normu, j=1, n=size(x, 1))
    @inbounds begin
        ξ1 = x[j, j]
        if iszero(normu)
            return zero(ξ1) #zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[j, j] = -ν
        @avx for i = j+1:n
            x[i, j] /= ξ1
        end
    end
    ξ1/ν
end


@bm function indmaxcolumn(A::Matrix{T}, j=1, n=size(A, 1)) where T
    max = zero(T)
    @avx for k in j:n
        max += conj(A[k, j]) * A[k, j]
    end
    ii = j
    @inbounds for i in j+1:n
        mi = zero(T)
        @avx for k in j:n
            mi += conj(A[k, i]) * A[k, i]
        end
        if abs(mi) > max
            max = mi
            ii = i
        end
    end
    return ii, max
end


# (Much?) higher accuracy, but a bit slower
"""
    udt_AVX_pivot!(U::Matrix, D::Vector, T::Matrix[, pivot::Vector, temp::Vector])

In-place calculation of a UDT decomposition. The matrix `T` is simultaniously
the input matrix that is decomposed and an output matrix.

This assumes correctly sized square matrices as inputs.
"""
function udt_AVX_pivot!(
        U::AbstractArray{C, 2}, 
        D::AbstractArray{C, 1}, 
        input::AbstractArray{C, 2},
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{C, 1} = Vector{C}(undef, length(D))
    ) where {C<:Number}
    # Assumptions:
    # - all matrices same size
    # - input can be fucked up (input becomes T)

    @bm "reset pivot" begin
        n = size(input, 1)
        @inbounds for i in 1:n
            pivot[i] = i
        end
    end

    @bm "QR decomposition" begin
        @inbounds for j = 1:n
            # Find column with maximum norm in trailing submatrix
            # @bm "get jm" begin
                jm, maxval = indmaxcolumn(input, j, n)
            # end

            # @bm "pivot" begin
                if jm != j
                    # Flip elements in pivoting vector
                    tmpp = pivot[jm]
                    pivot[jm] = pivot[j]
                    pivot[j] = tmpp

                    # Update matrix with
                    @avx for i = 1:n
                        tmp = input[i,jm]
                        input[i,jm] = input[i,j]
                        input[i,j] = tmp
                    end
                end
            # end

            # Compute reflector of columns j
            # @bm "Reflector" begin
                τj = reflector!(input, maxval, j, n)
                D[j] = τj
            # end

            # Update trailing submatrix with reflector
            # @bm "apply" begin
                # TODO optimize?
                x = LinearAlgebra.view(input, j:n, j)
                MonteCarlo.reflectorApply!(x, τj, LinearAlgebra.view(input, j:n, j+1:n))
            # end
        end
    end

    @bm "Calculate Q" begin
        copyto!(U, I)
        @inbounds begin
            U[n, n] -= D[n]
            for k = n-1:-1:1
                for j = k:n
                    vBj = U[k,j]
                    @avx for i = k+1:n
                        vBj += conj(input[i,k]) * U[i,j]
                    end
                    vBj = D[k]*vBj
                    U[k,j] -= vBj
                    @avx for i = k+1:n
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        # U done
    end

    @bm "Calculate D" begin
        @inbounds for i in 1:n
            D[i] = abs(real(input[i, i]))
        end
    end

    @bm "pivoted zeroed T w/ inv(D)" begin
        @inbounds for i in 1:n
            d = 1.0 / D[i]
            @inbounds for j in 1:i-1
                temp[pivot[j]] = zero(C)
            end
            @avx for j in i:n
                temp[pivot[j]] = d * input[i, j]
            end
            @avx for j in 1:n
                input[i, j] = temp[j]
            end
        end
    end

    nothing
end


"""
    calculate_greens_AVX!(Ul, Dl, Tl, Ur, Dr, Tr, G[, pivot])

Calculates the Greens function matrix `G` from two UDT decompositions
`Ul, Dl, Tl` and `Ur, Dr, Tr`. Additionally a `pivot` vector can be given. Note
that all inputs will be overwritten.

The UDT should follow from a set of slice_matrix multiplications, such that
Ur, Dr, Tr = B(slice)' ... B(M)'
Ul, Dl, Tl = B(slice-1) ... B(1)
"""
function calculate_greens_AVX!(
        Ul, Dl, Tl, Ur, Dr, Tr, G,
        pivot = Vector{Int64}(undef, length(Dl)),
        temp = Vector{eltype(G)}(undef, length(Dl))
    )
    # NOTE `inv!` still allocates (Thanks BLAS)

    # @bm "B1" begin
        # Requires: Ul, Dl, Tl, Ur, Dr, Tr
        vmul!(G, Tl, adjoint(Tr))
        rvmul!(G, Diagonal(Dr))
        lvmul!(Diagonal(Dl), G)
        # udt_AVX!(Tr, Dr, G)
        udt_AVX_pivot!(Tr, Dr, G, pivot, temp) # Dl available
    # end

    # @bm "B2" begin
        # Requires: G, Ul, Ur, G, Tr, Dr
        vmul!(Tl, Ul, Tr)
        vmul!(Ul, G, adjoint(Ur))
        copyto!(Tr, Ul)
        LinearAlgebra.inv!(RecursiveFactorization.lu!(Tr, pivot))
        vmul!(G, adjoint(Tl), Tr)
    # end

    # @bm "B3" begin
        # Requires: G, Dr, Ul, Tl
        @avx for i in 1:length(Dr)
            G[i, i] = G[i, i] + Dr[i]
        end
    # end

    # @bm "B4" begin
        # Requires: G, Ul, Tl
        # udt_AVX!(Ur, Dr, G)
        udt_AVX_pivot!(Ur, Dr, G, pivot, temp) # Dl available
        vmul!(Tr, G, Ul)
        vmul!(Ul, Tl, Ur)
    # end

    # @bm "B5" begin
        # Requires: Dr, Tr, Ul
        copyto!(Ur, Tr)
        LinearAlgebra.inv!(RecursiveFactorization.lu!(Ur, pivot))
        # Maybe merge with multiplication?
        @avx for i in eachindex(Dr)
            Dl[i] = 1.0 / Dr[i]
        end
    # end

    # @bm "B6" begin
        # Requires: Dl, Ur, Ul
        rvmul!(Ur, Diagonal(Dl))
        vmul!(G, Ur, adjoint(Ul))
    # end
end


function vmul!(C, A, B::Diagonal)
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A[m,n] * B[n,n]
    end
end
function vmul!(C, A, X::Adjoint)
    B = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * conj(B[n, k])
        end
        C[m,n] = Cmn
    end
end
function vmul!(C, X::Adjoint, B)
    A = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += conj(A[k,m]) * B[k,n]
        end
        C[m,n] = Cmn
    end
end
function rvmul!(A, B::Diagonal)
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        A[m,n] = A[m,n] * B[n,n]
    end
end
function lvmul!(A::Diagonal, B)
    @avx for m in 1:size(B, 1), n in 1:size(B, 2)
        B[m,n] = A[m, m] * B[m, n]
    end
end
