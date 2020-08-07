# Fully inplace, with @avx
# NOTE Results from QR match qr(A), but not qr(A, Val(true))
# However still more stable and slightly faster and U*D*T = input holds


@inline function reflector!(x::AbstractVector)
    n = length(x)
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        @avx for i = 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ξ1/normu)
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

@inline function reflectorApply!(x::AbstractVector, τ::Number, A::StridedMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        @avx for i = 2:m
            # TODO should be x[i]'
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

function udt_AVX!(U::AbstractArray{C, 2}, D::AbstractArray{C, 1}, input::AbstractArray{C, 2}) where {C<:Number}
    # Assumptions:
    # - all matrices same size
    # - input can be fucked up (input becomes T)

    # @bm "QR" begin
        # QR (use D as tau, input as A)
        n, _ = size(input)
        @inbounds D[n] = zero(C)
        @inbounds for k = 1:(n - 1 + !(C<:Real))
            x = LinearAlgebra.view(input, k:n, k)
            τk = reflector!(x)
            D[k] = τk
            reflectorApply!(x, τk, view(input, k:n, k + 1:n))
        end
    # end

    # @bm "diagonalize" begin
        inplace_identity!(U)
    # end
    # @bm "Q/U" begin
        # Compute Q/U (A.factors -> input, B -> U, A.τ -> D)
        # TODO: try to optimize away the inplace_identity
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
    # end

    # @bm "R" begin
        # Generate R
        @inbounds for j in 1:n-1
            @avx for i in max(1, j + 1):n
                input[i,j] = zero(input[i,j])
            end
        end
    # end

    # @bm "D" begin
        # Generate output D
        @inbounds for i in 1:n
            D[i] = abs(real(input[i, i]))
        end
    # end

    # @bm "T" begin
        @avx for i in 1:n
            d = 1.0 / D[i]
            for j in 1:n
                input[i, j] = d * input[i, j]
            end
        end
    # end

    nothing
end

function inplace_identity!(A::AbstractArray{T, 2}) where T
    T0 = zero(T)
    T1 = one(T)
    @avx for i in 1:size(A, 1), j in 1:size(A, 2)
        A[i, j] = T0
    end
    @avx for i in 1:size(A, 1)
        A[i, i] = T1
    end
    A
end

"""
Use dark magic to calculate G from two UDT decomposition
Ul, Dl, Tl,
Ur, Dr, Tr
"""

function calculate_greens_AVX!(
        Ul, Dl, Tl, Ur, Dr, Tr, G,
        D = Vector{Int64}(undef, length(Dl))
    )
    @bm "B1" begin
        # Requires: Ul, Dl, Tl, Ur, Dr, Tr
        vmul!(G, Tl, adjoint(Tr))
        rvmul!(G, Diagonal(Dr))
        lvmul!(Diagonal(Dl), G)
        udt_AVX!(Tr, Dr, G)
    end

    @bm "B2" begin
        # Requires: G, Ul, Ur, G, Tr, Dr
        vmul!(Tl, Ul, Tr)
        vmul!(Ul, G, adjoint(Ur))
        copyto!(Tr, Ul)
        LinearAlgebra.inv!(RecursiveFactorization.lu!(Tr, D))
        vmul!(G, adjoint(Tl), Tr)
    end

    @bm "B3" begin
        # Requires: G, Dr, Ul, Tl
        @avx for i in 1:length(Dr)
            G[i, i] = G[i, i] + Dr[i]
        end
    end

    @bm "B4" begin
        # Requires: G, Ul, Tl
        udt_AVX!(Ur, Dr, G)
        vmul!(Tr, G, Ul)
        vmul!(Ul, Tl, Ur)
    end

    @bm "B5" begin
        # Requires: Dr, Tr, Ul
        copyto!(Ur, Tr)
        LinearAlgebra.inv!(RecursiveFactorization.lu!(Ur, D))
        # Maybe merhe with multiplication?
        @avx for i in eachindex(Dr)
            Dl[i] = 1.0 / Dr[i]
        end
    end

    @bm "B6" begin
        # Requires: Dl, Ur, Ul
        rvmul!(Ur, Diagonal(Dl))
        vmul!(G, Ur, adjoint(Ul))
    end
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
