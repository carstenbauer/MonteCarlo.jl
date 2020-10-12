################################################################################
### Helpers
################################################################################



function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] += facotr * Cmn
    end
end
function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}, factor::T = T(1)) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] += factor * A[m,n] * B[n,n]
    end
end
function vmuladd!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}, factor::T = T(1)) where {T <: Real}
    B = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * conj(B[n, k])
        end
        C[m,n] += factor * Cmn
    end
end
function vmuladd!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    A = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += conj(A[k,m]) * B[k,n]
        end
        C[m,n] += factor * Cmn
    end
end



################################################################################
### Overloads for complex StructArrays
################################################################################



const CMat64 = StructArray{Complex{Float64},2,NamedTuple{(:re, :im),Tuple{Array{Float64,2},Array{Float64,2}}},Int64}
const CVec64 = StructArray{Complex{Float64},1,NamedTuple{(:re, :im),Tuple{Array{Float64,1},Array{Float64,1}}},Int64}

function vmul!(C::CMat64, A::CMat64, B::CMat64)
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    vmul!(   C.re, A.re, B.re)     # C.re = A.re * B.re
    vmuladd!(C.re, A.im, B.im, -1) # C.re = C.re - A.im * B.im
    vmul!(   C.im, A.re, B.im)     # C.im = A.re * B.im
    vmuladd!(C.im, A.im, B.re)     # C.im = C.im + A.im * B.re
end

function vmul!(C::CMat64, A::CMat64, B::Diagonal{T}) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    vmul!(C.re, A.re, B)
    vmul!(C.im, A.im, B)
end
function vmul!(C::CMat64, A::CMat64, B::Diagonal{ComplexF64, CVecF64})
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    vmul!(   C.re, A.re, B.re)
    vmuladd!(C.re, A.im, B.im, -1.0)
    vmul!(   C.im, A.re, B.im)
    vmuladd!(C.im, A.im, B.re)
end

function vmul!(C::CMat64, A::CMat64, X::Adjoint{T, CMat64}) where {T <: ComplexF64}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    B = X.parent
    vmul!(   C.re, A.re, adjoint(B.re))
    vmuladd!(C.re, A.im, adjoint(B.im), -1.0)
    vmul!(   C.im, A.re, adjoint(B.im))
    vmuladd!(C.im, A.im, adjoint(B.re))
end
function vmul!(C::CMat64, X::Adjoint{T}, B::CMat64) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    A = X.parent
    vmul!(   C.re, adjoint(A.re), B.re)
    vmuladd!(C.re, adjoint(A.im), B.im, -1.0)
    vmul!(   C.im, adjoint(A.re), B.im)
    vmuladd!(C.im, adjoint(A.im), B.re)
end

function rvmul!(A::CMat64, B::Diagonal{T}) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    rvmul!(A.re, B)
    rvmul!(A.im, B)
end
function rvmul!(A::CMat64, B::Diagonal{ComplexF64})
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    @inbounds for m in 1:size(A, 1), n in 1:size(A, 2)
        A[m,n] = A[m, n] * B[n,n]
    end
end

function lvmul!(A::Diagonal{T}, B::CMat64) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    lvmul!(A, B.re)
    lvmul!(A, B.im)
end
function lvmul!(A::Diagonal{ComplexF64}, B::CMat64)
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    @inbounds for m in 1:size(B, 1), n in 1:size(B, 2)
        B[m,n] = A[m, m] * B[m, n]
    end
end



################################################################################
### UDT
################################################################################



function udt_AVX_pivot!(
        U::CMat64, 
        D::AbstractArray{Float64, 1}, 
        input::CMat64,
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{C, 1} = Vector{C}(undef, length(D)),
        apply_pivot::Val = Val(true)
    ) where {C <: Complex}
    # Assumptions:
    # - all matrices same size
    # - input can be changed (input becomes T)
    @warn "Complex StructArrays are untested not really optimized." maxlog=10

    n = size(input, 1)
    @inbounds for i in 1:n
        pivot[i] = i
    end

    # TODO optimize
    @inbounds for j = 1:n
        # Find column with maximum norm in trailing submatrix
        jm, maxval = indmaxcolumn(input, j, n)

        if jm != j
            # Flip elements in pivoting vector
            tmpp = pivot[jm]
            pivot[jm] = pivot[j]
            pivot[j] = tmpp

            # Update matrix with
            @avx for i = 1:n
                tmp = input.re[i,jm]
                input.re[i,jm] = input.re[i,j]
                input.re[i,j] = tmp
            end
            @avx for i = 1:n
                tmp = input.im[i,jm]
                input.im[i,jm] = input.im[i,j]
                input.im[i,j] = tmp
            end
        end
        
        # Compute reflector of columns j
        τj = reflector!(input, maxval, j, n)
        temp[j] = τj

        # Update trailing submatrix with reflector
        x = LinearAlgebra.view(input, j:n, j)
        MonteCarlo.reflectorApply!(x, τj, LinearAlgebra.view(input, j:n, j+1:n))
    end

    copyto!(U, I)
    @inbounds begin
        U[n, n] -= temp[n]
        for k = n-1:-1:1
            for j = k:n
                vBj = U[k,j]
                for i = k+1:n
                    vBj += conj(input[i,k]) * U[i,j]
                end
                vBj = temp[k]*vBj
                U[k,j] -= vBj
                for i = k+1:n
                    U[i,j] -= input[i,k] * vBj
                end
            end
        end
    end

    @avx for i in 1:n
        D[i] = abs(input.re[i, i])
    end

    _apply_pivot!(input, D, temp, pivot, apply_pivot)

    nothing
end

function _apply_pivot!(input::CMatF64, D, temp, pivot, ::Val{true})
    # TODO: optimize
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        for j in 1:i-1
            temp[pivot[j]] = zero(C)
        end
        for j in i:n
            temp[pivot[j]] = d * input[i, j]
        end
        for j in 1:n
            input[i, j] = temp[j]
        end
    end
end
function _apply_pivot!(input::CMat64, D, temp, pivot, ::Val{false})
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @avx for j in i:n
            input.re[i, j] = d * input.re[i, j]
        end
    end
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @avx for j in i:n
            input.im[i, j] = d * input.im[i, j]
        end
    end
end