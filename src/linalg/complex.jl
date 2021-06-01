################################################################################
### Helpers
################################################################################



function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] += factor * Cmn
    end
end
function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}, factor::T = T(1)) where {T <: Real}
    @avx for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] += factor * A[m,n] * B.diag[n]
    end
end
function vmuladd!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}, factor::T = T(1)) where {T <: Real}
    B = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[n, k]
        end
        C[m,n] += factor * Cmn
    end
end
function vmuladd!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    A = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[k,n]
        end
        C[m,n] += factor * Cmn
    end
end
function vmul!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}, factor::T) where {T <: Real}
    B = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[n, k]
        end
        C[m,n] = factor * Cmn
    end
end
function vmul!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}, factor::T) where {T <: Real}
    A = X.parent
    @avx for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[k,n]
        end
        C[m,n] = factor * Cmn
    end
end



################################################################################
### Overloads for complex StructArrays
################################################################################



const CMat64 = StructArray{Complex{Float64},2,NamedTuple{(:re, :im), Tuple{AT, AT}}, I} where {AT <: AbstractArray{Float64, 2}, I}
const CVec64 = StructArray{Complex{Float64},1,NamedTuple{(:re, :im), Tuple{AT, AT}}, I} where {AT <: AbstractArray{Float64, 1}, I}

function vmul!(C::CMat64, A::CMat64, B::CMat64)
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    vmul!(   C.re, A.re, B.re)     # C.re = A.re * B.re
    vmuladd!(C.re, A.im, B.im, -1.0) # C.re = C.re - A.im * B.im
    vmul!(   C.im, A.re, B.im)     # C.im = A.re * B.im
    vmuladd!(C.im, A.im, B.re)     # C.im = C.im + A.im * B.re
end

function vmul!(C::CMat64, A::CMat64, B::Diagonal{T}) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    vmul!(C.re, A.re, B)
    vmul!(C.im, A.im, B)
end
function vmul!(C::CMat64, A::CMat64, B::Diagonal{ComplexF64, CVec64})
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
    vmuladd!(C.re, A.im, adjoint(B.im), 1.0)
    vmul!(   C.im, A.re, adjoint(B.im), -1.0)
    vmuladd!(C.im, A.im, adjoint(B.re))
end
function vmul!(C::CMat64, X::Adjoint{T}, B::CMat64) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    A = X.parent
    vmul!(   C.re, adjoint(A.re), B.re)
    vmuladd!(C.re, adjoint(A.im), B.im, 1.0)
    vmul!(   C.im, adjoint(A.re), B.im)
    vmuladd!(C.im, adjoint(A.im), B.re, -1.0)
end

function rvmul!(A::CMat64, B::Diagonal{T}) where {T <: Real}
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    rvmul!(A.re, B)
    rvmul!(A.im, B)
end
function rvmul!(A::CMat64, B::Diagonal{ComplexF64})
    @warn "Complex StructArrays are untested not really optimized." maxlog=10
    @inbounds for m in 1:size(A, 1), n in 1:size(A, 2)
        A[m,n] = A[m, n] * B.diag[n]
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
        B[m,n] = A.diag[m] * B[m, n]
    end
end

rvadd!(A::CMat64, D::Diagonal{T}) where {T <: Real} = rvadd!(A.re, D)



##################################################
### TODO
##################################################



function rdivp!(A::CMat64, T::CMat64, O::CMat64, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            @avx for i in 1:N
                O.re[i, j] = A.re[i, p]
            end
            @avx for i in 1:N
                O.im[i, j] = A.im[i, p]
            end
        end

        # do the rdiv
        # @avx will segfault on `k in 1:0`, so pull out first loop 
        invT = conj(T[1, 1]) / abs2(T[1, 1])
        re = real(invT); im = imag(invT)
        @avx for i in 1:N
            A.re[i, 1] = O.re[i, 1] * re
        end
        @avx for i in 1:N
            A.re[i, 1] -= O.im[i, 1] * im
        end
        @avx for i in 1:N
            A.im[i, 1] = O.im[i, 1] * re
        end
        @avx for i in 1:N
            A.im[i, 1] += O.re[i, 1] * im
        end

        # TODO Is this optimal?
        for j in 2:N
            invT = conj(T[j, j]) / abs2(T[j, j])
            re = real(invT); im = imag(invT)
            
            @avx for i in 1:N
                x = O.re[i, j]
                for k in 1:j-1
                    x -= A.re[i, k] * T.re[k, j]
                end
                for k in 1:j-1
                    x += A.im[i, k] * T.im[k, j]
                end
                A.re[i, j] = x * re
                A.im[i, j] = x * im
            end

            @avx for i in 1:N
                x = O.im[i, j]
                for k in 1:j-1
                    x -= A.im[i, k] * T.re[k, j]
                end
                for k in 1:j-1
                    x -= A.re[i, k] * T.im[k, j]
                end
                A.re[i, j] -= x * im
                A.im[i, j] += x * re
            end
        end
    end
    A
end



################################################################################
### UDT
################################################################################



@inline function reflector!(x::CMat64, normu, j=1, n=size(x, 1))
    @inbounds begin
        ξ1 = x[j, j]
        if iszero(normu)
            return zero(ξ1) #zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        invξ1 = 1.0 / ξ1
        x.re[j, j] = -ν
        @avx for i = j+1:n
            x.re[i, j] = x.re[i, j] * real(invξ1)
        end
        @avx for i = j+1:n
            x.re[i, j] = -x.im[i, j] * imag(invξ1)
        end
        @avx for i = j+1:n
            x.im[i, j] = x.im[i, j] * real(invξ1)
        end
        @avx for i = j+1:n
            x.im[i, j] = x.re[i, j] * imag(invξ1)
        end
    end
    ξ1/ν
end

@inline function reflectorApply!(x::CVec64, τ::Number, A::CMat64)
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj_re = A.re[1, j]
        @avx for i = 2:m
            vAj_re += x.re[i] * A.re[i, j]
        end
        @avx for i = 2:m
            vAj_re += x.im[i] * A.im[i, j]
        end

        vAj_im = A.im[1, j]
        @avx for i = 2:m
            vAj_im -= x.im[i] * A.re[i, j]
        end
        @avx for i = 2:m
            vAj_im += x.re[i] * A.im[i, j]
        end

        temp = real(τ) * vAj_re + imag(τ) * vAj_im
        vAj_im = real(τ) * vAj_im - imag(τ) * vAj_re
        vAj_re = temp

        # ger
        A.re[1, j] -= vAj_re
        @avx for i = 2:m
            A.re[i, j] -= x.re[i] * vAj_re
        end
        @avx for i = 2:m
            A.re[i, j] += x.im[i] * vAj_im
        end

        A.im[1, j] -= vAj_im
        @avx for i = 2:m
            A.im[i, j] -= x.re[i] * vAj_im
        end
        @avx for i = 2:m
            A.im[i, j] -= x.im[i] * vAj_re
        end
    end
    return A
end

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

    copyto!(U.re, I)
    copyto!(U.im, 0.0)
    @inbounds begin
        U.re[n, n] -= temp.re[n]
        U.im[n, n] -= temp.im[n]
        for k = n-1:-1:1
            for j = k:n
                vBj_re = U.re[k,j]
                @avx for i = k+1:n
                    vBj_re += input.re[i,k] * U.re[i,j]
                end
                @avx for i = k+1:n
                    vBj_re += input.im[i,k] * U.im[i,j]
                end
                vBj_im = U.im[k,j]
                @avx for i = k+1:n
                    vBj_im += input.re[i,k] * U.im[i,j]
                end
                @avx for i = k+1:n
                    vBj_im -= input.im[i,k] * U.re[i,j]
                end

                re = temp.re[k] * vBj_re - temp.im[k] * vBj_im
                vBj_im = temp.im[k] * vBj_re + temp.re[k] * vBj_im
                vBj_re = re

                U.re[k,j] -= vBj_re
                U.im[k,j] -= vBj_im
                @avx for i = k+1:n
                    U.re[i,j] -= input.re[i,k] * vBj_re
                end
                @avx for i = k+1:n
                    U.re[i,j] += input.im[i,k] * vBj_im
                end
                @avx for i = k+1:n
                    U.im[i,j] -= input.im[i,k] * vBj_re
                end
                @avx for i = k+1:n
                    U.im[i,j] -= input.re[i,k] * vBj_im
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

function _apply_pivot!(input::CMat64, D, temp, pivot, ::Val{true})
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