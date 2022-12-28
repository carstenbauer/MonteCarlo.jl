################################################################################
### Helpers
################################################################################


"""
    vmuladd!(C, A, B, factor=1.0)

`C += factor * A * B` with A, B, C matrix types
"""
function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] += factor * Cmn
    end
end

function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}, factor::T = T(1)) where {T <: Real}
    vmuladd!(C, A, B, axes(A, 2), factor)
end
function vmuladd!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}, range::AbstractVector, factor::T = T(1)) where {T <: Real}
    @views d = B.diag[range]
    @turbo for m in 1:size(A, 1), n in axes(A, 2)
        C[m,n] += factor * A[m,n] * d[n]
    end
end
function vmuladd!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    vmuladd!(C, A, B, axes(A, 1), factor)
end
function vmuladd!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}, range::AbstractVector, factor::T = T(1)) where {T <: Real}
    @views d = A.diag[range]
    @turbo for m in axes(A, 1), n in 1:size(A, 2)
        C[m,n] += factor * d[m] * B[m,n]
    end
end

function vmuladd!(C::Matrix{T}, A::Adjoint{T}, B::Diagonal{T}, factor::T = T(1)) where {T <: Real}
    vmuladd!(C, A, B, axes(A, 2), factor)
end
function vmuladd!(C::Matrix{T}, A::Adjoint{T}, B::Diagonal{T}, range::AbstractVector, factor::T = T(1)) where {T <: Real}
    @views d = B.diag[range]
    @turbo for m in 1:size(A, 1), n in axes(A, 2)
        C[m,n] += factor * A.parent[n,m] * d[n]
    end
end
function vmuladd!(C::Matrix{T}, A::Diagonal{T}, B::Adjoint{T}, factor::T = T(1)) where {T <: Real}
    vmuladd!(C, A, B, axes(A, 1), factor)
end
function vmuladd!(C::Matrix{T}, A::Diagonal{T}, B::Adjoint{T}, range::AbstractVector, factor::T = T(1)) where {T <: Real}
    @views d = A.diag[range]
    @turbo for m in axes(A, 1), n in 1:size(A, 2)
        C[m,n] += factor * d[m] * B.parent[n,m]
    end
end


function vmuladd!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}, factor::T = T(1)) where {T <: Real}
    B = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[n, k]
        end
        C[m,n] += factor * Cmn
    end
end
function vmuladd!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}, factor::T = T(1)) where {T <: Real}
    A = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[k,n]
        end
        C[m,n] += factor * Cmn
    end
end
function vmuladd!(C::Matrix{T}, X1::Adjoint{T}, X2::Adjoint{T}, factor::T = T(1)) where {T <: Real}
    A = X1.parent
    B = X2.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[n,k]
        end
        C[m,n] += factor * Cmn
    end
end


"""
    vmul!(C, A, B, factor)

`C = factor * A * B` with C, A, B matrices
"""
function vmul!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}, factor::T) where {T <: Real}
    B = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[n, k]
        end
        C[m,n] = factor * Cmn
    end
end
function vmul!(C::Matrix{T}, X1::Adjoint{T}, X2::Adjoint{T}, factor::T) where {T <: Real}
    A = X1.parent
    B = X2.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[n,k]
        end
        C[m,n] = factor * Cmn
    end
end
function vmul!(C::Matrix{T}, X1::Adjoint{T}, B::Matrix{T}, factor::T) where {T <: Real}
    A = X1.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[k, n]
        end
        C[m,n] = factor * Cmn
    end
end



################################################################################
### Overloads for complex StructArrays
################################################################################



const CMat64 = StructArray{ComplexF64, 2, NamedTuple{(:re, :im), Tuple{Matrix{Float64}, Matrix{Float64}}}, Int64}
const CVec64 = StructVector{ComplexF64, NamedTuple{(:re, :im), Tuple{Vector{Float64}, Vector{Float64}}}, Int64}

# constructors
CVec64(::UndefInitializer, n) = StructArray(Vector{ComplexF64}(undef, n))::CVec64
CMat64(::UndefInitializer, n, m) = StructArray(Matrix{ComplexF64}(undef, n, m))::CMat64
CMat64(::UniformScaling, n, m) = StructArray(Matrix{ComplexF64}(I, n, m))::CMat64

# util
function Base.copyto!(C::CMat64, ::UniformScaling)
    @turbo for i in axes(C.re, 1), j in axes(C.re, 2)
        C.re[i, j] = Float64(i == j)
    end
    @turbo for i in axes(C.re, 1), j in axes(C.re, 2)
        C.im[i, j] = 0.0
    end
    nothing
end
function Base.copyto!(C::CMat64, D::Diagonal{<: Real})
    @turbo for i in axes(C.re, 1), j in axes(C.re, 2)
        C.re[i, j] = Float64(i == j) * D.diag[i]
    end
    @turbo for i in axes(C.im, 1), j in axes(C.im, 2)
        C.im[i, j] = 0.0
    end
    nothing
end

# unoptimized - these (should) only run during initialization anyway
Base.exp(C::CMat64) = StructArray(exp(Matrix(C)))
fallback_exp(C::CMat64) = StructArray(fallback_exp(Matrix(C)))
Base.:(*)(C1::CMat64, C2::CMat64) = StructArray(Matrix(C1) * Matrix(C2))

# tested
@inline function vmul!(C::CMat64, A::CMat64, B::CMat64)
    vmul!(   C.re, A.re, B.re)       # C.re = A.re * B.re
    vmuladd!(C.re, A.im, B.im, -1.0) # C.re = C.re - A.im * B.im
    vmul!(   C.im, A.re, B.im)       # C.im = A.re * B.im
    vmuladd!(C.im, A.im, B.re)       # C.im = C.im + A.im * B.re
end

#tested
@inline function vmul!(C::CMat64, A::CMat64, B::Diagonal{<: Real})
    vmul!(C.re, A.re, B)
    vmul!(C.im, A.im, B)
end
@inline function vmul!(C::CMat64, A::CMat64, B::Diagonal{<: Real}, range::AbstractVector)
    vmul!(C.re, A.re, Diagonal(B.diag), range)
    vmul!(C.im, A.im, Diagonal(B.diag), range)
end

#tested
@inline function vmul!(C::CMat64, A::CMat64, D::Diagonal{ComplexF64, <: CVec64})
    vmul!(   C.re, A.re, Diagonal(D.diag.re))
    vmuladd!(C.re, A.im, Diagonal(D.diag.im), -1.0)
    vmul!(   C.im, A.re, Diagonal(D.diag.im))
    vmuladd!(C.im, A.im, Diagonal(D.diag.re))
end
@inline function vmul!(C::CMat64, A::CMat64, D::Diagonal{ComplexF64, <: CVec64}, range::AbstractVector)
    vmul!(   C.re, A.re, Diagonal(D.diag.re), range)
    vmuladd!(C.re, A.im, Diagonal(D.diag.im), range, -1.0)
    vmul!(   C.im, A.re, Diagonal(D.diag.im), range)
    vmuladd!(C.im, A.im, Diagonal(D.diag.re), range)
end

@inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::CMat64)
    vmul!(C.re, A, B.re)
    vmul!(C.im, A, B.im)
end
@inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::CMat64, range::AbstractVector)
    vmul!(C.re, A, B.re, range)
    vmul!(C.im, A, B.im, range)
end
@inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::CMat64)
    vmul!(   C.re, Diagonal(A.diag.re), B.re)
    vmuladd!(C.re, Diagonal(A.diag.im), B.im, -1.0)
    vmul!(   C.im, Diagonal(A.diag.re), B.im)
    vmuladd!(C.im, Diagonal(A.diag.im), B.re)
end
# @inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::CMat64, range)
#     vmul!(   C.re, Diagonal(A.diag.re), B.re, range)
#     vmuladd!(C.re, Diagonal(A.diag.im), B.im, range, -1.0)
#     vmul!(   C.im, Diagonal(A.diag.re), B.im, range)
#     vmuladd!(C.im, Diagonal(A.diag.im), B.re, range)
# end

# tested
@inline function vmul!(C::CMat64, A::CMat64, X::Adjoint{ComplexF64, <: CMat64})
    B = X.parent
    vmul!(   C.re, A.re, adjoint(B.re))
    vmuladd!(C.re, A.im, adjoint(B.im), 1.0)
    vmul!(   C.im, A.re, adjoint(B.im), -1.0)
    vmuladd!(C.im, A.im, adjoint(B.re))
end
# tested
@inline function vmul!(C::CMat64, X::Adjoint{ComplexF64, <:CMat64}, B::CMat64)
    A = X.parent
    vmul!(   C.re, adjoint(A.re), B.re)
    vmuladd!(C.re, adjoint(A.im), B.im, 1.0)
    vmul!(   C.im, adjoint(A.re), B.im)
    vmuladd!(C.im, adjoint(A.im), B.re, -1.0)
end
# tested
@inline function vmul!(C::CMat64, X::Adjoint{ComplexF64, <: CMat64}, Y::Adjoint{ComplexF64, <: CMat64})
    A = X.parent; B = Y.parent
    vmul!(   C.re, adjoint(A.re), adjoint(B.re))
    vmuladd!(C.re, adjoint(A.im), adjoint(B.im), -1.0)
    vmul!(   C.im, adjoint(A.re), adjoint(B.im), -1.0)
    vmuladd!(C.im, adjoint(A.im), adjoint(B.re), -1.0)
end

function vmul!(C::CMat64, D::Adjoint{<: Complex, <:Diagonal}, B::CMat64)
    vmul!(   C.re, Diagonal(D.parent.diag.re), B.re)
    vmuladd!(C.re, Diagonal(D.parent.diag.im), B.im)
    vmul!(   C.im, Diagonal(D.parent.diag.re), B.im)
    vmuladd!(C.im, Diagonal(D.parent.diag.im), B.re, -1.0)
end

# tested
@inline function rvmul!(A::CMat64, B::Diagonal{T}) where {T <: Real}
    rvmul!(A.re, B)
    rvmul!(A.im, B)
end
# tested
function rvmul!(A::CMat64, B::Diagonal{ComplexF64, <: CVec64})
    error("This multiplication cannot be done inplace.")
end

# tested
@inline function lvmul!(A::Diagonal{T}, B::CMat64) where {T <: Real}
    lvmul!(A, B.re)
    lvmul!(A, B.im)
end
# tested
function lvmul!(A::Diagonal{ComplexF64, <: CVec64}, B::CMat64)
    error("This multiplication cannot be done inplace.")
end

# tested
@inline rvadd!(A::CMat64, D::Diagonal{T}) where {T <: Real} = rvadd!(A.re, D)
# tested
@inline function rvadd!(A::CMat64, B::CMat64)
    rvadd!(A.re, B.re)
    rvadd!(A.im, B.im)
end

# test
@inline function vsub!(O::CMat64, A::CMat64, ::UniformScaling)
    copyto!(O.im, A.im)
    vsub!(O.re, A.re, I)
end



@bm function rdivp!(A::CMat64, T::CMat64, O::CMat64, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            @turbo for i in 1:N
                O.re[i, j] = A.re[i, p]
            end
            @turbo for i in 1:N
                O.im[i, j] = A.im[i, p]
            end
        end

        # do the rdiv
        # @turbo will segfault on `k in 1:0`, so pull out first loop 
        # invT = conj(T[1, 1]) / abs2(T[1, 1])
        invT = 1.0 / T[1, 1]
        re = real(invT)
        im = imag(invT)

        @turbo for i in 1:N
            A.re[i, 1] = O.re[i, 1] * re
        end
        @turbo for i in 1:N
            A.re[i, 1] -= O.im[i, 1] * im
        end
        @turbo for i in 1:N
            A.im[i, 1] = O.im[i, 1] * re
        end
        @turbo for i in 1:N
            A.im[i, 1] += O.re[i, 1] * im
        end

        # TODO Is this optimal?
        for j in 2:N
            # invT = conj(T[j, j]) / abs2(T[j, j])
            invT = 1.0 / T[j, j] # TODO this might always be real?
            re = real(invT)
            im = imag(invT)

            # These loops need to be seperated. If they are not results are 
            # with @turbo. I guess this is because x_re and x_im get written
            # to in parallel, i.e. values are overwritten?
            # this is worth it performance wise. 5-10x faster for 64^2 Matrix
            @turbo for i in 1:N 
                x_re = O.re[i, j]
                for k in 1:j-1
                    x_re -= A.re[i, k] * T.re[k, j]
                end
                A.re[i, j] = x_re * re
                A.im[i, j] = x_re * im
            end
            @turbo for i in 1:N
                x_re = 0.0
                for k in 1:j-1
                    x_re += A.im[i, k] * T.im[k, j]
                end
                A.re[i, j] += x_re * re
                A.im[i, j] += x_re * im
            end
            @turbo for i in 1:N
                x_im = O.im[i, j]
                for k in 1:j-1
                    x_im -= A.im[i, k] * T.re[k, j]
                end
                A.re[i, j] -= x_im * im
                A.im[i, j] += x_im * re
            end
            @turbo for i in 1:N
                x_im = 0.0
                for k in 1:j-1
                    x_im -= A.re[i, k] * T.im[k, j]
                end
                A.re[i, j] -= x_im * im
                A.im[i, j] += x_im * re
            end
            
        end
    end
    A
end



################################################################################
### UDT
################################################################################


# tested
# annoyingly we need a temp buffer for a real and imaginary numbers along the column...
# 0 allocs in @benchmark
@inline function reflector!(x::CMat64, normu, j, n, temp::CMat64)
    # Note:
    # `temp` is a CMat64 because that will be available in udt_AVX_pivot. Could
    # also use CVec64 of length size(x, 1) or Vector{Float64}(undef, 2size(x, 1))
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
        x.im[j, j] = 0.0

        # maybe this is partial copying suboptimal? (cache)
        @turbo for i in j+1:n
            temp.re[i] = x.re[i, j]
        end
        @turbo for i in j+1:n
            temp.im[i] = x.im[i, j]
        end

        @turbo for i = j+1:n
            x.re[i, j] = temp.re[i, 1] * real(invξ1)
        end
        @turbo for i = j+1:n
            x.re[i, j] -= temp.im[i, 1] * imag(invξ1)
        end
        @turbo for i = j+1:n
            x.im[i, j] = temp.im[i, 1] * real(invξ1)
        end
        @turbo for i = j+1:n
            x.im[i, j] += temp.re[i, 1] * imag(invξ1)
        end
    end
    ξ1/ν
end

# tested, 0 allocs in @benchmark
@inline function reflectorApply!(τ::Number, A::CMat64, j, n)
    @inbounds for k = j+1:n
        # dot
        vAk_re = A.re[j, k]
        @turbo for i = j+1:n
            vAk_re += A.re[i, j] * A.re[i, k]
        end
        @turbo for i = j+1:n
            vAk_re += A.im[i, j] * A.im[i, k] # conj implied
        end

        vAk_im = A.im[j, k]
        @turbo for i = j+1:n
            vAk_im -= A.im[i, j] * A.re[i, k]
        end
        @turbo for i = j+1:n
            vAk_im += A.re[i, j] * A.im[i, k]
        end

        # conj(τ) implied
        temp = real(τ) * vAk_re + imag(τ) * vAk_im
        vAk_im = real(τ) * vAk_im - imag(τ) * vAk_re
        vAk_re = temp

        # ger
        A.re[j, k] -= vAk_re
        @turbo for i = j+1:n
            A.re[i, k] -= A.re[i, j] * vAk_re
        end
        @turbo for i = j+1:n
            A.re[i, k] += A.im[i, j] * vAk_im
        end

        A.im[j, k] -= vAk_im
        @turbo for i = j+1:n
            A.im[i, k] -= A.re[i, j] * vAk_im
        end
        @turbo for i = j+1:n
            A.im[i, k] -= A.im[i, j] * vAk_re
        end
    end
  
    return A
end

# tested, 0 allocs in @benchmark
@inline function indmaxcolumn(A::CMat64, j=1, n=size(A, 1))
    squared_norm = 0.0
    @turbo for k in j:n
        squared_norm += abs2(A.re[k, j])
    end
    @turbo for k in j:n
        squared_norm += abs2(A.im[k, j])
    end
    ii = j
    @inbounds for i in j+1:n
        mi = 0.0
        @turbo for k in j:n
            mi += abs2(A.re[k, i])
        end
        @turbo for k in j:n
            mi += abs2(A.im[k, i])
        end
        if abs(mi) > squared_norm
            squared_norm = mi
            ii = i
        end
    end
    return ii, squared_norm
end

@bm function udt_AVX_pivot!(
        U::CMat64, 
        D::AbstractArray{Float64, 1}, 
        input::CMat64,
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{ComplexF64, 1} = Vector{ComplexF64}(undef, length(D)),
        apply_pivot::Val = Val(true)
    )
    # Assumptions:
    # - all matrices same size
    # - input can be changed (input becomes T)
    
    n = size(input.re, 1)
    @inbounds for i in 1:n
        pivot[i] = i
    end

    # # TODO optimize
    @inbounds for j in 1:n
        # # Find column with maximum norm in trailing submatrix
        jm, squared_max_norm = indmaxcolumn(input, j, n)

        if jm != j
            # Flip elements in pivoting vector
            tmpp = pivot[jm]
            pivot[jm] = pivot[j]
            pivot[j] = tmpp

            # Update matrix with
            @turbo for i = 1:n
                tmp = input.re[i,jm]
                input.re[i,jm] = input.re[i,j]
                input.re[i,j] = tmp
            end
            @turbo for i = 1:n
                tmp = input.im[i,jm]
                input.im[i,jm] = input.im[i,j]
                input.im[i,j] = tmp
            end
        end
        
        # Compute reflector of columns j
        τj = reflector!(input, squared_max_norm, j, n, U) # complex result
        temp[j] = τj

        # # Update trailing submatrix with reflector
        reflectorApply!(τj, input, j, n)
    end

    @turbo for i in axes(U.re, 1), j in axes(U.re, 2)
        U.re[i, j] = Float64(i == j)
    end
    @turbo for i in axes(U.im, 1), j in axes(U.im, 2)
        U.im[i, j] = 0.0
    end

    @inbounds begin
        U.re[n, n] -= real(temp[n])
        U.im[n, n] -= imag(temp[n])
        for k = n-1:-1:1
            for j = k:n
                vBj_re = U.re[k,j]
                @turbo for i = k+1:n
                    vBj_re += input.re[i,k] * U.re[i,j]
                end
                @turbo for i = k+1:n
                    vBj_re += input.im[i,k] * U.im[i,j]
                end

                vBj_im = U.im[k,j]
                @turbo for i = k+1:n
                    vBj_im += input.re[i,k] * U.im[i,j]
                end
                @turbo for i = k+1:n
                    vBj_im -= input.im[i,k] * U.re[i,j]
                end

                # we need the results in sync and this is not a turbo loop, so
                # temp should probably be Vector{Complex} rather than CVec64
                temp_re = real(temp[k])
                temp_im = imag(temp[k])
                re = temp_re * vBj_re - temp_im * vBj_im
                vBj_im = temp_im * vBj_re + temp_re * vBj_im
                vBj_re = re

                U.re[k,j] -= vBj_re
                U.im[k,j] -= vBj_im
                @turbo for i = k+1:n
                    U.re[i,j] -= input.re[i,k] * vBj_re
                end
                @turbo for i = k+1:n
                    U.re[i,j] += input.im[i,k] * vBj_im
                end
                @turbo for i = k+1:n
                    U.im[i,j] -= input.im[i,k] * vBj_re
                end
                @turbo for i = k+1:n
                    U.im[i,j] -= input.re[i,k] * vBj_im
                end
                
            end
        end
    end

    @turbo for i in 1:n
        D[i] = abs(input.re[i, i])
    end

    _apply_pivot!(input, D, pivot, apply_pivot)

    nothing
end

# function invert_pivot!(pivot::Vector{Int64})
#     from = 1
#     to = pivot[1]
#     @inbounds for _ in 1:16
#         temp = pivot[to]
#         pivot[to] = from
#         from = to
#         to = temp
#     end
#     pivot
# end

@bm function _apply_pivot!(input::CMat64, D, pivot, ::Val{true})
    n = size(input, 1)
    
    # without applying pivot we have a upper triangular matrix and we know
    # input.re[:, 1] = [sign(input.re[1, 1]), 0, ..., 0]
    # input.im[:, 1] = [input.im[1, 1] / D[1], 0, ..., 0]
    first_re = sign(input.re[1, 1])
    first_im = input.im[1, 1] / D[1]

    # use input.im[:, 1] as temporary storage (good cache)
    # I don't think we can do swaps like A[:, i] = A[:, j] because pivot can
    # have loops...
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        for j in 1:i-1 # can be 0-length which makes @turbo crash
            input.im[pivot[j], 1] = 0.0
        end
        @turbo for j in i:n
            input.im[pivot[j], 1] = d * input.re[i, j]
        end
        @turbo for j in 1:n
            input.re[i, j] = input.im[j, 1]
        end
    end

    # Now we use the one element column from input.re as a temp and repeat
    one_column = pivot[1]
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        for j in 1:i-1
            input.re[pivot[j], one_column] = 0.0
        end
        @turbo for j in i:n
            input.re[pivot[j], one_column] = d * input.im[i, j]
        end
        @turbo for j in 1:n
            input.im[i, j] = input.re[j, one_column]
        end
    end

    # Finally we restore the messed up columns
    input.re[1, one_column] = first_re
    @turbo for i in 2:n
        input.re[i, one_column] = 0.0
    end
    input.im[1, one_column] = first_im
    @turbo for i in 2:n
        input.im[i, one_column] = 0.0
    end

    input
end
@bm function _apply_pivot!(input::CMat64, D, pivot, ::Val{false})
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @turbo for j in i:n
            input.re[i, j] = d * input.re[i, j]
        end
    end
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @turbo for j in i:n
            input.im[i, j] = d * input.im[i, j]
        end
    end
end


################################################################################
### GHQ
################################################################################


@inline function vmul!(C::CMat64, A::Matrix{Float64}, D::Diagonal{Float64, FVec64})
    vmul!(C.re, A, D)
    C.im .= 0
    return
end
@inline function vmul!(C::CMat64, A::Matrix{Float64}, D::Diagonal{ComplexF64, <: CVec64})
    vmul!(C.re, A, Diagonal(D.diag.re))
    vmul!(C.im, A, Diagonal(D.diag.im))
end
# @inline function vmul!(C::CMat64, A::Matrix{<: Float64}, D::Diagonal{ComplexF64, <: CVec64}, range)
#     vmul!(C.re, A, Diagonal(D.diag.re[range]))
#     vmul!(C.im, A, Diagonal(D.diag.im[range]))
# end

@inline function vmul!(C::CMat64, D::Diagonal{Float64, FVec64}, B::Matrix{Float64})
    vmul!(C.re, D, B)
    C.im .= 0.0
    return
end
@inline function vmul!(C::CMat64, D::Diagonal{ComplexF64, <: CVec64}, B::Matrix{Float64})
    vmul!(C.re, Diagonal(D.diag.re), B)
    vmul!(C.im, Diagonal(D.diag.im), B)
end
# @inline function vmul!(C::CMat64, D::Diagonal{ComplexF64, <: CVec64}, B::Matrix{<: Float64}, range)
#     vmul!(C.re, Diagonal(D.diag.re[range]), B)
#     vmul!(C.im, Diagonal(D.diag.im[range]), B)
# end

@inline function vmul!(C::CMat64, A::Adjoint{ComplexF64, CMat64}, B::Matrix{Float64})
    vmul!(C.re, adjoint(A.parent.re), B)
    vmul!(C.im, adjoint(A.parent.im), B, -1.0)
end
@inline function vmul!(C::CMat64, A::CMat64, B::Matrix{Float64})
    vmul!(C.re, A.re, B)
    vmul!(C.im, A.im, B)
end

@inline function vmul!(C::CMat64, A::Matrix{Float64}, B::Adjoint{ComplexF64, CMat64})
    vmul!(C.re, A, adjoint(B.parent.re))
    vmul!(C.im, A, adjoint(B.parent.im), -1.0)
end
@inline function vmul!(C::CMat64, A::Matrix{Float64}, B::CMat64)
    vmul!(C.re, A, B.re)
    vmul!(C.im, A, B.im)
end


################################################################################
### Hopping Matrix / Hermitian extension
################################################################################


@inline function vmul!(C::CMat64, A::Adjoint{ComplexF64, <: CMat64}, B::Diagonal{<: Real})
    vmul!(C.re, adjoint(A.parent.re), B)
    vmul!(C.im, adjoint(A.parent.im), B, -1.0)
end
# @inline function vmul!(C::CMat64, A::Adjoint{ComplexF64, <: CMat64}, B::Diagonal{<: Real}, range)
#     vmul!(C.re, adjoint(A.parent.re), Diagonal(B.diag), range)
#     vmul!(C.im, adjoint(A.parent.im), Diagonal(B.diag), range, -1.0)
# end

@inline function vmul!(C::CMat64, A::Adjoint{ComplexF64, <: CMat64}, D::Diagonal{ComplexF64, <: CVec64})
    vmul!(   C.re, adjoint(A.parent.re), Diagonal(D.diag.re))
    vmuladd!(C.re, adjoint(A.parent.im), Diagonal(D.diag.im))
    vmul!(   C.im, adjoint(A.parent.re), Diagonal(D.diag.im))
    vmuladd!(C.im, adjoint(A.parent.im), Diagonal(D.diag.re), -1.0)
end
# @inline function vmul!(C::CMat64, A::Adjoint{ComplexF64, <: CMat64}, D::Diagonal{ComplexF64, <: CVec64}, range)
#     vmul!(   C.re, adjoint(A.parent.re), Diagonal(D.diag.re), range)
#     vmuladd!(C.re, adjoint(A.parent.im), Diagonal(D.diag.im), range)
#     vmul!(   C.im, adjoint(A.parent.re), Diagonal(D.diag.im), range)
#     vmuladd!(C.im, adjoint(A.parent.im), Diagonal(D.diag.re), range, -1.0)
# end

@inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::Adjoint{ComplexF64, <: CMat64})
    vmul!(C.re, A, adjoint(B.parent.re))
    vmul!(C.im, A, adjoint(B.parent.im), -1.0)
end
# @inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::Adjoint{ComplexF64, <: CMat64}, range)
#     vmul!(C.re, A, adjoint(B.parent.re), range)
#     vmul!(C.im, A, adjoint(B.parent.im), range, -1.0)
# end
@inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::Adjoint{ComplexF64, <: CMat64})
    vmul!(   C.re, Diagonal(A.diag.re), adjoint(B.parent.re))
    vmuladd!(C.re, Diagonal(A.diag.im), adjoint(B.parent.im))
    vmul!(   C.im, Diagonal(A.diag.im), adjoint(B.parent.re))
    vmuladd!(C.im, Diagonal(A.diag.re), adjoint(B.parent.im), -1.0)
end
# @inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::Adjoint{ComplexF64, <: CMat64}, range)
#     vmul!(   C.re, Diagonal(A.diag.re), adjoint(B.parent.re), range)
#     vmuladd!(C.re, Diagonal(A.diag.im), adjoint(B.parent.im), range)
#     vmul!(   C.im, Diagonal(A.diag.im), adjoint(B.parent.re), range)
#     vmuladd!(C.im, Diagonal(A.diag.re), adjoint(B.parent.im), range, -1.0)
# end

@inline function vmul!(C::CMat64, A::Adjoint{Float64}, B::Diagonal{<: Real})
    vmul!(C.re, A, B)
    copyto!(C.im, 0)
end
# @inline function vmul!(C::CMat64, A::Adjoint{Float64}, B::Diagonal{<: Real}, range)
#     vmul!(C.re, A, B, range)
#     copyto!(C.im, 0)
# end

@inline function vmul!(C::CMat64, A::Adjoint{Float64}, D::Diagonal{ComplexF64, <: CVec64})
    vmul!(C.re, A, Diagonal(D.diag.re))
    vmul!(C.im, A, Diagonal(D.diag.im))
end
@inline function vmul!(C::CMat64, A::Adjoint{Float64}, D::Diagonal{ComplexF64, <: CVec64}, range::AbstractVector)
    vmul!(C.re, A, Diagonal(D.diag.re), range)
    vmul!(C.im, A, Diagonal(D.diag.im), range)
end

@inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::Adjoint{Float64})
    vmul!(C.re, A, B)
    copyto!(C.im, 0)
end
# @inline function vmul!(C::CMat64, A::Diagonal{<: Real}, B::Adjoint{Float64}, range)
#     vmul!(C.re, A, B, range)
#     copyto!(C.im, 0)
# end
@inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::Adjoint{Float64})
    vmul!(C.re, Diagonal(A.diag.re), B)
    vmul!(C.im, Diagonal(A.diag.im), B)
end
@inline function vmul!(C::CMat64, A::Diagonal{ComplexF64, <: CVec64}, B::Adjoint{Float64}, range::AbstractVector)
    vmul!(C.re, Diagonal(A.diag.re), B, range)
    vmul!(C.im, Diagonal(A.diag.im), B, range)
end

@inline function vmul!(C::CMat64, A::CMat64, X::Adjoint{Float64})
    vmul!(C.re, A.re, X)
    vmul!(C.im, A.im, X)
end
@inline function vmul!(C::CMat64, X::Adjoint{Float64}, B::CMat64)
    vmul!(C.re, X, B.re)
    vmul!(C.im, X, B.im)
end
@inline function vmul!(C::CMat64, X::Adjoint{Float64}, Y::Adjoint{ComplexF64, <: CMat64})
    vmul!(C.re, X, adjoint(Y.parent.re))
    vmul!(C.im, X, adjoint(Y.parent.im), -1.0)
end
@inline function vmul!(C::CMat64, X::Adjoint{ComplexF64, <: CMat64}, Y::Adjoint{Float64})
    vmul!(C.re, adjoint(X.parent.re), Y)
    vmul!(C.im, adjoint(X.parent.im), Y, -1.0)
end