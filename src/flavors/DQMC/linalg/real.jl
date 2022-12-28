################################################################################
### AVX-powered matrix multiplications for real matrices
################################################################################



function vmul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end
function vmul!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A[m,n] * B.diag[n]
    end
end
function vmul!(C::Matrix{T}, A::Matrix{T}, B::Diagonal{T}, range::AbstractVector) where {T <: Real}
    @views d = B.diag[range]
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A[m,n] * d[n]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = A.diag[m] * B[m,n]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Matrix{T}, range::AbstractVector) where {T <: Real}
    @views d = A.diag[range]
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = d[m] * B[m,n]
    end
end

function vmul!(C::Matrix{T}, A::Adjoint, B::Diagonal{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A.parent[n,m] * B.diag[n]
    end
end
function vmul!(C::Matrix{T}, A::Adjoint, B::Diagonal{T}, factor::Real) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = factor * A.parent[n,m] * B.diag[n]
    end
end
function vmul!(C::Matrix{T}, A::Adjoint, B::Diagonal{T}, range::AbstractVector) where {T <: Real}
    @views d = B.diag[range]
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        C[m,n] = A.parent[n,m] * d[n]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Adjoint) where {T <: Real}
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = A.diag[m] * B.parent[n,m]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Adjoint, factor::Real) where {T <: Real}
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = factor * A.diag[m] * B.parent[n,m]
    end
end
function vmul!(C::Matrix{T}, A::Diagonal{T}, B::Adjoint, range::AbstractVector) where {T <: Real}
    @views d = A.diag[range]
    @turbo for m in 1:size(C, 1), n in 1:size(C, 2)
        C[m,n] = d[m] * B.parent[n,m]
    end
end

function vmul!(C::Matrix{T}, A::Matrix{T}, X::Adjoint{T}) where {T <: Real}
    B = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[m,k] * conj(B[n, k])
        end
        C[m,n] = Cmn
    end
end
function vmul!(C::Matrix{T}, X::Adjoint{T}, B::Matrix{T}) where {T <: Real}
    A = X.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += conj(A[k,m]) * B[k,n]
        end
        C[m,n] = Cmn
    end
end
function vmul!(C::Matrix{T}, X1::Adjoint{T}, X2::Adjoint{T}) where {T <: Real}
    A = X1.parent
    B = X2.parent
    @turbo for m in 1:size(A, 1), n in 1:size(B, 2)
        Cmn = zero(eltype(C))
        for k in 1:size(A, 2)
            Cmn += A[k,m] * B[n,k]
        end
        C[m,n] = Cmn
    end
end

function vmul!(C::MT, D::Adjoint{T, <: Diagonal}, B::MT) where {T <: Real, MT <: Matrix{T}}
    vmul!(C, D.parent, B)
end

function rvmul!(A::Matrix{T}, B::Diagonal{T}) where {T <: Real}
    @turbo for m in 1:size(A, 1), n in 1:size(A, 2)
        A[m,n] = A[m,n] * B.diag[n]
    end
end
function lvmul!(A::Diagonal{T}, B::Matrix{T}) where {T <: Real}
    @turbo for m in 1:size(B, 1), n in 1:size(B, 2)
        B[m,n] = A.diag[m] * B[m, n]
    end
end
function rvadd!(A::Matrix{T}, D::Diagonal{T}) where {T <: Real}
    @turbo for i in axes(A, 1)
        A[i, i] = A[i, i] + D.diag[i]
    end
end
function rvadd!(A::Matrix{T}, B::Matrix{T}) where {T <: Real}
    @turbo for i in axes(A, 1), j in axes(A, 2)
        A[i, j] = A[i, j] + B[i, j]
    end
end
function vsub!(O::Matrix{T}, A::Matrix{T}, ::UniformScaling) where {T <: Real}
    # Note: 
    # one loop with A[i, j] - T(i = j) seems to have similar min, worse max time
    T1 = one(T)
    @turbo for i in axes(O, 1), j in axes(O, 2)
        O[i, j] = A[i, j]
    end
    @turbo for i in axes(O, 1)
        O[i, i] -= T1
    end
end

# NOTE
# These should only be called with real Vectors, no need to implement extra
# methods
function vmin!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(w)
        v[i] = min(T1, w[i])
    end
    v
end
function vmininv!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(w)
        v[i] = T1 / min(T1, w[i])
    end
    v
end
function vmax!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(w)
        v[i] = max(T1, w[i])
    end
    v
end
function vmaxinv!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(w)
        v[i] = T1 / max(T1, w[i])
    end
    v
end
function vinv!(v::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(v)
        v[i] = T1 / v[i]
    end
    v
end
function vinv!(v::Vector{T}, w::Vector{T}) where {T<:Real}
    T1 = one(T)
    @turbo for i in eachindex(v)
        v[i] = T1 / w[i]
    end
    v
end



"""
    rdivp!(A, T, O, pivot)

Computes `A * T^-1` where `T` is an upper triangular matrix which should be 
pivoted according to a pivoting vector `pivot`. `A`, `T` and `O` should all be
square matrices of the same size. The result will be written to `A` without 
changing `T`.

This function is written to work with (@ref)[`udt_AVX_pivot!`].
"""
function rdivp!(A::Matrix, T, O, pivot)
    # assume Diagonal is ±1!
    @inbounds begin
        N = size(A, 1)

        # Apply pivot
        for (j, p) in enumerate(pivot)
            @turbo for i in 1:N
                O[i, j] = A[i, p]
            end
        end

        # do the rdiv
        # @turbo will segfault on `k in 1:0`, so pull out first loop 
        @turbo for i in 1:N
            A[i, 1] = O[i, 1] / T[1, 1]
        end
        for j in 2:N
            @turbo for i in 1:N
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

function fallback_exp(A)
    try
        exp(A)
    catch e
        @error "LAPACK failed to calculate matrix exponentials... Using Taylor expansion" exception = e
        taylor_exp(A)
    end
end

to_bigfloat(x::Float64) = BigFloat(x)
to_bigfloat(x::Complex) = Complex(BigFloat(real(x)), BigFloat(imag(x)))
function taylor_exp(M::AbstractMatrix{T}; step_precision = 1, max_iter = 10_000) where T
    # if the next step adds less than step_precision (in unit of output float 
    # epsilons) we stop
    old = precision(BigFloat)
    setprecision(BigFloat, 128)
    A = to_bigfloat.(M)
    temp = A
    output = I + A
    for n in 2:max_iter
        temp = temp * A ./ n
        output += temp
        if all(abs.(temp) ./ eps.(abs.(output)) .< step_precision) # abs for Complex
            setprecision(BigFloat, old)
            return T.(output)
        end
    end
    setprecision(BigFloat, old)
    error("Failed to generate sufficiently accurate matrix exponential.")
end