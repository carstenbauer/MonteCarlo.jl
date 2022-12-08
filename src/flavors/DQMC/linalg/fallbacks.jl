function vmul!(C, A, B)
    @debug "vmul!($(typeof(C)), $(typeof(A)), $(typeof(B)))"
    mul!(C, A, B)
end
function vmul!(C, A::Diagonal, B, range)
    @debug "vmul!($(typeof(C)), $(typeof(A)), $(typeof(B)), $(typeof(range)))"
    @views mul!(C, Diagonal(A.diag[range]), B)
end
function vmul!(C, A, B::Diagonal, range)
    @debug "vmul!($(typeof(C)), $(typeof(A)), $(typeof(B)), $(typeof(range)))"
    @views mul!(C, A, Diagonal(B.diag[range]))
end
function rvmul!(A, B)
    @debug "rvmul!($(typeof(A)), $(typeof(B)))"
    rmul!(A, B)
end
function lvmul!(A, B)
    @debug "lvmul!($(typeof(A)), $(typeof(B)))"
    lmul!(A, B)
end
function rvadd!(A, B)
    @debug "rvadd!($(typeof(A)), $(typeof(B)))"
    A .+= B
end
function vsub!(A, B, C)
    @debug "vsub!($(typeof(A)), $(typeof(B)), $(typeof(C)))"
    A .= B .- C
end
function vsub!(A, B, ::UniformScaling)
    @debug "vsub!($(typeof(A)), $(typeof(B)), ::UniformScaling)"
    T1 = one(eltype(A))
    T0 = zero(eltype(A))
    @inbounds for i in axes(A, 1), j in axes(A, 2)
        A[i, j] = B[i, j] - ifelse(i==j, T1, T0)
    end
    A
end

function rdivp!(A::Matrix{<: Complex}, T::Matrix{<: Complex}, O::Matrix{<: Complex}, pivot)
    @debug "rdivp!($(typeof(A)), $(typeof(T)), $(typeof(O)), $(typeof(pivot)))"
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

# Hermitian optimization
vmul!(C::AbstractArray, A::AbstractArray, B::Hermitian) = vmul!(C, A, adjoint(B.data)) # slightly faster
vmul!(C::AbstractArray, A::Hermitian, B::AbstractArray) = vmul!(C, adjoint(A.data), B) # lots faster
function vmul!(C::AbstractArray, A::AbstractArray, B::Adjoint{<: Any, <: Hermitian})
    vmul!(C, A, adjoint(B.parent.data))
end
function vmul!(C::AbstractArray, A::Adjoint{<: Any, <: Hermitian}, B::AbstractArray)
    vmul!(C, adjoint(A.parent.data), B)
end