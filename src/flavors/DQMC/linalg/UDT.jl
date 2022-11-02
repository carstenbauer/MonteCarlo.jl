################################################################################
### UDT Decomposition (no pivoting)
################################################################################

# fully in-place
# Probably don't use this as it has a larger error than the pivoted version

# Taken from julia Base
@inline function reflector!(x::AbstractVector)
    n = length(x)
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        @turbo for i = 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ξ1/normu) #zero(ξ1) ?
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        @turbo for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end

# Needed for pivoted as well
@inline function reflectorApply!(x::AbstractVector{<: Real}, τ::Real, A::StridedMatrix{<: Real})
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        @turbo for i = 2:m
            vAj += conj(x[i]) * A[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        A[1, j] -= vAj
        @turbo for i = 2:m
            A[i, j] -= x[i]*vAj
        end
    end
    return A
end

# Needed for pivoted as well
@inline function reflectorApply!(M::StridedArray{<: Real}, τ::Real, k::Int, n::Int)
    @inbounds for j = k+1:n
        # dot
        vAj = M[k, j]
        @turbo for i = k+1:n
            vAj += conj(M[i, k]) * M[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        M[k, j] -= vAj
        @turbo for i = k+1:n
            M[i, j] -= M[i, k]*vAj
        end
    end
    return M
end


"""
    udt_AVX!(U::Matrix, D::Vector, T::Matrix)

In-place calculation of a UDT (unitary - diagonal - (upper-)triangular) 
decomposition. The matrix `T` is simultaniously the input matrix that is 
decomposed and the triangular output matrix.

This assumes correctly sized square matrices as inputs.
"""
function udt_AVX!(U::AbstractMatrix{C}, D::AbstractVector{C}, input::AbstractMatrix{C}) where {C<:Number}
    # Assumptions:
    # - all matrices same size
    # - input can be changed (input becomes T)

    # @bm "Compute QR decomposition" begin
        n, _ = size(input)
        @inbounds D[n] = zero(C)
        @inbounds for k = 1:(n - 1 + !(C<:Real))
            x = LinearAlgebra.view(input, k:n, k)
            τk = reflector!(x)
            D[k] = τk
            # reflectorApply!(x, τk, view(input, k:n, k + 1:n))
            reflectorApply!(input, τk, k, n)
        end
    # end

    # @bm "Calculate Q" begin
        copyto!(U, I)
        @inbounds begin
            U[n, n] -= D[n]
            for k = n-1:-1:1
                for j = k:n
                    vBj = U[k,j]
                    @turbo for i = k+1:n
                        vBj += conj(input[i,k]) * U[i,j]
                    end
                    vBj = D[k]*vBj
                    U[k,j] -= vBj
                    @turbo for i = k+1:n
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        # U done
    # end

    # @bm "Calculate R" begin
        @inbounds for j in 1:n-1
            @turbo for i in max(1, j + 1):n
                input[i,j] = zero(input[i,j])
            end
        end
    # end

    # @bm "Calculate D" begin
        @inbounds for i in 1:n
            D[i] = abs(real(input[i, i]))
        end
    # end

    # @bm "Calculate T" begin
        @turbo for i in 1:n
            d = 1.0 / D[i]
            for j in 1:n
                input[i, j] = d * input[i, j]
            end
        end
    # end

    nothing
end



################################################################################
### UDT Decomposition (with pivoting)
################################################################################


# this method skips using a view(x, j:n, j) and requires passing of 
# normu = dot(view(x, j:n, j), view(x, j:n, j))
@inline function reflector!(x::Matrix{C}, normu, j=1, n=size(x, 1)) where {C <: Real}
    @inbounds begin
        ξ1 = x[j, j]
        if iszero(normu)
            return zero(ξ1) #zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[j, j] = -ν
        @turbo for i = j+1:n
            x[i, j] /= ξ1
        end
    end
    ξ1/ν
end


function indmaxcolumn(A::Matrix{C}, j=1, n=size(A, 1)) where {C <: Real}
    squared_norm = 0.0
    @turbo for k in j:n
        squared_norm += abs2(A[k, j])
    end
    ii = j
    @inbounds for i in j+1:n
        mi = 0.0
        @turbo for k in j:n
            mi += abs2(A[k, i])
        end
        if abs(mi) > squared_norm
            squared_norm = mi
            ii = i
        end
    end
    return ii, squared_norm
end


# Much higher accuracy, but a bit slower
"""
    udt_AVX_pivot!(
        U::Matrix, D::Vector, T::Matrix[, 
        pivot::Vector, temp::Vector, apply_pivoting = Val(true)
    ])

In-place calculation of a UDT (unitary - diagonal - (upper-)triangular) 
decomposition. The matrix `T` is simultaniously the input matrix that is 
decomposed and the triangular output matrix.

If `apply_pivoting = Val(true)` the `T` matrix will be pivoted such that 
`U * Diagonal(D) * T` matches the input. 
If `apply_pivoting = Val(false)` `T` will be a "dirty" upper triangular matrix 
(i.e. with random values elsewhere) which still requires pivoting. 
`rdivp!(A, T, temp, pivot)` is built explicitly for this case - it applies the 
pivoting while calculating `A T^-1`. Warning: BlockDiagonal matrices use 
per-block pivoting.

This assumes correctly sized square matrices as inputs.
"""
function udt_AVX_pivot!(
        U::AbstractArray{C, 2}, 
        D::AbstractArray{C, 1}, 
        input::AbstractArray{C, 2},
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{C, 1} = Vector{C}(undef, length(D)),
        apply_pivot::Val = Val(true)
    ) where {C<:Real}
    # Assumptions:
    # - all matrices same size
    # - input can be mutated (input becomes T)
    _temp = copy(input)

    # @bm "reset pivot" begin
        n = size(input, 1)
        @inbounds for i in 1:n
            pivot[i] = i
        end
    # end

    # @bm "QR decomposition" begin
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
                    @turbo for i = 1:n
                        tmp = input[i,jm]
                        input[i,jm] = input[i,j]
                        input[i,j] = tmp
                    end
                end
            # end

            # Compute reflector of columns j
            # @bm "Reflector" begin
                τj = reflector!(input, maxval, j, n)
                temp[j] = τj
            # end

            # Update trailing submatrix with reflector
            # @bm "apply" begin
                # TODO optimize?
                x = LinearAlgebra.view(input, j:n, j)
                # reflectorApply!(x, τj, LinearAlgebra.view(input, j:n, j+1:n))
                reflectorApply!(input, τj, j, n)
            # end
        end
    # end

    # @bm "Calculate Q" begin
        copyto!(U, I)
        @inbounds begin
            U[n, n] -= temp[n]
            for k = n-1:-1:1
                for j = k:n
                    vBj = U[k,j]
                    @turbo for i = k+1:n
                        vBj += conj(input[i,k]) * U[i,j]
                    end
                    vBj = temp[k]*vBj
                    U[k,j] -= vBj
                    @turbo for i = k+1:n
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        # U done
    # end

    # @bm "Calculate D" begin
        @inbounds for i in 1:n
            D[i] = abs(input[i, i])
        end
    # end

    # @bm "pivoted zeroed T w/ inv(D)" begin
        _apply_pivot!(input, D, temp, pivot, apply_pivot)
    # end

    if _isnan(input)
        @info "NaN in UDT"

        copyto!(input, _temp)
        @info "Intial input:"
        println(input)
        display(input)
        

        n = size(input, 1)
        @inbounds for i in 1:n
            pivot[i] = i
        end
        
        @inbounds for j = 1:n
            @info input[:, j]
            jm, maxval = indmaxcolumn(input, j, n)
            @info jm, maxval

            if jm != j
                tmpp = pivot[jm]
                pivot[jm] = pivot[j]
                pivot[j] = tmpp

                @turbo for i = 1:n
                    tmp = input[i,jm]
                    input[i,jm] = input[i,j]
                    input[i,j] = tmp
                end
            end
            @info input[:, j]

            τj = reflector!(input, maxval, j, n)
            temp[j] = τj
            @info τj
            display(input)
        
            x = LinearAlgebra.view(input, j:n, j)
            reflectorApply!(input, τj, j, n)
            display(input)
        end

        @info "input -> T:"
        display(input)
        
        copyto!(U, I)
        @inbounds begin
            U[n, n] -= temp[n]
            for k = n-1:-1:1
                for j = k:n
                    vBj = U[k,j]
                    @turbo for i = k+1:n
                        vBj += conj(input[i,k]) * U[i,j]
                    end
                    vBj = temp[k]*vBj
                    U[k,j] -= vBj
                    @turbo for i = k+1:n
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        
        @info "U (calc Q):"
        display(U)
        
        @inbounds for i in 1:n
            D[i] = abs(input[i, i])
        end
        
        @info "strip D:"
        display(D)
        
        _apply_pivot!(input, D, temp, pivot, apply_pivot)

        @info "U:"
        display(U)
        @info "D:"
        display(D)
        @info "T:"
        display(input)
    end

    nothing
end

function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{true}) where {C <: Real}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @inbounds for j in 1:i-1
            temp[pivot[j]] = zero(C)
        end
        @turbo for j in i:n
            temp[pivot[j]] = d * input[i, j]
        end
        @turbo for j in 1:n
            input[i, j] = temp[j]
        end
    end
end
function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{false}) where {C <: Real}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @turbo for j in i:n
            input[i, j] = d * input[i, j]
        end
    end
end



################################################################################
### Complex pivoted UDT
################################################################################



@inline function reflector!(x::Matrix{C}, normu, j=1, n=size(x, 1)) where {C <: ComplexF64}
    @inbounds begin
        ξ1 = x[j, j]
        if iszero(normu)
            return zero(ξ1) #zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = LinearAlgebra.copysign(normu, real(ξ1))
        ξ1 += ν
        x[j, j] = -ν
        for i = j+1:n
            x[i, j] /= ξ1
        end
    end
    ξ1/ν
end


@inline function reflectorApply!(x::AbstractVector, τ::Number, A::StridedMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        @inbounds for i = 2:m
            vAj += conj(x[i]) * A[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        A[1, j] -= vAj
        @inbounds for i = 2:m
            A[i, j] -= x[i]*vAj
        end
    end
    return A
end

function reflectorApply!(M::StridedArray, τ::Number, k::Int, n::Int)
    @inbounds for j = k+1:n
        # dot
        vAj = M[k, j]
        for i = k+1:n
            vAj += conj(M[i, k]) * M[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        M[k, j] -= vAj
        for i = k+1:n
            M[i, j] -= M[i, k]*vAj
        end
    end
    return M
end

function indmaxcolumn(A::AbstractMatrix{C}, j=1, n=size(A, 1)) where {C <: ComplexF64}
    max = 0.0
    for k in j:n
        max += abs2(A[k, j])
    end
    ii = j
    @inbounds for i in j+1:n
        mi = 0.0
        for k in j:n
            mi += abs2(A[k, i])
        end
        if abs(mi) > max
            max = mi
            ii = i
        end
    end
    return ii, max
end


# avx-less method for compatability with ComplexF64
function udt_AVX_pivot!(
        U::AbstractArray{C, 2}, 
        D::AbstractArray{Float64, 1}, 
        input::AbstractArray{C, 2},
        pivot::AbstractArray{Int64, 1} = Vector(UnitRange(1:size(input, 1))),
        temp::AbstractArray{C, 1} = Vector{C}(undef, length(D)),
        apply_pivot::Val = Val(true)
    ) where {C <: Complex}
    # Assumptions:
    # - all matrices same size
    # - input can be changed (input becomes T)

    # @bm "reset pivot" begin
        n = size(input, 1)
        @inbounds for i in 1:n
            pivot[i] = i
        end
    # end

    # @bm "QR decomposition" begin
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
                    for i = 1:n
                        tmp = input[i,jm]
                        input[i,jm] = input[i,j]
                        input[i,j] = tmp
                    end
                end
            # end

            # Compute reflector of columns j
            # @bm "Reflector" begin
                τj = reflector!(input, maxval, j, n)
                temp[j] = τj
            # end

            # Update trailing submatrix with reflector
            # @bm "apply" begin
                # TODO optimize?
                x = LinearAlgebra.view(input, j:n, j)
                # reflectorApply!(x, τj, LinearAlgebra.view(input, j:n, j+1:n))
                reflectorApply!(input, τj, j, n)
            # end
        end
    # end

    # @bm "Calculate Q" begin
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
                        U[i,j] -= input[i,k]*vBj
                    end
                end
            end
        end
        # U done
    # end

    # @bm "Calculate D" begin
        @inbounds for i in 1:n
            D[i] = abs(real(input[i, i]))
        end
    # end

    # @bm "pivoted zeroed T w/ inv(D)" begin
        _apply_pivot!(input, D, temp, pivot, apply_pivot)
    # end

    nothing
end

function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{true}) where {C <: Complex}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        @inbounds for j in 1:i-1
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
function _apply_pivot!(input::Matrix{C}, D, temp, pivot, ::Val{false}) where {C <: Complex}
    n = size(input, 1)
    @inbounds for i in 1:n
        d = 1.0 / D[i]
        for j in i:n
            input[i, j] = d * input[i, j]
        end
    end
end
