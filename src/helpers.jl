"""
    sparsity(A)

Calculates the sparsity of the given array.
The sparsity is defined as number of zero-valued elements divided by
total number of elements.
"""
function sparsity(A::AbstractArray{T}) where T<:Number
    (length(A)-countnz(A))/length(A)
end

"""
    reldiff(A, B)

Relative difference of absolute values of `A` and `B` defined as

``
\\operatorname{reldiff} = 2 \\dfrac{\\operatorname{abs}(A - B)}{\\operatorname{abs}(A+B)}.
``
"""
function reldiff(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
    return 2*abs.(A-B)./abs.(A+B)
end


"""
    effreldiff(A, B, threshold=1e-14)

Same as `reldiff(A,B)` but with all elements set to zero where corresponding element of
`absdiff(A,B)` is smaller than `threshold`. This is useful in avoiding artificially large
relative errors.
"""
function effreldiff(A::AbstractArray{T}, B::AbstractArray{S}, threshold::Float64=1e-14) where T<:Number where S<:Number
    r = reldiff(A,B)
    r[findall(x -> abs.(x)<threshold, absdiff(A,B))] .= 0.
    return r
end


"""
    absdiff(A, B)

Difference of absolute values of `A` and `B`.
"""
function absdiff(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
    return abs.(A-B)
end


"""
    compare(A, B)

Compares two matrices `A` and `B`, prints out the maximal absolute and relative differences
and returns a boolean indicating wether `isapprox(A,B)`.
"""
function compare(A::AbstractArray{T}, B::AbstractArray{S}) where T<:Number where S<:Number
    @printf("max absdiff: %.1e\n", maximum(absdiff(A,B)))
    @printf("mean absdiff: %.1e\n", mean(absdiff(A,B)))
    @printf("max reldiff: %.1e\n", maximum(reldiff(A,B)))
    @printf("mean reldiff: %.1e\n", mean(reldiff(A,B)))

    r = effreldiff(A,B)
    @printf("effective max reldiff: %.1e\n", maximum(r))
    @printf("effective mean reldiff: %.1e\n", mean(r))

    return isapprox(A,B)
end


# NOTE
# Currenlty julia/sparearrays does not implement this function (type signature)
# once it does this can be removed/depracted in favor of mul!
# see also: test/slice_matrices.jl
occursin(
    "SparseArrays",
    string(which(mul!, (Matrix, Matrix, SparseMatrixCSC)).file)
) && @warn(
    "A Method `mul!(::Matrix, ::Matrix, ::SparseMatrixCSC)` now exists in " *
    "`SparseArrays`. The method defined in `helpers.jl` is likely to be  " *
    "unnecessary now."
)

function SparseArrays.mul!(C::StridedMatrix, X::StridedMatrix, A::SparseMatrixCSC)
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(C, zero(eltype(C)))
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mX
            C[multivec_row, col] += X[multivec_row, ki] * kv
        end
    end
    C
end
