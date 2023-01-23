# For now this is only used in measurements to differentiate a full Matrix from
# one which contains copies at flv = 1 and flv = 2

struct DiagonallyRepeatingMatrix{T, MT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    val::MT
end

Base.size(D::DiagonallyRepeatingMatrix) = size(D.val)
Base.getindex(D::DiagonallyRepeatingMatrix, idx) = getindex(D.val, idx)