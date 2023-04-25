# Linear Algebra

MonteCarlo.jl makes heavy use of LoopVectorization to speed up linear algebra. The respective functions are mostly in "flavors/DQMC/linalg", with a few more specific functions in places where they are needed. These functions are generally identified with a `v` prefix.

Generally the methods mirror the respective methods from LinearAlgebra - `vmul!` is a matrix multiplication like `mul!`, `rvmul!` and `lvmul!` match `rmul!` and `vmul!`. Usually MonteCarlo.jl only implements the methods that are actually used, so there won't be a method for every possible type combination. On the other hand there are also some new methods which are used with specialized matrix types, such as `vmul!(output, left, right, range)`.

To summarize the existing functions:

- `vmul!(target, left, right)` calculates `target = left * right` in place
- `rvmul!(left, right::Diagonal)` calculates `left = left * right` in place
- `lvmul!(left::Diagonal, right)` calculates `right = left * right` in place
- `rvadd!(left, right)` calculates `left += right` in place
- `vsub!(output, left, ::UniformScaling)` calculates `output = left - I` in place
- `vmin!(v::Vector, w::Vector)` calculates `v .= min.(1, w)`
- `vmininv!(v::Vector, w::Vector)` calculates `v .= 1 ./ min.(1, w)`
- `vmax!(v::Vector, w::Vector)` calculates `v .= max.(1, w)`
- `vmaxinv!(v::Vector, w::Vector)` calculates `v .= 1 ./ max.(1, w)`
- `vinv!(v::Vector)` calculates `v .= 1 ./ v`
- `vinv!(v::Vector, w::Vector)` calculates `v .= 1 ./ w`
- `rdivp!(left, T, temp, pivot)` calculates `left * T^-1` where T is a pivoted upper triangular matrix as returned by `udt_AVX_pivot(U, D, T, pivot, temp, Val(false))`

And finally `udt_AVX_pivot!(U, D::Diagonal, T[, pivot, temp, apply_pivot::Val])` which performs a pivoted UDT decomposition, which is a QR decomposition with the diagonal values pulled out of the `R` matrix. The `T` matrix doubles as the input, `pivot` is an integer pivoting vector and `temp` is a temporary vector with eltype matching the input matrix . `apply_pivot` sets whether the output `T` matrix has pivoting applied, making it a true upper triangular matrix or not.

## Special Types

For optimization purposes we have a couple of custom matrix types.

The simplest is `CVec64` and `CMat64` which is just a shortened name for the `StructArray` of a complex vector or matrix. As noted in LoopVectorization `StructArray` is needed to handle complex matrices efficiently.

The next type is `BlockDiagonal` which is a matrix type consisting of square matrices on the diagonal. This type is used with, for example, two flavor problems which don't include terms mixing flavors. In that case the off-diagonal blocks are 0 matrices and don't need to be considered. Note that this type is only thoroughly tested for two equally sized blocks, and may need some extra work to extend to an arbitrary number of blocks.

The most complicated one is `CheckerboardDecomposed` which encodes a checkerboard decomposition of a hopping matrix. This decomposition uses the bond structure of the lattice and that the hopping matrix must be Hermitian to decompose it into a chain of sparse matrices, which can be multiplied onto a regular matrix more quickly. Note that this decomposition may increase or decrease the error caused by the Trotter decomposition.

Finally we have `DiagonallyRepeatingMatrix` which is a thin wrapper around a regular matrix type. It simply indicates that the contained matrix repeats along the diagonal. This is the case for models with multiple flavors, where the flavors mirror each other. This is only used in measurements.