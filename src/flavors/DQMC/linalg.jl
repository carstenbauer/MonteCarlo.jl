# See StableDQMC.jl

# UDT decomposition (basically QR)
struct UDT{E,Er<:Real,M<:AbstractArray{E}} <: Factorization{E}
    U::M
    D::Vector{Er}
    T::M
    function UDT{E,Er,M}(U, D, T) where {E,Er,M<:AbstractArray{E}}
        new{E,Er,M}(U, D, T)
    end
end

UDT(U::AbstractArray{E}, D::Vector{Er}, T::AbstractArray{E}) where {E,Er<:Real} =
    UDT{E,Er,typeof(U)}(U, D, T)
function UDT{E}(U::AbstractArray, D::AbstractVector{Er}, T::AbstractArray) where {E,Er<:Real}
    UDT(convert(AbstractArray{E}, U),
        convert(Vector{Er}, D),
        convert(AbstractArray{E}, T))
end


# Conversion
Base.AbstractMatrix(F::UDT) = (F.U * Diagonal(F.D)) * F.T
Base.AbstractArray(F::UDT) = AbstractMatrix(F)
Base.Matrix(F::UDT) = Array(AbstractArray(F))
Base.Array(F::UDT) = Matrix(F)

function Matrix!(res::AbstractMatrix, F::UDT)
    tmp = F.U * Diagonal(F.D)
    mul!(res, tmp, F.T)
end


# iteration for destructuring into components
Base.iterate(S::UDT) = (S.U, Val(:D))
Base.iterate(S::UDT, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::UDT, ::Val{:T}) = (S.T, Val(:done))
Base.iterate(S::UDT, ::Val{:done}) = nothing


Base.size(A::UDT, dim::Integer) = dim == 1 ? size(A.U, dim) : size(A.T, dim)
Base.size(A::UDT) = (size(A, 1), size(A, 2))

Base.similar(A::UDT) = UDT(similar(A.U), similar(A.D), similar(A.T))



# decomposition functions
"""
Compute the UDT decomposition of `A` and return an `UDT` object.

`U`, `D`, and `T`, can be obtained from the factorization `F`
with `F.U`, `F.D`, and `F.T` such that `A = U * Diagonal(D) * T`.

Iterating the decomposition produces the components `U`, `D`, and `V`.

Note that `T` is upper triangular only up to permutations of columns of `T`.
"""
function udt(A::AbstractMatrix{C}) where {C<:Number}
  F = qr(A, Val(true))
  _qr_to_udt(A, F)
end


"""
`udv!` is the same as `svd`, but saves space by overwriting the input `A`,
instead of creating a copy.
"""
function udt!(A::AbstractMatrix{C}) where {C<:Number}
  F = qr!(A, Val(true))
  _qr_to_udt(A, F)
end


@inline function _qr_to_udt(A::AbstractMatrix{C}, F::QRPivoted) where {C<:Number}
    n = size(A, 1)
    D = Vector{real(float(C))}(undef, n)
    R = F.R # F.R has regular matrix type
    @views F.p[F.p] = 1:n

    @inbounds for i in 1:n
    D[i] = abs(real(R[i,i]))
    end
    lmul!(Diagonal(1 ./ D), R)
    UDT(Matrix(F.Q), D, R[:, F.p])
end


function udt(x::Number)
    UDT(x == 0 ? fill(one(x), 1, 1) : fill(x/abs(x), 1, 1), [abs(x)], fill(one(x), 1, 1))
end
function udt(x::Integer)
    udt(float(x))
end


# operations
"""
    inv(F::UDT) -> AbstractMatrix

Computes the inverse matrix of the `UDT` decomposition of a matrix.
"""
function Base.inv(F::UDT)
    inv!(similar(F.U), F)
end


"""
    inv!(res, F::UDT) -> res

Same as `inv` but writes result into preallocated `res`.
"""
function inv!(res::M, F::UDT{E, Er, M}) where {E,Er,M}
    tmp = similar(F.U)
    ldiv!(tmp, lu(F.T), Diagonal(1 ./ F.D))
    mul!(res, tmp, F.U')
    return res
end


"""
    fact_mult(A::UDT, B::UDT) -> UDT

Stabilized multiplication of two `UDT` decompositions.
Returns a `UDT` factorization object.
"""
function fact_mult(A::UDT, B::UDT)
    mat = A.T * B.U
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = udt!(mat)
    UDT(A.U * F.U, F.D, F.T * B.T)
end


# """
#     *(A::UDT, B::UDT)

# Stabilized multiplication of two `UDT` decompositions.
# """
# function Base.:*(A::UDT, B::UDT)
#     mat = A.T * B.U
#     lmul!(Diagonal(A.D), mat)
#     rmul!(mat, Diagonal(B.D))
#     F = udt!(mat)
#     (A.U * F.U) * Diagonal(F.D) * (F.T * B.T)
# end






##############################################################
#
#                   QR / UDT
#
##############################################################


"""
    udt_inv_one_plus(F::UDT) -> UDT

Stabilized calculation of [1 + UDT]^(-1). Returns and
`UDT` factorization object.

Optional preallocations via keyword arguments:

  * `u = similar(F.U)`
  * `t = similar(F.T)`
"""
function udt_inv_one_plus(F::UDT; u = similar(F.U), t = similar(F.T))
  U, D, T = F

  m = U' / T
  m[diagind(m)] .+= D

  utmp, d, ttmp = udt!(m)
  mul!(u, U, utmp)
  mul!(t, ttmp, T)

  UDT(inv(t), 1 ./ d, copy(u'))
end


"""
  inv_one_plus!(res, F::UDT) -> res

Same as `inv_one_plus` but stores the result in preallocated `res`.
"""
function inv_one_plus!(res, F::UDT;
                       u = similar(F.U),
                       d = similar(F.D),
                       t = similar(F.T))
  U, D, T = F

  m = U' / T
  m[diagind(m)] .+= D

  utmp, d, ttmp = udt!(m)
  mul!(u, U, utmp)
  mul!(t, ttmp, T)

  ldiv!(m, lu!(t), Diagonal(1 ./ d))

  mul!(res, m, u')
  res
end


"""
    inv_one_plus(F::UDT) -> AbstractMatrix

Stabilized calculation of `[1 + UDT]^(-1)`:

  * Use one intermediate UDT decomposition.

Faster but potentially less accurate than `inv_one_plus_loh`.

See `udt_inv_one_plus` for preallocation options.
"""
function inv_one_plus(F::UDT; kwargs...)
  res = similar(F.U)
  inv_one_plus!(res, F; kwargs...)
  return res
end


"""
    udt_inv_one_plus(A::UDT, Bdagger::UDT) -> UDT

Stabilized calculation of [1 + UlDlTl(UrDrTr)^†]^(-1). Returns and
`UDT` factorization object.

Optional preallocations via keyword arguments:

  * `tmp = similar(A.U)`
  * `tmp2 = similar(A.U)`
  * `tmp3 = similar(A.U)`
"""
function udt_inv_one_plus(A::UDT, Bdagger::UDT;
                          tmp = similar(A.U),
                          tmp2 = similar(A.U),
                          tmp3 = similar(A.U),
                          internaluse = false)
  Ul,Dl,Tl = A
  Ur,Dr,Tr = Bdagger

  mul!(tmp, Tl, adjoint(Tr))
  rmul!(tmp, Diagonal(Dr))
  lmul!(Diagonal(Dl), tmp)
  U1, D1, T1 = udt!(tmp)

  mul!(tmp3, Ul, U1)
  mul!(tmp2, T1, adjoint(Ur))
  mul!(tmp, adjoint(tmp3), inv(tmp2))

  tmp .+= Diagonal(D1)

  u, d, t = udt!(tmp)
  mul!(tmp, t, tmp2)
  mul!(tmp2, tmp3, u)

  if internaluse
    UDT(inv(tmp), 1 ./ d, tmp2')
  else
    UDT(inv(tmp), 1 ./ d, copy(tmp2'))
  end
end


"""
  inv_one_plus!(res, A::UDT, Bdagger::UDT) -> res

Stabilized calculation of [1 + UlDlTl(UrDrTr)^†]^(-1).
Writes the result into `res`.

See `udt_inv_one_plus` for preallocation options.
"""
function inv_one_plus!(res, A::UDT, Bdagger::UDT; kwargs...)
  F = udt_inv_one_plus(A, Bdagger; internaluse = true, kwargs...)
  rmul!(F.U, Diagonal(F.D))
  mul!(res, F.U, F.T)
  res
end


"""
  inv_one_plus(A::UDT, Bdagger::UDT) -> AbstractMatrix

Stabilized calculation of [1 + UlDlTl(UrDrTr)^†]^(-1).

See `udt_inv_one_plus` for preallocation options.
"""
function inv_one_plus(A::UDT, Bdagger::UDT; kwargs...)
  res = similar(A.U)
  inv_one_plus!(res, A, Bdagger; kwargs...)
  return res
end




"""
    udt_inv_sum(A::UDT, B::UDT) -> UDT

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Use one intermediate UDT decompositions.

Optional preallocations via keyword arguments:

  * `m2 = similar(A.U)`
"""
function udt_inv_sum(A::UDT, B::UDT; m2 = similar(A.U))
  Ua,Da,Ta = A
  Ub,Db,Tb = B

  m1 = Ta / Tb
  lmul!(Diagonal(Da), m1)

  mul!(m2, Ua', Ub)
  rmul!(m2, Diagonal(Db))

  u,d,t = udt!(m1 + m2)

  mul!(m1, Ua, u)
  mul!(m2, t, Tb)

  UDT(inv(m2), 1 ./ d, m1')
end


"""
    inv_sum!(res, A::UDT, B::UDT) -> res

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Use one intermediate UDT decompositions.

See `udt_inv_sum` for preallocation options.
"""
function inv_sum!(res, A::UDT, B::UDT; kwargs...)
  F = udt_inv_sum(A, B; kwargs...)
  rmul!(F.U, Diagonal(F.D))
  mul!(res, F.U, F.T)
  res
end


"""
    inv_sum(A::UDT, B::UDT) -> AbstractMatrix

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Use one intermediate UDT decompositions.

See `udt_inv_sum` for preallocation options.
"""
function inv_sum(A::UDT, B::UDT; kwargs...)
  res = similar(A.U)
  inv_sum!(res, A, B; kwargs...)
  res
end










########################################
#
#         Loh et. al. schemes
#
########################################


"""
    udt_inv_one_plus_loh(F::UDT) -> UDT

Stabilized calculation of [1 + UDT]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

Options for preallocation via keyword arguments:

  * `l = similar(F.U)`
  * `r = similar(F.U)`
  * `Dp = similar(F.D)`
  * `Dm = similar(F.D)`
"""
function udt_inv_one_plus_loh(F::UDT; r = similar(F.U),
                                      l = similar(F.U),
                                      Dp = similar(F.D),
                                      Dm = similar(F.D),
                                      internaluse = false)
  U,D,T = F

  Dp .= max.(D,1.)
  Dm .= min.(D,1.)

  Dp .\= 1
  Dpinv = Dp # renaming

  ldiv!(l, lu(T), Diagonal(Dpinv)) # Don't use lu!(T) because T is input

  mul!(r, U, Diagonal(Dm))
  r .+= l

  u, d, t = udt!(r)

  ldiv!(r, lu!(t), Diagonal(1 ./ d))
  mul!(l, r, u')

  lmul!(Diagonal(Dpinv), l)
  u, d, t = udt!(l)

  ldiv!(l, lu(T), u)

  if internaluse
    UDT(l, d, t)
  else
    UDT(copy(l), d, t)
  end
end


"""
  inv_one_plus_loh!(res, F::UDT) -> res

Stabilized calculation of [1 + UDT]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

Writes the result into `res`.

See `udt_inv_one_plus_loh` for preallocation options.
"""
function inv_one_plus_loh!(res, F::UDT; kwargs...)
  X = udt_inv_one_plus_loh(F; internaluse = true, kwargs...)
  rmul!(X.U, Diagonal(X.D))
  mul!(res, X.U, X.T)
  res
end


"""
  inv_one_plus_loh(F::UDT) -> AbstractMatrix

Stabilized calculation of [1 + UDT]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

See `udt_inv_one_plus_loh` for preallocation options.
"""
function inv_one_plus_loh(F::UDT; kwargs...)
  res = similar(F.U)
  inv_one_plus_loh!(res, F; kwargs...)
  res
end


"""
    udt_inv_sum_loh(A::UDT, B::UDT) -> UDT

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

Options for preallocations via keyword arguments:

  * `mat2 = similar(A.U)`
  * `Dap = similar(A.D)`
  * `Dam = similar(A.D)`
  * `Dbp = similar(A.D)`
  * `Dbm = similar(A.D)`

"""
function udt_inv_sum_loh(A::UDT, B::UDT; mat2 = similar(A.U),
                                         Dap = similar(A.D),
                                         Dam = similar(A.D),
                                         Dbp = similar(A.D),
                                         Dbm = similar(A.D),
                                         internaluse = false)
    Ua, Da, Ta = A
    Ub, Db, Tb = B

    d = length(Da)

    # separating scales larger and smaller than unity
    Dap .= max.(Da,1.)
    Dam .= min.(Da,1.)
    Dbp .= max.(Db,1.)
    Dbm .= min.(Db,1.)

    # mat1 = Dam * Vda * Vdb' / Dbp
    mat1 = Ta / Tb
    @inbounds for j in 1:d, k in 1:d
        mat1[j,k]=mat1[j,k] * Dam[j]/Dbp[k]
    end

    # mat2 = 1/(Dap) * Ua' * Ub * Dbm
    mul!(mat2, Ua', Ub)
    @inbounds for j in 1:d, k in 1:d
        mat2[j,k]=mat2[j,k] * Dbm[k]/Dap[j]
    end

    # mat1 = mat1 + mat2
    mat1 .+= mat2

    # decompose mat1: U, D, T
    U, D, T = udt!(mat1)

    # invert and combine inner part: mat1 = (U D T)^(-1)
    lmul!(Diagonal(D), T)
    ldiv!(mat1, lu!(T), U') # mat1 = T \ (U')

    # mat1 = 1/Dbp * mat1 /Dap
    @inbounds for j in 1:d, k in 1:d
        mat1[j,k]=mat1[j,k] / Dbp[j] / Dap[k]
    end

    #mat1 = U D T
    U, D, T = udt!(mat1)

    # U = Tb^(-1) * U , T = T * Ua'
    ldiv!(mat1, lu(Tb), U) # mat1 = Tb \ U
    mul!(mat2, T, Ua')

    if internaluse
      UDT(mat1, D, mat2)
    else
      UDT(mat1, D, copy(mat2))
    end
end


"""
    inv_sum_loh!(res, A::UDT, B::UDT) -> res

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

Writes the result into `res`.

See `udt_inv_sum_loh` for preallocation options.
"""
function inv_sum_loh!(res, A::UDT, B::UDT; kwargs...)
  F = udt_inv_sum_loh(A, B; internaluse = true, kwargs...)
  rmul!(F.U, Diagonal(F.D))
  mul!(res, F.U, F.T)
  res
end


"""
    inv_sum_loh(A::UDT, B::UDT) -> AbstractMatrix

Stabilized calculation of [UaDaTa + UbDbTb]^(-1):

  * Separate scales larger and smaller than unity
  * Use two intermediate UDT decompositions.

See `udt_inv_sum_loh` for preallocation options.
"""
function inv_sum_loh(A::UDT, B::UDT; kwargs...)
  res = similar(A.U)
  inv_sum_loh!(res, A, B; kwargs...)
  res
end
