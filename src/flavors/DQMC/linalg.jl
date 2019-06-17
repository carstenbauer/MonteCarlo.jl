#### SVD, i.e. UDV decomposition
function decompose_udv!(A::Matrix{T}) where T<:Number

  Base.LinAlg.LAPACK.gesvd!('A','A',A)

  # F = svdfact!(A) # based on Base.LinAlg.LAPACK.gesdd!('A',A)
  # return F[:U], F[:S], F[:Vt]
end

function decompose_udv(A::Matrix{T}) where T<:Number
  X = copy(A)
  decompose_udv!(X)
  return X
end


#### QR, i.e. UDT decomposition
function decompose_udt(A::AbstractMatrix{C}) where C<:Number
  Q, R, p = qr(A, Val(true))
  @views p[p] = collect(1:length(p))
  # D = abs.(real(diag(triu(R))))
  D = abs.(real(diag(R)))
  T = (spdiagm(0 => 1. ./ D) * R)[:, p]
  return Q, D, T
end


#### Other
function expm_diag!(A::Matrix{T}) where T<:Number
  F = eigfact!(A)
  return F[:vectors] * spdiagm(0 => exp(F[:values])) * ctranspose(F[:vectors])
end

function lu_det(M)
    L, U, p = lu(M)
    return prod(diag(L)) * prod(diag(U))
end
