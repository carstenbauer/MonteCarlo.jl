"""
    GreensMatrix{eltype, mattype}

`GreensMatrix` is a thin wrapper for `mattype` eturned by `greens(...)`, 
`greens!(...)` and related iterators. The elements `G[i, j]` represent 
`⟨cᵢ(k) cⱼ^†(l)⟩` where `k` and `l` are time slice indices contained in the 
type.

Related to this is the `P::Permuted` type, returned by `swapop(::GreensMatrix)`. 
It represents `P[i, j] = ⟨cᵢ^†(l) cⱼ(k)⟩ = δᵢⱼ δₖₗ - ⟨cⱼ(k) cᵢ^†(l)⟩`.

Note that `GreensMatrix` and `Permuted` allow access over the physical bounds
of the wrapped matrix. In these cases the matrix is conceptually copied on the 
diagonal and zeros are isnerted on the offdiagonal blocks. This is done to more
easily work with flavor symmetric matrices.
M 0 … 0
0 ⋱ 0 ⋮
⋮ 0 ⋱ 0
0 … 0 M
"""
struct GreensMatrix{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    k::Int
    l::Int
    val::M
end

"""
see GreensMatrix
"""
struct Permuted{T, GT <: GreensMatrix{T}} <: AbstractMatrix{T}
    x::GT
end

# function Base.show(io::IO, x::GreensMatrix)
#     println(io, "G[i, j] = cᵢ(", x.k, ") cⱼ^†(", x.l, ") =")
#     show(io, x.val)
# end
# function Base.show(io::IO, x::Permuted{<: GreensMatrix})
#     println(io, "G'[i, j] = cᵢ^†(", x.l, ") cⱼ(", x.k, ") =")
#     show(io, I[x.x.k, x.x.l] * I - transpose(x.x.val))
# end

Base.size(x::GreensMatrix) = size(x.val)
Base.size(x::Permuted) = size(x.x.val)

function Base.getindex(M::GreensMatrix{T}, i, j) where {T}
    N = size(M.val, 1)
    xsection, x = divrem(i-1, N)
    ysection, y = divrem(j-1, N)
    @inbounds return (xsection == ysection) * M.val[x+1, y+1]
end
function Base.getindex(M::Permuted, i, j)
    N = size(M.x.val, 1)
    xsection, x = divrem(i-1, N)
    ysection, y = divrem(j-1, N)
    @inbounds return (xsection == ysection) * (I[M.x.k, M.x.l] * I[x, y] - M.x.val[y+1, x+1])
end

"""
    swapop(G::GreensMatrix)

Permute the operator order of a `GreensMatrix` from ⟨cᵢ(k) cⱼ^†(l)⟩ to 
⟨cᵢ^†(l) cⱼ(k)⟩.
"""
swapop(x::GreensMatrix) = Permuted(x)
swapop(x::Permuted) = x.x
Base.copy(x::GreensMatrix) = GreensMatrix(x.k, x.l, copy(x.val))
function Base.:(==)(a::GreensMatrix, b::GreensMatrix)
    a.k == b.k && a.l == b.l && a.val == b.val
end
function Base.isapprox(a::GreensMatrix, b::GreensMatrix; kwargs...)
    a.k == b.k && a.l == b.l && isapprox(a.val, b.val; kwargs...)
end

"""
    greens(mc::DQMC)

Obtain the current equal-time Green's function, i.e. the fermionic expectation
value of `Gᵢⱼ = ⟨cᵢcⱼ^†⟩`. The indices relate to sites and flavors, but the
exact meanign depends on the model. For the attractive Hubbard model
`G[i, j] = ⟨c_{i, ↑} c_{j, ↑}^†⟩ = ⟨c_{i, ↓} c_{j, ↓}^†⟩` due to symmetry.

Internally, `mc.stack.greens` is an effective Green's function. This method
transforms it to the actual Green's function by multiplying hopping matrix
exponentials from left and right.
"""
@bm greens(mc::DQMC) = GreensMatrix(0, 0, copy(_greens!(mc)))

"""
    greens!(mc::DQMC[; output=mc.stack.greens_temp, input=mc.stack.greens, temp=mc.stack.curr_U])

Inplace version of `greens`.
"""
@bm function greens!(mc::DQMC; output=mc.stack.greens_temp, input=mc.stack.greens, temp=mc.stack.curr_U, temp2 = mc.stack.tmp2)
    # TODO rework measurements to work well with StructArrays and remove this    
    if isdefined(mc.stack, :complex_greens_temp)
        _greens!(mc, output, input, temp, temp2)
        copyto!(mc.stack.complex_greens_temp, output)
        GreensMatrix(0, 0, mc.stack.complex_greens_temp)
    else
        GreensMatrix(0, 0, _greens!(mc, output, input, temp, temp2))
    end
end



# Note that we use the Trotter decomposition 
# `exp(-Δτ(V + T)) = exp(-0.5 Δτ T) exp(-Δτ V) exp(-0.5 Δτ T)`
# alongside the cyclic property of determinants 
# `1 / det(I + eT eV1 eT ... eT eVN eT) = 1 / det(I * eT eT eV1 ... eT eT eVN)`
# to increase analytic accuracy. When calculating the greens function we need to
# reverse the cyclic permutation, which happens here.

function _greens!(
        mc::DQMC, target::AbstractMatrix = mc.stack.greens_temp, 
        source::AbstractMatrix = mc.stack.greens, 
        temp::AbstractMatrix = mc.stack.curr_U,
        temp2::AbstractMatrix = mc.stack.tmp2
    )
    eThalfminus = mc.stack.hopping_matrix_exp
    eThalfplus = mc.stack.hopping_matrix_exp_inv
    vmul!(temp, source, eThalfminus, temp2)
    vmul!(target, eThalfplus, temp, temp2)
    return target
end



# Same stuff with a specified time slice.

"""
    greens(mc::DQMC, l::Integer)

Calculates the equal-time greens function at a given slice index `l`, i.e. 
`G_{ij}(l, l) = G_{ij}(l⋅Δτ, l⋅Δτ) = ⟨cᵢ(l⋅Δτ)cⱼ(l⋅Δτ)^†⟩`.

Note: This internally overwrites the stack variables `Ul`, `Dl`, `Tl`, `Ur`, 
`Dr`, `Tr`, `curr_U`, `tmp1` and `tmp2`. All of those can be used as temporary
or output variables here, however keep in mind that other results may be 
invalidated. (I.e. `G = greens!(mc)` would get overwritten.)
"""
@bm greens(mc::DQMC, slice::Integer) = GreensMatrix(slice, slice, copy(_greens!(mc, slice)))

"""
    greens!(mc::DQMC, l::Integer[; output=mc.stack.greens_temp, temp1=mc.stack.tmp1, temp2=mc.stack.tmp2])

Inplace version of `greens!(mc, l)`
"""
@bm function greens!(
        mc::DQMC, slice::Integer; 
        output=mc.stack.greens_temp, temp1 = mc.stack.tmp1, temp2 = mc.stack.tmp2
    ) 
    GreensMatrix(slice, slice, _greens!(mc, slice, output, temp1, temp2))
end


function _greens!(
        mc::DQMC, slice::Integer, output::AbstractMatrix = mc.stack.greens_temp, 
        temp1::AbstractMatrix = mc.stack.tmp1, temp2::AbstractMatrix = mc.stack.tmp2
    )
    calculate_greens(mc, slice, temp1)
    _greens!(mc, output, temp1, temp2)
end