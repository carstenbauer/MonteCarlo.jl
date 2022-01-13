# This file includes stuff that both Hubbard models use
abstract type HubbardModel <: Model end

"""
    HubbardModel([args..., ]; U[, kwargs...])

Creates an attractive or repulsive Hubbard model. Assumes the interaction to be
given by -U ∑ᵢ (nᵢ↑ - 0.5) (nᵢ↓ - 0.5), i.e. U > 0 is the attractive Hubbard 
model.
"""
function HubbardModel(args...; U, kwargs...)
    if U < 0.0
        HubbardModelRepulsive(args..., U = U; kwargs...)
    else
        HubbardModelAttractive(args..., U = U; kwargs...)
    end
end


function choose_lattice(::Type{<: HubbardModel}, dims, L)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

# implement DQMC interface: optional
@inline lattice(m::HubbardModel) = m.l
nflavors(::HubbardModel) = 1
# @inline greenseltype(::Type{DQMC}, m::HubbardModel) = Float64

hopping_eltype(model::HubbardModel) = typeof(model.t)
function hopping_matrix_type(field::AbstractField, model::HubbardModel)
    flv = max(nflavors(field), nflavors(model))
    T = hopping_eltype(model)
    MT = matrix_type(T)
    if flv == 1
        return MT
    else
        return BlockDiagonal{T, flv, MT}
    end
end

function greens_matrix_type(f::AbstractHirschField{T}, m::HubbardModel) where {T}
    if max(nflavors(f), nflavors(m)) == 1
        return matrix_type(T)
    else
        return BlockDiagonal{T, 2, matrix_type(T)}
    end
end
function greens_matrix_type(f::AbstractGHQField{T}, m::HubbardModel) where {T}
    if max(nflavors(f), nflavors(m)) == 1
        return matrix_type(T)
    else
        return BlockDiagonal{T, 2, matrix_type(T)}
    end
end

function intE_kernel(mc, model::HubbardModel, G::GreensMatrix, ::Val{1})
    # ⟨U (n↑ - 1/2)(n↓ - 1/2)⟩ = ... 
    # = U [G↑↑ G↓↓ - G↓↑ G↑↓ - 0.5 G↑↑ - 0.5 G↓↓ + G↑↓ + 0.25]
    # = U [(G↑↑ - 1/2)(G↓↓ - 1/2) + G↑↓(1 + G↑↓)]
    # with up-up = down-down and up-down = 0
    - model.U * sum((diag(G.val) .- 0.5).^2)
end
# Technically this only applies to BlockDiagonal
function intE_kernel(mc, model::HubbardModel, G::GreensMatrix, ::Val{2})
    - model.U * sum((diag(G.val.blocks[1]) .- 0.5) .* (diag(G.val.blocks[2]) .- 0.5))
end

include("HubbardModelAttractive.jl")
include("HubbardModelRepulsive.jl")