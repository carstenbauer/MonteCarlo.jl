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
@inline greenseltype(::Type{DQMC}, m::HubbardModel) = Float64


include("HubbardModelAttractive.jl")
include("HubbardModelRepulsive.jl")