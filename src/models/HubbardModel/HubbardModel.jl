# This file includes stuff that both Hubbard models use

# conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardConf = Array{Int8, 2} 
const HubbardDistribution = (Int8(-1), Int8(1))

abstract type HubbardModel <: Model end

"""
    HubbardModel([args..., ]; U[, kwargs...])

Creates an attractive or repulsive Hubbard model.
"""
function HubbardModel(args...; U, kwargs...)
    if U > 0.0
        HubbardModelRepulsive(args..., U = U; kwargs...)
    else
        HubbardModelAttractive(args..., U = -U; kwargs...)
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

interaction_matrix_type(::Type{DQMC}, ::HubbardModel) = Diagonal{Float64, Vector{Float64}}

function init_interaction_matrix(m::HubbardModel)
    N = length(lattice(m))
    flv = nflavors(m)
    Diagonal(zeros(Float64, N*flv))
end

# deprecate this? maybe add nstates instead to avoid accessing model.flv?
@inline nflavors(m::HubbardModel) = m.flv
@inline lattice(m::HubbardModel) = m.l

# implement `DQMC` interface: mandatory
@inline function Base.rand(::Type{DQMC}, m::HubbardModel, nslices::Int)
    rand(HubbardDistribution, length(m.l), nslices)
end

# implement DQMC interface: optional
# Green's function is real for the repulsive Hubbard model.
@inline greenseltype(::Type{DQMC}, m::HubbardModel) = Float64


# See configurations.jl - compression of configurations
compress(::DQMC, ::HubbardModel, c) = BitArray(c .== 1)
function decompress(mc::DQMC{M, CB, CT}, ::HubbardModel, c) where {M, CB, CT}
    CT(2c .- 1)
end

include("HubbardModelAttractive.jl")
include("HubbardModelRepulsive.jl")