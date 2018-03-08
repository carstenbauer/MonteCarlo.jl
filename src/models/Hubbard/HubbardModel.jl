const HubbardConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardConfs = Array{Int8, 3}
const HubbardDistribution = Int8[-1,1]
const HubbardGreens = Complex{Float64}

"""
Famous Hubbard model on a cubic lattice.
Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the density/charge channel.

    HubbardModel(; dims, L[, kwargs...])

Create Hubbard model on `dims`-dimensional cubic lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `mu::Float64=0.0`: chemical potential
 * `U::Float64=1.0`: interaction strength
 * `t::Float64=1.0`: hopping energy
"""
@with_kw_noshow mutable struct HubbardModel{C<:AbstractCubicLattice} <: Model
	# mandatory
	dims::Int
    l::C = choose_lattice(HubbardModel, dims, L)
	L::Int

	# mandatory?
	flv::Int = 2 # flavors: GF matrix will have size flv*l.sites x flv*l.sites

	# model specific
	mu::Float64 = 0.0
	lambda::Float64 = 1.0
	t::Float64 = 1.0
end

function choose_lattice(::Type{HubbardModel}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    HubbardModel(kwargs::Dict{String, Any})

Create Hubbard model with (keyword) parameters as specified in `kwargs` dict.
"""
HubbardModel(kwargs::Dict{String, Any}) = HubbardModel(; convert(Dict{Symbol,Any}, kwargs)...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModel) = "$(model.dims)D-Hubbard model"
Base.show(io::IO, model::HubbardModel) = print(io, "$(model.dims)D-Hubbard model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HubbardModel) = print(io, model)

# methods
"""
    energy(m::HubbardModel, hsfield::HubbardConf)

Calculate energy of Hubbard hsfieldiguration `hsfield` for Hubbard model `m`.
"""
function energy(m::HubbardModel, hsfield::HubbardConf) # not needed for propose_local
    return m.lambda * sum(hsfield)
end

import Base.rand
"""
    rand(mc::DQMC, m::HubbardModel)

Draw random HS field configuration.
"""
rand(mc::DQMC, m::HubbardModel) = rand(HubbardDistribution, m.l.sites, mc.p.slices)

"""
    conftype(::Type{DQMC}, m::HubbardModel)

Returns the type of a (Hubbard-Stratonovich field) configuration of the Hubbard model.
"""
conftype(::Type{DQMC}, m::HubbardModel) = HubbardConf

"""
    greenseltype(::Type{DQMC}, m::HubbardModel)

Returns the element type of the Green's function.
"""
greenseltype(::Type{DQMC}, m::HubbardModel) = Complex{Float64}

"""
    propose_local(m::HubbardModel, i::Int, conf::HubbardConf, E::Float64) -> delta_E, delta_i

Propose a local HS field flip at site `i` of current configuration `conf`
with energy `E`. Returns `(delta_E, nothing)`.
"""
@inline function propose_local(m::HubbardModel, i::Int, slice::Int, conf::HubbardConf, E::Float64)
	gamma = exp(-1. * 2 * p.hsfield[i, s.current_slice] * p.lambda) - 1
    prob = (1 + gamma * (1 - s.greens[i,i]))^2 / (gamma + 1)

    # if abs(imag(prob)) > 1e-6
    #     println("Did you expect a sign problem?", abs(imag(prob)))
    #     @printf "%.10e" abs(imag(prob))
    # end

    return -log(abs(prob)), nothing
end

"""
    accept_local(m::HubbardModel, i::Int, conf::HubbardConf, E::Float64, delta_i, delta_E::Float64)

Accept a local HS field flip at site `i` of current configuration `conf`
with energy `E`. Arguments `delta_i` and `delta_E` correspond to output of `propose_local()`
for that flip.
"""
@inline function accept_local!(m::HubbardModel, i::Int, slice::Int, conf::HubbardConf, E::Float64, delta_i, delta_E::Float64)
    u = -s.greens[:, i]
    u[i] += 1.
    s.greens -= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * s.greens[i, :]))
    conf[i, s.current_slice] *= -1.
    nothing
end

# include("matrix_exponentials.jl")
include("observables.jl")
