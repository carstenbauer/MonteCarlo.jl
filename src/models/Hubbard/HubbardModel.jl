const HubbardConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardConfs = Array{Int8, 3}
const HubbardDistribution = Int8[-1,1]
const HubbardGreens = Complex{Float64}

"""
Famous Hubbard model on a cubic lattice.

Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the density/charge channel.
"""
mutable struct HubbardModel{C<:CubicLattice} <: Model
	# mandatory
	L::Int
	dims::Int
	beta::Float64
	l::C

	# mandatory?
	flv::Int # flavors: GF matrix will have size flv*l.sites x flv*l.sites

	# model specific
	μ::Float64
	λ::Float64
	t::Float64
end

"""
    HubbardModel(dims::Int, L::Int, beta::Float64)
    HubbardModel(; dims::Int=2, L::Int=8, beta::Float64=1.0)

Create Hubbard model on `dims`-dimensional cubic lattice
with linear system size `L` and inverse temperature `beta`.
"""
function HubbardModel(dims::Int, L::Int, beta::Float64)
    if dims == 2
        return HubbardModel(L, 2, beta, SquareLattice(L))
    else
        error("Only `dims=2` supported for now.")
    end
end
HubbardModel(; dims::Int=2, L::Int=8, beta::Float64=1.0) = HubbardModel(dims, L, beta)

# methods
"""
    energy(m::HubbardModel, hsfield::HubbardConf)

Calculate energy of Hubbard hsfieldiguration `hsfield` for Hubbard model `m`.
"""
function energy(m::HubbardModel, hsfield::HubbardConf) # not needed for propose_local
    return m.λ * sum(hsfield)
end

import Base.rand
"""
    rand(m::HubbardModel)

Draw random HS field configuration.
"""
rand(m::HubbardModel) = rand(HubbardDistribution, m.l.L, m.l.L)

"""
    conftype(m::HubbardModel)

Returns the type of an Hubbard model configuration.
"""
conftype(m::HubbardModel) = HubbardConf

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