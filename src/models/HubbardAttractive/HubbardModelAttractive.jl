const HubbardConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardConfs = Array{Int8, 3}
const HubbardDistribution = Int8[-1,1]
const HubbardGreens = Complex{Float64}

"""
Famous attractive (negative U) Hubbard model on a cubic lattice.
Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the density/charge channel,
such that HS-field is real.

    HubbardModelAttractive(; dims, L[, kwargs...])

Create an attractive Hubbard model on `dims`-dimensional cubic lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `mu::Float64=0.0`: chemical potential
 * `U::Float64=1.0`: interaction strength
 * `t::Float64=1.0`: hopping energy
"""
@with_kw_noshow mutable struct HubbardModelAttractive{C<:AbstractCubicLattice} <: Model
	# user mandatory
	dims::Int
	L::Int

    l::C = choose_lattice(HubbardModelAttractive, dims, L)
	flv::Int = 1

	mu::Float64 = 0.0
	lambda::Float64 = 1.0
	t::Float64 = 1.0
end

function choose_lattice(::Type{HubbardModelAttractive}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    HubbardModelAttractive(kwargs::Dict{String, Any})

Create an attractive Hubbard model with (keyword) parameters as specified in `kwargs` dict.
"""
HubbardModelAttractive(kwargs::Dict{String, Any}) = HubbardModelAttractive(; convert(Dict{Symbol,Any}, kwargs)...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelAttractive) = "$(model.dims)D attractive Hubbard model"
Base.show(io::IO, model::HubbardModelAttractive) = print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(model.l.sites) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelAttractive) = print(io, model)

# methods
"""
    energy(m::HubbardModelAttractive, hsfield::HubbardConf)

Calculate bosonic part of the energy for configuration `hsfield`.
"""
@inline function energy_boson(m::HubbardModelAttractive, hsfield::HubbardConf) # not needed for propose_local
    return m.lambda * sum(hsfield)
end

import Base.rand
"""
    rand(mc::DQMC, m::HubbardModelAttractive)

Draw random HS field configuration.
"""
@inline rand(mc::DQMC, m::HubbardModelAttractive) = rand(HubbardDistribution, m.l.sites, mc.p.slices)

"""
    conftype(::Type{DQMC}, m::HubbardModelAttractive)

Returns the type of a (Hubbard-Stratonovich field) configuration of the attractive Hubbard model.
"""
@inline conftype(::Type{DQMC}, m::HubbardModelAttractive) = HubbardConf

"""
    greenseltype(::Type{DQMC}, m::HubbardModelAttractive)

Returns the element type of the Green's function.
"""
@inline greenseltype(::Type{DQMC}, m::HubbardModelAttractive) = Complex{Float64}

"""
    propose_local(m::HubbardModelAttractive, i::Int, conf::HubbardConf, E_boson::Float64) -> detratio, delta_E_boson, delta

Propose a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
"""
@inline function propose_local(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, E_boson::Float64)
    # see for example dos Santos (2002)
    const greens = mc.s.greens

    delta_E_boson = -2. * m.lambda * conf[i, slice]
	gamma = exp(delta_E_boson) - 1
    detratio = (1 + gamma * (1 - greens[i,i]))^2 # squared because of two spin sectors.

    # if abs(imag(prob_fermion)) > 1e-6
    #     println("Did you expect a sign problem?", abs(imag(prob)))
    #     @printf "%.10e" abs(imag(prob))
    # end

    return detratio, delta_E_boson, gamma
end

"""
    accept_local(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)

Accept a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
Arguments `delta`, `detratio` and `delta_E_boson` correspond to output of `propose_local()`
for that flip.
"""
@inline function accept_local!(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, delta, detratio, delta_E_boson::Float64)
    const greens = mc.s.greens
    const gamma = delta_i

    u = -greens[:, i]
    u[i] += 1.
    # OPT: speed check, maybe @views/@inbounds
    greens .-= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * greens[i, :]))
    conf[i, slice] .*= -1.
    mc.energy_boson += delta_E_boson
    nothing
end


"""
    interaction_matrix_exp!(mc::DQMC, m::HubbardModelAttractive, result::Matrix, conf::HubbardConf, slice::Int, power::Float64=1.) -> nothing

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::HubbardModelAttractive,
            result::Matrix, conf::HubbardConf, slice::Int, power::Float64=1.)
    # const dtau = mc.p.delta_tau
    # V = - 1/dtau * m.lambda * conf[:,slice]
    # result = spdiagm(exp(- sign(power) * dtau * V))

    result .= spdiagm(exp.(sign(power) * m.lambda * conf[:,slice]))
    nothing
end

"""
	hopping_matrix(mc::DQMC, m::HubbardModelAttractive)

Calculates the hopping matrix \$ T_{i, j} \$ where \$ i, j \$ are
site indices.

Note that since we have a time reversal symmetry relating spin-up
to spin-down we only consider one spin sector (one flavor) for the attractive
Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, m::HubbardModelAttractive)
  const N = m.l.sites
  const neighs = m.l.neighs # row = up, right, down, left; col = siteidx

  T = diagm(fill(-m.mu, N))

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:N
      for nb in 1:size(neighs,1)
        trg = neighs[nb,src]
        T[trg,src] += -m.t
      end
    end
  end

  # const dtau = mc.p.delta_tau
  # l.hopping_matrix_exp = expm(-0.5 * dtau * T)
  # l.hopping_matrix_exp_inv = expm(0.5 * dtau * T)
  return T
end

include("observables.jl")
