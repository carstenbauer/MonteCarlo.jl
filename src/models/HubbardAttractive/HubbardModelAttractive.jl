const HubbardConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardDistribution = Int8[-1,1]

"""
Famous attractive (negative U) Hubbard model on a cubic lattice.
Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the density/charge channel,
such that HS-field is real.

    HubbardModelAttractive(; dims, L[, kwargs...])

Create an attractive Hubbard model on `dims`-dimensional cubic lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `mu::Float64=0.0`: chemical potential
 * `U::Float64=1.0`: onsite interaction strength, "Hubbard U"
 * `t::Float64=1.0`: hopping energy
"""
@with_kw_noshow mutable struct HubbardModelAttractive{C<:AbstractCubicLattice} <: Model
    # user mandatory
    dims::Int
    L::Int

    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0

    # non-user fields
    l::C = choose_lattice(HubbardModelAttractive, dims, L)
    flv::Int = 1
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
function HubbardModelAttractive(kwargs::Dict{String, Any})
    symbol_dict = Dict([Symbol(k) => v for (k, v) in kwargs])
    HubbardModelAttractive(; symbol_dict...)
end

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
@inline function energy_boson(m::HubbardModelAttractive, hsfield::HubbardConf)
  dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
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
@inline greenseltype(::Type{DQMC}, m::HubbardModelAttractive) = Float64

"""
    propose_local(m::HubbardModelAttractive, i::Int, conf::HubbardConf, E_boson::Float64) -> detratio, delta_E_boson, delta

Propose a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
"""
@inline function propose_local(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, E_boson::Float64)
    # see for example dos Santos (2002)
  greens = mc.s.greens
  dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))

    delta_E_boson = -2. * lambda * conf[i, slice]
    gamma = exp(delta_E_boson) - 1
    detratio = (1 + gamma * (1 - greens[i,i]))^2 # squared because of two spin sectors.

    return detratio, delta_E_boson, gamma
end

"""
    accept_local(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)

Accept a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
Arguments `delta`, `detratio` and `delta_E_boson` correspond to output of `propose_local()`
for that flip.
"""
@inline function accept_local!(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, delta, detratio, delta_E_boson::Float64)
  greens = mc.s.greens
  gamma = delta

    u = -greens[:, i]
    u[i] += 1.
    # OPT: speed check, maybe @views/@inbounds
    greens .-= kron(u * 1. /(1 + gamma * u[i]), transpose(gamma * greens[i, :]))
    conf[i, slice] *= -1
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
  dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    result .= spdiagm(0 => exp.(sign(power) * lambda * conf[:,slice]))
    nothing
end

"""
	hopping_matrix(mc::DQMC, m::HubbardModelAttractive)

Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

Note that since we have a time reversal symmetry relating spin-up
to spin-down we only consider one spin sector (one flavor) for the attractive
Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, m::HubbardModelAttractive)
  N = m.l.sites
  neighs = m.l.neighs # row = up, right, down, left; col = siteidx

  T = diagm(0 => fill(-m.mu, N))

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:N
      for nb in 1:size(neighs,1)
        trg = neighs[nb,src]
        T[trg,src] += -m.t
      end
    end
  end

  return T
end

include("observables.jl")
