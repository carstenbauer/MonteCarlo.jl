# interactionm = exp(- power delta_tau V(slice)), with power = +- 1.
@inline function interaction_matrix_exp!(mc::DQMC, m::HubbardModelAttractive, result::Matrix, slice::Int, power::Float64=1.)
  const dtau = mc.p.delta_tau
  # return spdiagm(exp(sign(power) * p.lambda * p.hsfield[:,slice]))
  return - 1/dtau * m.lambda * mc.conf[:,slice]
end

"""
	hopping_matrix(mc::DQMC, m::HubbardModelAttractive)

Calculates the hopping matrix \$ T_{i, j} \$ where \$ i, j \$ are
site indices.

Note that since we have a time reversal symmetry relating spin-up
to spin-down we only consider one spin sector (one flavor) for the attractive
Hubbard model in the DQMC simulation.
"""
function hopping_matrix(mc::DQMC, m::HubbardModelAttractive)
  const N = m.l.sites
  const neighs = m.l.neighs # row = up, right, down, left; col = siteidx
  const dtau = mc.p.delta_tau

  T = diagm(fill(-m.mu, N)) #TODO: Sign of mu?

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:N
      for nb in 1:size(neighs,1)
        trg = neighs[nb,src]
        T[trg,src] += -m.t
      end
    end
  end

  # l.hopping_matrix_exp = expm(-0.5 * dtau * T)
  # l.hopping_matrix_exp_inv = expm(0.5 * dtau * T)
  return T
end
