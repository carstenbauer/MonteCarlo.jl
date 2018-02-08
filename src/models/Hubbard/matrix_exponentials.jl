# interactionm = exp(- power delta_tau V(slice)), with power = +- 1.
@inline function interaction_matrix_exp!(mc::DQMC, m::HubbardModel, result::Matrix, slice::Int, power::Float64=1.)
  # return spdiagm(exp(sign(power) * p.lambda * p.hsfield[:,slice]))
  return - 1/p.delta_tau * p.lambda * p.hsfield[:,slice]

  # TODO needs delta_tau. How to provide it. Maybe Monte Carlo object as argument everywhere?
end

function hopping_matrix(m::HubbardModel)
  const N = m.l.sites
  const neighs = m.l.neighs # row = up, right, down, left; col = siteidx

  T = diagm(fill(-m.μ, N)) #TODO: Sign of mu?

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:N
      for nb in 1:4
        trg = neighs[nb,src]
        T[trg,src] += -m.t
      end
    end
  end

  # l.hopping_matrix_exp = expm(-0.5 * p.delta_tau * T)
  # l.hopping_matrix_exp_inv = expm(0.5 * p.delta_tau * T)
  return T
end