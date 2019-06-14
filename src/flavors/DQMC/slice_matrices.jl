# CheckerboardFalse
const DQMC_CBFalse = DQMC{M, CheckerboardFalse} where M

"""
	slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64=1.)

Direct calculation of effective slice matrix, i.e. no checkerboard.
Calculates `Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))`.
"""
function slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int,
					power::Float64=1.)
	eT = mc.s.hopping_matrix_exp
	eTinv = mc.s.hopping_matrix_exp_inv
	eV = mc.s.eV

	interaction_matrix_exp!(mc, m, eV, mc.conf, slice, power)

	if power > 0
		return eT * eT * eV
	else
		return eV * eTinv * eTinv
	end
end
function multiply_slice_matrix_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	M .= slice_matrix(mc, m, slice, 1.) * M
	nothing
end
function multiply_slice_matrix_right!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	M .= M * slice_matrix(mc, m, slice, 1.)
	nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	M .= M * slice_matrix(mc, m, slice, -1.)
	nothing
end
function multiply_slice_matrix_inv_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	M .= slice_matrix(mc, m, slice, -1.) * M
	nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	M .= ctranspose(slice_matrix(mc, m, slice, 1.)) * M
	nothing
end


# CheckerboardTrue
const DQMC_CBTrue = DQMC{M, CheckerboardTrue} where M

function slice_matrix(mc::DQMC_CBTrue, m::Model, slice::Int,
					power::Float64=1.)
  M = eye(heltype(mc), m.flv*m.l.sites)
  if power > 0
    multiply_slice_matrix_left!(mc, m, slice, M)
  else
    multiply_slice_matrix_inv_left!(mc, m, slice, M)
  end
  return M
end
function multiply_slice_matrix_left!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
  s = mc.s
  interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, 1.)

  M[:] = s.eV * M
  M[:] = s.chkr_mu * M

  @inbounds @views begin
    for i in reverse(2:s.n_groups)
      M[:] = s.chkr_hop_half[i] * M
    end
    M[:] = s.chkr_hop[1] * M
    for i in 2:s.n_groups
      M[:] = s.chkr_hop_half[i] * M
    end
  end
  nothing
end
function multiply_slice_matrix_right!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
  s = mc.s
  @inbounds @views begin
    for i in reverse(2:s.n_groups)
      M[:] = M * s.chkr_hop_half[i]
    end
    M[:] = M * s.chkr_hop[1]
    for i in 2:s.n_groups
      M[:] = M * s.chkr_hop_half[i]
    end
  end

  interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, 1.)
  M[:] = M * s.chkr_mu
  M[:] = M * s.eV
  nothing
end
function multiply_slice_matrix_inv_left!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
  s = mc.s
  @inbounds @views begin
    for i in reverse(2:s.n_groups)
      M[:] = s.chkr_hop_half_inv[i] * M
    end
    M[:] = s.chkr_hop_inv[1] * M
    for i in 2:s.n_groups
      M[:] = s.chkr_hop_half_inv[i] * M
    end
  end

  interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, -1.)
  M[:] = s.chkr_mu_inv * M
  M[:] = s.eV * M
  nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
  s = mc.s
  interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, -1.)
  M[:] = M * s.eV
  M[:] = M * s.chkr_mu_inv

  @inbounds @views begin
    for i in reverse(2:s.n_groups)
      M[:] = M * s.chkr_hop_half_inv[i]
    end
    M[:] = M * s.chkr_hop_inv[1]
    for i in 2:s.n_groups
      M[:] = M * s.chkr_hop_half_inv[i]
    end
  end
  nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
  s = mc.s
  @inbounds @views begin
    for i in reverse(2:s.n_groups)
      M[:] = s.chkr_hop_half_dagger[i] * M
    end
    M[:] = s.chkr_hop_dagger[1] * M
    for i in 2:s.n_groups
      M[:] = s.chkr_hop_half_dagger[i] * M
    end
  end

  interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, 1.)
  # s.eV == ctranspose(s.eV) and s.chkr_mu == ctranspose(s.chkr_mu)
  M[:] = s.chkr_mu * M
  M[:] = s.eV * M
  nothing
end
