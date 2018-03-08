# CheckerboardFalse
const DQMC_CBFalse = DQMC{M, CheckerboardFalse} where M

"""
	slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64=1.)

Direct calculation of effective slice matrix, i.e. no checkerboard.
Calculates `Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))`.
"""
function slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int,
					power::Float64=1.)
	const eT = mc.s.hopping_matrix_exp
	const eTinv = mc.s.hopping_matrix_exp_inv
	const eV = mc.s.eV

	interaction_matrix_exp!(mc, m, eV, slice, power)

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
	M .= ctranspose(slice_matrix(mc, m, slice, -1.)) * M
	nothing
end


# CheckerboardTrue
const DQMC_CBTrue = DQMC{M, CheckerboardTrue} where M
