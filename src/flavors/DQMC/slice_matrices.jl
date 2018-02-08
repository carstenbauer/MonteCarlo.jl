# We need all multiply_slice_matrix etc. functions (see checkerboard)
# Somehow dispatch based on mc.p.chkr wether to use checkerboard or usual slice matrix

# Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))
function slice_matrix(mc::DQMC, m::Model, slice::Int, power::Float64=1.) # direct calculation of effective slice matrix, i.e. no checkerboard
	interaction_matrix_exp!(mc, m, mc.s.eV, slice, power)
	const eT = mc.s.hopping_matrix_exp
	const eTinv = mc.s.hopping_matrix_exp_inv

	if power > 0
		return eT * eT * mc.s.eV
	else
		return expV * eTinv * eTinv
	end
end

# multiply_slice_matrix_left!(mc::DQMC, m::Model, slice::Int, M::AbstractMatrix)
# multiply_slice_matrix_inv_left!
# multiply_slice_matrix_right!
# multiply_slice_matrix_inv_right!
# multiply_daggered_slice_matrix_left!