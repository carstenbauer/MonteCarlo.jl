# We need all multiply_slice_matrix etc. functions (see checkerboard)
# Somehow dispatch based on mc.p.chkr wether to use checkerboard or usual slice matrix

# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix(mc::DQMC, slice::Int, power::Float64=1.) # direct calculation of effective slice matrix, i.e. no checkerboard
	const expV = interaction_matrix_exp(mc, slice, power)
	const expT = l.hopping_matrix_exp
	const expTinv = l.hopping_matrix_exp_inv

	if power > 0
		return expT * expT * expV
	else
		return expV * expTinv * expTinv
	end
end

# multiply_slice_matrix_left!(mc::DQMC, m::Model, slice::Int, M::AbstractMatrix)
# multiply_slice_matrix_inv_left!
# multiply_slice_matrix_right!
# multiply_slice_matrix_inv_right!
# multiply_daggered_slice_matrix_left!