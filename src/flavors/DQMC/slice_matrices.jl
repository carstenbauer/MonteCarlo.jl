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
function slice_matrix!(mc::DQMC_CBFalse, m::Model, slice::Int,
					power::Float64=1., result = mc.s.U)
	eT = mc.s.hopping_matrix_exp
	eTinv = mc.s.hopping_matrix_exp_inv
	eV = mc.s.eV

	interaction_matrix_exp!(mc, m, eV, mc.conf, slice, power)

	if power > 0
		# eT * (eT * eV)
		mul!(mc.s.tmp, eT, eV)
		mul!(result, eT, mc.s.tmp)
	else
		# ev * (eTinv * eTinv)
		mul!(mc.s.tmp, eTinv, eTinv)
		mul!(result, eV, mc.s.tmp)
	end
	return result
end


function multiply_slice_matrix_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	slice_matrix!(mc, m, slice, 1.0, mc.s.U)
	mul!(mc.s.tmp, mc.s.U, M)
	M .= mc.s.tmp
	nothing
end
function multiply_slice_matrix_right!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	slice_matrix!(mc, m, slice, 1.0, mc.s.U)
	mul!(mc.s.tmp, M, mc.s.U)
	M .= mc.s.tmp
	nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	slice_matrix!(mc, m, slice, -1.0, mc.s.U)
	mul!(mc.s.tmp, M, mc.s.U)
	M .= mc.s.tmp
	nothing
end
function multiply_slice_matrix_inv_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	slice_matrix!(mc, m, slice, -1.0, mc.s.U)
	mul!(mc.s.tmp, mc.s.U, M)
	M .= mc.s.tmp
	nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_CBFalse, m::Model,
								slice::Int, M::AbstractMatrix)
	slice_matrix!(mc, m, slice, 1.0, mc.s.U)
	mul!(mc.s.tmp, adjoint(mc.s.U), M)
	M .= mc.s.tmp
	nothing
end


# CheckerboardTrue
const DQMC_CBTrue = DQMC{M, CheckerboardTrue} where M

function slice_matrix(mc::DQMC_CBTrue, m::Model, slice::Int,
					power::Float64=1.)
    M = Matrix{heltype(mc)}(I, m.flv*m.l.sites, m.flv*m.l.sites)
  	if power > 0
    	multiply_slice_matrix_left!(mc, m, slice, M)
  	else
    	multiply_slice_matrix_inv_left!(mc, m, slice, M)
  	end
  	return M
end
function slice_matrix!(mc::DQMC_CBTrue, m::Model, slice::Int,
					power::Float64=1., M = mc.s.U)
	copyto!(M, I)
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

	mul!(s.tmp, s.eV, M)
	M .= s.tmp
	mul!(s.tmp, s.chkr_mu, M)
  	M .= s.tmp

    @inbounds begin
    	for i in reverse(2:s.n_groups)
			mul!(s.tmp, s.chkr_hop_half[i], M)
			M .= s.tmp
    	end
		mul!(s.tmp, s.chkr_hop[1], M)
		M .= s.tmp
    	for i in 2:s.n_groups
			mul!(s.tmp, s.chkr_hop_half[i], M)
			M .= s.tmp
    	end
  	end
  	nothing
end
function multiply_slice_matrix_right!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
    s = mc.s
  	@inbounds begin
    	for i in reverse(2:s.n_groups)
			right_mul!(s.tmp, M, s.chkr_hop_half[i])
			M .= s.tmp
    	end
		right_mul!(s.tmp, M, s.chkr_hop[1])
		M .= s.tmp
    	for i in 2:s.n_groups
			right_mul!(s.tmp, M, s.chkr_hop_half[i])
			M .= s.tmp
    	end
  	end

  	interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, 1.)
	right_mul!(s.tmp, M, s.chkr_mu)
	M .= s.tmp
	mul!(s.tmp, M, s.eV)
	M .= s.tmp
  	nothing
end
function multiply_slice_matrix_inv_left!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
    s = mc.s
  	@inbounds begin
    	for i in reverse(2:s.n_groups)
			mul!(s.tmp, s.chkr_hop_half_inv[i], M)
			M .= s.tmp
    	end
		mul!(s.tmp, s.chkr_hop_inv[1], M)
		M .= s.tmp
    	for i in 2:s.n_groups
			mul!(s.tmp, s.chkr_hop_half_inv[i], M)
			M .= s.tmp
    	end
  	end

  	interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, -1.)
	mul!(s.tmp, s.chkr_mu_inv, M)
	M .= s.tmp
	mul!(s.tmp, s.eV, M)
	M .= s.tmp
  	nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
    s = mc.s
  	interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, -1.)
	mul!(s.tmp, M, s.eV)
	M .= s.tmp
	right_mul!(s.tmp, M, s.chkr_mu_inv)
	M .= s.tmp


  	@inbounds begin
    	for i in reverse(2:s.n_groups)
			right_mul!(s.tmp, M, s.chkr_hop_half_inv[i])
			M .= s.tmp
    	end
		right_mul!(s.tmp, M, s.chkr_hop_inv[1])
		M .= s.tmp
    	for i in 2:s.n_groups
			right_mul!(s.tmp, M, s.chkr_hop_half_inv[i])
			M .= s.tmp
    	end
  	end
  	nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    M::AbstractMatrix{T}) where T<:Number
    s = mc.s

  	@inbounds begin
    	for i in reverse(2:s.n_groups)
			mul!(s.tmp, s.chkr_hop_half_dagger[i], M)
			M .= s.tmp
    	end
		mul!(s.tmp, s.chkr_hop_dagger[1], M)
		M .= s.tmp
    	for i in 2:s.n_groups
			mul!(s.tmp, s.chkr_hop_half_dagger[i], M)
			M .= s.tmp
    	end
  	end

  	interaction_matrix_exp!(mc, m, s.eV, mc.conf, slice, 1.)
  	# s.eV == adjoint(s.eV) and s.chkr_mu == adjoint(s.chkr_mu)
	mul!(s.tmp, s.chkr_mu, M)
	M .= s.tmp
	mul!(s.tmp, s.eV, M)
	M .= s.tmp
  	nothing
end
