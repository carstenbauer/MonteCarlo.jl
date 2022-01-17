# CheckerboardFalse
const DQMC_CBFalse = DQMC{M, CheckerboardFalse} where M

"""
    slice_matrix(mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64=1.)

Direct calculation of effective slice matrix, i.e. no checkerboard.
Calculates `Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))`.
"""
@bm function slice_matrix(
        mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64 = 1.0, field = field(mc)
    )
    eT2 = mc.stack.hopping_matrix_exp_squared
    eTinv2 = mc.stack.hopping_matrix_exp_inv_squared
    eV = mc.stack.eV

    interaction_matrix_exp!(mc, m, field, eV, slice, power)

    if power > 0
        return eT2 * eV
    else
        return eV * eTinv2
    end
end
@bm function slice_matrix!(
        mc::DQMC_CBFalse, m::Model, slice::Int, power::Float64 = 1.0, 
        result::AbstractArray = mc.stack.tmp2, field = field(mc)
    )
    eT2 = mc.stack.hopping_matrix_exp_squared
    eTinv2 = mc.stack.hopping_matrix_exp_inv_squared
    eV = mc.stack.eV

    interaction_matrix_exp!(mc, m, field, eV, slice, power)

    if power > 0
        # eT * (eT * eV)
        vmul!(result, eT2, eV)
    else
        # ev * (eTinv * eTinv)
        vmul!(result, eV, eTinv2)
    end
    return result
end


@bm function multiply_slice_matrix_left!(
        mc::DQMC_CBFalse, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_right!(
        mc::DQMC_CBFalse, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_right!(
        mc::DQMC_CBFalse, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_left!(
        mc::DQMC_CBFalse, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_daggered_slice_matrix_left!(
        mc::DQMC_CBFalse, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, adjoint(mc.stack.tmp2), M)
    M .= mc.stack.tmp1
    nothing
end


################################################################################
### CheckerboardTrue
################################################################################


const DQMC_CBTrue = DQMC{M, CheckerboardTrue} where M

@bm function slice_matrix(mc::DQMC_CBTrue, m::Model, slice::Int,
                    power::Float64=1.)
    N = length(lattice(m)) * nflavors(field(mc))
    M = Matrix{heltype(mc)}(I, N, N)
    if power > 0
        multiply_slice_matrix_left!(mc, m, slice, M)
    else
        multiply_slice_matrix_inv_left!(mc, m, slice, M)
    end
    return M
end
@bm function slice_matrix!(mc::DQMC_CBTrue, m::Model, slice::Int,
                    power::Float64=1., M = mc.stack.U)
    copyto!(M, I)
    if power > 0
        multiply_slice_matrix_left!(mc, m, slice, M)
    else
        multiply_slice_matrix_inv_left!(mc, m, slice, M)
    end
    return M
end

@bm function multiply_slice_matrix_left!(
        mc::DQMC_CBTrue, m::Model, slice::Int, M::AbstractMatrix{T}, field = field(mc)
    ) where T<:Number
    s = mc.stack
    interaction_matrix_exp!(mc, m, field, s.eV, slice, 1.)

    mul!(s.tmp1, s.eV, M)
    M .= s.tmp1
    mul!(s.tmp1, s.chkr_mu, M)
    M .= s.tmp1

    @inbounds begin
        for i in reverse(2:s.n_groups)
            mul!(s.tmp1, s.chkr_hop_half[i], M)
            M .= s.tmp1
        end
        mul!(s.tmp1, s.chkr_hop[1], M)
        M .= s.tmp1
        for i in 2:s.n_groups
            mul!(s.tmp1, s.chkr_hop_half[i], M)
            M .= s.tmp1
        end
    end
    nothing
end
@bm function multiply_slice_matrix_right!(
        mc::DQMC_CBTrue, m::Model, slice::Int, M::AbstractMatrix{T}, field = field(mc)
    ) where T<:Number
    s = mc.stack
    @inbounds begin
        for i in reverse(2:s.n_groups)
            mul!(s.tmp1, M, s.chkr_hop_half[i])
            M .= s.tmp1
        end
        mul!(s.tmp1, M, s.chkr_hop[1])
        M .= s.tmp1
        for i in 2:s.n_groups
            mul!(s.tmp1, M, s.chkr_hop_half[i])
            M .= s.tmp1
        end
    end

    interaction_matrix_exp!(mc, m, field, s.eV, slice, 1.)
    mul!(s.tmp1, M, s.chkr_mu)
    M .= s.tmp1
    mul!(s.tmp1, M, s.eV)
    M .= s.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_left!(
        mc::DQMC_CBTrue, m::Model, slice::Int, M::AbstractMatrix{T}, field = field(mc)
    ) where T<:Number
    s = mc.stack
    @inbounds begin
        for i in reverse(2:s.n_groups)
            mul!(s.tmp1, s.chkr_hop_half_inv[i], M)
            M .= s.tmp1
        end
        mul!(s.tmp1, s.chkr_hop_inv[1], M)
        M .= s.tmp1
        for i in 2:s.n_groups
            mul!(s.tmp1, s.chkr_hop_half_inv[i], M)
            M .= s.tmp1
        end
    end

    interaction_matrix_exp!(mc, m, field, s.eV, slice, -1.)
    mul!(s.tmp1, s.chkr_mu_inv, M)
    M .= s.tmp1
    mul!(s.tmp1, s.eV, M)
    M .= s.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_right!(
        mc::DQMC_CBTrue, m::Model, slice::Int, M::AbstractMatrix{T}, field = field(mc)
    ) where T<:Number
    s = mc.stack
    interaction_matrix_exp!(mc, m, field, s.eV, slice, -1.)
    mul!(s.tmp1, M, s.eV)
    M .= s.tmp1
    mul!(s.tmp1, M, s.chkr_mu_inv)
    M .= s.tmp1


    @inbounds begin
        for i in reverse(2:s.n_groups)
            mul!(s.tmp1, M, s.chkr_hop_half_inv[i])
            M .= s.tmp1
        end
        mul!(s.tmp1, M, s.chkr_hop_inv[1])
        M .= s.tmp1
        for i in 2:s.n_groups
            mul!(s.tmp1, M, s.chkr_hop_half_inv[i])
            M .= s.tmp1
        end
    end
    nothing
end
@bm function multiply_daggered_slice_matrix_left!(
        mc::DQMC_CBTrue, m::Model, slice::Int, M::AbstractMatrix{T}, field = field(mc)
    ) where T<:Number
    s = mc.stack

    @inbounds begin
        for i in reverse(2:s.n_groups)
            mul!(s.tmp1, s.chkr_hop_half_dagger[i], M)
            M .= s.tmp1
        end
        mul!(s.tmp1, s.chkr_hop_dagger[1], M)
        M .= s.tmp1
        for i in 2:s.n_groups
            mul!(s.tmp1, s.chkr_hop_half_dagger[i], M)
            M .= s.tmp1
        end
    end

    interaction_matrix_exp!(mc, m, field, s.eV, slice, 1.)
    # s.eV == adjoint(s.eV) and s.chkr_mu == adjoint(s.chkr_mu)
    mul!(s.tmp1, s.chkr_mu, M)
    M .= s.tmp1
    mul!(s.tmp1, s.eV, M)
    M .= s.tmp1
    nothing
end
