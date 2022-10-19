"""
    slice_matrix(mc::DQMC, m::Model, slice::Int, power::Float64=1.)

Direct calculation of effective slice matrix, i.e. no checkerboard.
Calculates `Beff(slice) = exp(−1/2∆tauT)exp(−1/2∆tauT)exp(−∆tauV(slice))`.
"""
@bm function slice_matrix(
        mc::DQMC, m::Model, slice::Int, power::Float64 = 1.0, field = field(mc)
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
        mc::DQMC, m::Model, slice::Int, power::Float64 = 1.0, 
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
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, M, mc.stack.tmp2)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_slice_matrix_inv_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, -1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, mc.stack.tmp2, M)
    M .= mc.stack.tmp1
    nothing
end
@bm function multiply_daggered_slice_matrix_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    slice_matrix!(mc, m, slice, 1.0, mc.stack.tmp2, field)
    vmul!(mc.stack.tmp1, adjoint(mc.stack.tmp2), M)
    M .= mc.stack.tmp1
    nothing
end