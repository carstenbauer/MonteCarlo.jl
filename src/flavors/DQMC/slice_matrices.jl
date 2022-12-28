@bm function multiply_slice_matrix_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # eT^2 eV M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, mc.stack.eV, M)
    vmul!(M, mc.stack.hopping_matrix_exp_squared, mc.stack.tmp1, mc.stack.tmp2)
    nothing
end

@bm function multiply_slice_matrix_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # M eT^2 eV
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, M, mc.stack.hopping_matrix_exp_squared, mc.stack.tmp2)
    vmul!(M, mc.stack.tmp1, mc.stack.eV)
    nothing
end

@bm function multiply_slice_matrix_inv_right!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # M * eV^-1 eT2^-1 
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, -1.0)
    vmul!(mc.stack.tmp1, M, mc.stack.eV)
    vmul!(M, mc.stack.tmp1, mc.stack.hopping_matrix_exp_inv_squared, mc.stack.tmp2)
    nothing
end

@bm function multiply_slice_matrix_inv_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # eV^-1 eT2^-1 M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, -1.0)
    vmul!(mc.stack.tmp1, mc.stack.hopping_matrix_exp_inv_squared, M, mc.stack.tmp2)
    vmul!(M, mc.stack.eV, mc.stack.tmp1)
    nothing
end

@bm function multiply_daggered_slice_matrix_left!(
        mc::DQMC, m::Model, slice::Int, M::AbstractMatrix, field = field(mc)
    )
    # adjoint(eT^2 eV) M = eV' eT2' M
    interaction_matrix_exp!(mc, m, field, mc.stack.eV, slice, 1.0)
    vmul!(mc.stack.tmp1, adjoint(mc.stack.hopping_matrix_exp_squared), M, mc.stack.tmp2)
    vmul!(M, adjoint(mc.stack.eV), mc.stack.tmp1)
    nothing
end