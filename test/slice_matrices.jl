using SparseArrays

@testset "Slice Matrices" begin
    m = HubbardModel(8, 1)
    dqmc = DQMC(m, beta=5.0)
    MonteCarlo.init!(dqmc)
    eT = dqmc.stack.hopping_matrix_exp
    eV = similar(dqmc.stack.eV)
    A = similar(eT)


    # No checkerboard
    for slice in rand(1:50, 2)
        MonteCarlo.interaction_matrix_exp!(dqmc, m, dqmc.field, eV, slice, 1.)

        # MonteCarlo.multiply_slice_matrix...
        A = eT * eT * eV
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_left!(dqmc, m, slice, input)
        @test input ≈ result
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_right!(dqmc, m, slice, input)
        @test input ≈ result

        A = inv(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_inv_left!(dqmc, m, slice, input)
        @test input ≈ result
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_inv_right!(dqmc, m, slice, input)
        @test input ≈ result

        A = adjoint(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_daggered_slice_matrix_left!(dqmc, m, slice, input)
        @test input ≈ result
    end



    # Checkerboard
    dqmc = DQMC(m, beta=5.0, checkerboard=true)
    MonteCarlo.init!(dqmc)

    for slice in rand(1:50, 2)
        MonteCarlo.interaction_matrix_exp!(dqmc, m, dqmc.field, eV, slice, 1.)

        # MonteCarlo.multiply_slice_matrix...
        A = eT * eT * eV
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.parameters.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_right!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.parameters.delta_tau

        A = inv(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_inv_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.parameters.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_inv_right!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.parameters.delta_tau

        A = adjoint(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_daggered_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.parameters.delta_tau
    end
end