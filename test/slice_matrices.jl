@testset "Slice Matrices" begin
    m = HubbardModelAttractive(dims=1, L=8)
    dqmc = DQMC(m, beta=5.0)
    eT = dqmc.s.hopping_matrix_exp
    eV = similar(eT)
    A = similar(eT)


    # No checkerboard
    for slice in rand(1:50, 2)
        MonteCarlo.interaction_matrix_exp!(dqmc, m, eV, dqmc.conf, slice, 1.)

        # MonteCarlo.slice_matrix
        @test MonteCarlo.slice_matrix(dqmc, m, slice, 1.) ≈ eT * eT * eV
        @test MonteCarlo.slice_matrix(dqmc, m, slice, -1.) ≈ inv(eT * eT * eV)
        # MonteCarlo.slice_matrix!(dqmc, m, slice, 1., A)
        # @test A ≈ eT * eT * eV
        # MonteCarlo.slice_matrix!(dqmc, m, slice, -1., A)
        # @test A ≈ inv(eT * eT * eV)

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

    for slice in rand(1:50, 2)
        MonteCarlo.interaction_matrix_exp!(dqmc, m, eV, dqmc.conf, slice, 1.)

        # MonteCarlo.slice_matrix
        @test maximum(abs.(
            MonteCarlo.slice_matrix(dqmc, m, slice, 1.) .- eT * eT * eV
        )) < 2dqmc.p.delta_tau
        @test maximum(abs.(
            MonteCarlo.slice_matrix(dqmc, m, slice, -1.) .- inv(eT * eT * eV)
        )) < 2dqmc.p.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, 1., A)
        @test maximum(abs.(A .- eT * eT * eV)) < 2dqmc.p.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, -1., A)
        @test maximum(abs.(A .- inv(eT * eT * eV))) < 2dqmc.p.delta_tau

        # MonteCarlo.multiply_slice_matrix...
        A = eT * eT * eV
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs.(input .- result)) < 2dqmc.p.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_right!(dqmc, m, slice, input)
        @test maximum(abs.(input .- result)) < 2dqmc.p.delta_tau

        A = inv(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_inv_left!(dqmc, m, slice, input)
        @test maximum(abs.(input .- result)) < 2dqmc.p.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_inv_right!(dqmc, m, slice, input)
        @test maximum(abs.(input .- result)) < 2dqmc.p.delta_tau

        A = adjoint(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_daggered_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs.(input .- result)) < 2dqmc.p.delta_tau
    end
end
