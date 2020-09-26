using SparseArrays

@testset "right_mul!" begin

    # NOTE
    # currently julia/sparsearrays does not implement a
    # mul!(C::StridedMatrix, X::StridedMatrix, A::SparseMatrixCSC)
    # instead it falls back to some generic matrix multiplciation in LinearAlgebra
    #
    # If this throws:
    # - check if mul!() is now specialized for the above type signature
    #   - if it is, remove right_mul! in helpers.jl & change right_muL! to mul! in
    #     slice_matrices.jl
    #   - otherwise, update the type signature
    #
    # see:
    # Julia PR: https://github.com/JuliaLang/julia/pull/24045
    # Julia Issue: https://github.com/JuliaLang/julia/issues/29956#issuecomment-440867024
    # MC PR: https://github.com/crstnbr/MonteCarlo.jl/pull/9
    # discourse: https://discourse.julialang.org/t/asymmetric-speed-of-in-place-sparse-dense-matrix-product/10256

    mul_dense_sparse_is_missing = !occursin("SparseArrays", string(which(mul!, (Matrix, Matrix, SparseMatrixCSC)).file))
    @test mul_dense_sparse_is_missing

    # check that right_mul! is correct
    for Lx in (4, 8)
        for Ly in (4, 8)
            M = rand(Lx, Ly)
            S = sparse(rand(Ly, Lx))
            X = rand(Lx, Lx)
            @test M * S == MonteCarlo.mul!(X, M, S)
        end
    end
end

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
        @test maximum(abs,
            MonteCarlo.slice_matrix(dqmc, m, slice, 1.) .- eT * eT * eV
        ) < 2dqmc.p.delta_tau
        @test maximum(abs,
            MonteCarlo.slice_matrix(dqmc, m, slice, -1.) .- inv(eT * eT * eV)
        ) < 2dqmc.p.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, 1., A)
        @test maximum(abs, A .- eT * eT * eV) < 2dqmc.p.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, -1., A)
        @test maximum(abs, A .- inv(eT * eT * eV)) < 2dqmc.p.delta_tau

        # MonteCarlo.multiply_slice_matrix...
        A = eT * eT * eV
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.p.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_right!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.p.delta_tau

        A = inv(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_slice_matrix_inv_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.p.delta_tau
        input = rand(size(A)...)
        result = input * A
        MonteCarlo.multiply_slice_matrix_inv_right!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.p.delta_tau

        A = adjoint(eT * eT * eV)
        input = rand(size(A)...)
        result = A * input
        MonteCarlo.multiply_daggered_slice_matrix_left!(dqmc, m, slice, input)
        @test maximum(abs, input .- result) < 2dqmc.p.delta_tau
    end
end


@testset "Custom Linear Algebra" begin
    @testset "avx multiplications" begin
        A = rand(16, 16)
        B = rand(16, 16)
        C = rand(16, 16)
        
        MonteCarlo.vmul!(C, A, B)
        @test A * B ≈ C
        
        MonteCarlo.vmul!(C, A, B')
        @test A * B' ≈ C
        
        MonteCarlo.vmul!(C, A', B)
        @test A' * B ≈ C
        
        D = Diagonal(rand(16))
        MonteCarlo.vmul!(C, A, D)
        @test A * D ≈ C
        
        copyto!(C, A)
        MonteCarlo.rvmul!(C, D)
        @test A * D ≈ C

        copyto!(C, A)
        MonteCarlo.lvmul!(D, C)
        @test D * A ≈ C
    end

    @testset "UDT transformations + rdivp!" begin
        U = Matrix{Float64}(undef, 16, 16)
        D = Vector{Float64}(undef, 16)
        T = rand(16, 16)
        X = copy(T)
        MonteCarlo.udt_AVX!(U, D, T)
        @test U * Diagonal(D) * T ≈ X

        copyto!(T, X)
        pivot = Vector{Int64}(undef, 16)
        tempv = Vector{Float64}(undef, 16)
        MonteCarlo.udt_AVX_pivot!(U, D, T)
        @test U * Diagonal(D) * T ≈ X

        copyto!(T, X)
        pivot = Vector{Int64}(undef, 16)
        tempv = Vector{Float64}(undef, 16)
        MonteCarlo.udt_AVX_pivot!(U, D, T, pivot, tempv, Val(false))
        # pivoting matrix
        P = zeros(length(pivot), length(pivot))
        for (i, j) in enumerate(pivot)
            P[i, j] = 1.0
        end
        @test U * Diagonal(D) * UpperTriangular(T) * P ≈ X

        u = copy(U)
        t = copy(T)
        tmp = similar(T)
        MonteCarlo.rdivp!(u, t, tmp, pivot)
        @test u ≈ U * P' / UpperTriangular(T)
    end
end