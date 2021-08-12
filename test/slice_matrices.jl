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
    m = HubbardModelAttractive(8, 1)
    dqmc = DQMC(m, beta=5.0)
    eT = dqmc.stack.hopping_matrix_exp
    eV = similar(dqmc.stack.eV)
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
        ) < 2dqmc.parameters.delta_tau
        @test maximum(abs,
            MonteCarlo.slice_matrix(dqmc, m, slice, -1.) .- inv(eT * eT * eV)
        ) < 2dqmc.parameters.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, 1., A)
        @test maximum(abs, A .- eT * eT * eV) < 2dqmc.parameters.delta_tau
        MonteCarlo.slice_matrix!(dqmc, m, slice, -1., A)
        @test maximum(abs, A .- inv(eT * eT * eV)) < 2dqmc.parameters.delta_tau

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



using MonteCarlo: vmul!, lvmul!, rvmul!, rdivp!, udt_AVX_pivot!, rvadd!, vsub!
using MonteCarlo: vmin!, vmininv!, vmax!, vmaxinv!, vinv!
using MonteCarlo: BlockDiagonal#, CMat64, CVec64, StructArray



@testset "Custom Linear Algebra " begin
    for type in (Float64, ComplexF64)
        @testset "avx multiplications ($type)" begin
            A = rand(type, 16, 16)
            B = rand(type, 16, 16)
            C = rand(type, 16, 16)
            atol = 100eps(Float64)
            type == ComplexF64 && (atol *= sqrt(2))

            vmul!(C, A, B)
            @test A * B ≈ C atol = atol

            vmul!(C, A, B')
            @test A * B' ≈ C atol = atol

            vmul!(C, A', B)
            @test A' * B ≈ C atol = atol

            vmul!(C, A', B')
            @test A' * B' ≈ C atol = atol

            D = Diagonal(rand(16))
            vmul!(C, A, D)
            @test A * D ≈ C atol = atol

            copyto!(C, A)
            rvmul!(C, D)
            @test A * D ≈ C atol = atol

            copyto!(C, A)
            lvmul!(D, C)
            @test D * A ≈ C atol = atol

            copyto!(C, A)
            rvadd!(C, D)
            @test A + D ≈ C atol = atol

            copyto!(C, A)
            rvadd!(C, B)
            @test A + B ≈ C atol = atol

            vsub!(C, A, I)
            @test A - I ≈ C atol = atol

            if type == Float64
                v = rand(16) .+ 0.5
                w = copy(v)
                
                vmin!(v, w)
                @test v ≈ min.(1.0, w) atol = atol

                vmininv!(v, w)
                @test v ≈ 1.0 ./ min.(1.0, w) atol = atol
                
                vmax!(v, w)
                @test v ≈ max.(1.0, w) atol = atol

                vmaxinv!(v, w)
                @test v ≈ 1.0 ./ max.(1.0, w) atol = atol

                v = copy(w)
                vinv!(w)
                @test w ≈ 1.0 ./ v atol = atol
            end
        end

        @testset "UDT transformations + rdivp! ($type)" begin
            U = Matrix{Float64}(undef, 16, 16)
            D = Vector{Float64}(undef, 16)
            T = rand(16, 16)
            X = copy(T)
            MonteCarlo.udt_AVX!(U, D, T)
            @test U * Diagonal(D) * T ≈ X

            U = Matrix{type}(undef, 16, 16)
            T = rand(type, 16, 16)
            X = copy(T)
            pivot = Vector{Int64}(undef, 16)
            tempv = Vector{type}(undef, 16)
            udt_AVX_pivot!(U, D, T)
            @test U * Diagonal(D) * T ≈ X

            copyto!(T, X)
            pivot = Vector{Int64}(undef, 16)
            tempv = Vector{type}(undef, 16)
            udt_AVX_pivot!(U, D, T, pivot, tempv, Val(false))
            # pivoting matrix
            P = zeros(length(pivot), length(pivot))
            for (i, j) in enumerate(pivot)
                P[i, j] = 1.0
            end
            @test U * Diagonal(D) * UpperTriangular(T) * P ≈ X

            u = copy(U)
            t = copy(T)
            tmp = similar(T)
            rdivp!(u, t, tmp, pivot)
            @test u ≈ U * P' / UpperTriangular(T)
        end
    end


    #=
    @testset "Complex StructArray" begin
        M1 = rand(ComplexF64, 8, 8)    
        C1 = StructArray(M1)
        M2 = rand(ComplexF64, 8, 8)    
        C2 = StructArray(M2)
        M3 = rand(ComplexF64, 8, 8)    
        C3 = StructArray(M3)

        @test C1 isa CMat64
        @test M1 == C1

        # Test avx multiplications
        vmul!(C1, C2, C3)
        vmul!(M1, M2, M3)
        @test M1 == C1

        vmul!(C1, C2, adjoint(C3))
        vmul!(M1, M2, adjoint(M3))
        @test M1 ≈ C1 atol=1e-14 # check

        vmul!(C1, adjoint(C2), C3)
        vmul!(M1, adjoint(M2), M3)
        @test M1 ≈ C1 atol=1e-14 #check

        D = Diagonal(rand(8))
        vmul!(C1, C2, D)
        vmul!(M1, M2, D)
        @test M1 == C1

        rvmul!(C1, D)
        rvmul!(M1, D)
        @test M1 == C1

        lvmul!(D, C1)
        lvmul!(D, M1)
        @test M1 == C1

        # Test UDT and rdivp!
        M2 = Matrix(C2)
        D = rand(8)
        pivot = Vector{Int64}(undef, 8)
        tempv = StructArray(Vector{ComplexF64}(undef, 8))
        @test tempv isa CVec64
        udt_AVX_pivot!(C1, D, C2, pivot, tempv, Val(false))
        P = zeros(length(pivot), length(pivot))
        for (i, j) in enumerate(pivot)
            P[i, j] = 1.0
        end
        @test Matrix(C1) * Diagonal(D) * UpperTriangular(Matrix(C2)) * P ≈ M2 #check

        M1 = Matrix(C1)
        M2 = Matrix(C2)
        rdivp!(C1, C2, C3, pivot)
        @test C1 ≈ M1 * P' / UpperTriangular(M2) # check
    end
    =#


    @testset "BlockDiagonal" begin
        b1 = rand(4, 4)
        b2 = rand(4, 4)
        atol = 100eps(Float64)

        B = BlockDiagonal(b1, b2)
        @test B isa BlockDiagonal{Float64, 2, Matrix{Float64}}
        
        # Check values/indexing
        for i in 1:4, j in 1:4
            @test B[i, j] == b1[i, j]
            @test B[4+i, 4+j] == b2[i, j]
            @test B[4+i, j] == 0.0
            @test B[i, 4+j] == 0.0
        end

        B1 = copy(B)
        M1 = Matrix(B1)
        @test M1 == B1

        B2 = BlockDiagonal(rand(4, 4), rand(4, 4))
        M2 = Matrix(B2)
        B3 = BlockDiagonal(rand(4, 4), rand(4, 4))
        M3 = Matrix(B3)

        # Test avx multiplications
        vmul!(B1, B2, B3)
        vmul!(M1, M2, M3)
        @test M1 ≈ B1 atol = atol

        vmul!(B1, B2, adjoint(B3))
        vmul!(M1, M2, adjoint(M3))
        @test M1 ≈ B1

        vmul!(B1, adjoint(B2), B3)
        vmul!(M1, adjoint(M2), M3)
        @test M1 ≈ B1 atol = atol

        D = Diagonal(rand(8))
        vmul!(B1, B2, D)
        vmul!(M1, M2, D)
        @test M1 ≈ B1 atol = atol

        rvmul!(B1, D)
        rvmul!(M1, D)
        @test M1 ≈ B1 atol = atol

        lvmul!(D, B1)
        lvmul!(D, M1)
        @test M1 ≈ B1 atol = atol

        rvadd!(B1, D)
        rvadd!(M1, D)
        @test M1 ≈ B1 atol = atol

        rvadd!(B1, B2)
        rvadd!(M1, M2)
        @test M1 ≈ B1 atol = atol

        vsub!(B1, B2, I)
        @test M2 - I ≈ B1 atol = atol

        x = rand()
        @test B2 * x ≈ M2 * x atol = atol

        @test log(B2) ≈ log(M2) atol = atol

        @test det(B2) ≈ det(M2) atol = atol

        transpose!(B1, B2)
        @test B1 ≈ M2' atol = atol

        vmul!(B1, B2', B3')
        @test B1 ≈ M2' * M3' atol = atol

        # Test UDT and rdivp!
        M2 = Matrix(B2)
        D = rand(8)
        pivot = Vector{Int64}(undef, 8)
        tempv = Vector{Float64}(undef, 8)
        udt_AVX_pivot!(B1, D, B2, pivot, tempv, Val(false))
        
        P = zeros(length(pivot), length(pivot))
        # pivot is per block, temporary make it global (in the sense of matrix indices)
        n = size(B1.blocks[1], 1); N = length(B.blocks)
        for i in 1:N; pivot[(i-1)*n+1 : i*n] .+= (i-1)*n; end
        for (i, j) in enumerate(pivot)
            P[i, j] = 1.0
        end
        for i in 1:N; pivot[(i-1)*n+1 : i*n] .-= (i-1)*n; end
        
        @test Matrix(B1) * Diagonal(D) * UpperTriangular(Matrix(B2)) * P ≈ M2

        M1 = Matrix(B1)
        M2 = Matrix(B2)
        rdivp!(B1, B2, B3, pivot)
        @test B1 ≈ M1 * P' / UpperTriangular(M2)
    end
end 
