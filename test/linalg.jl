let

    # Complex and Real Matrix mults, BlockDiagonal, UDT
    for type in (Float64, ComplexF64)
        M = rand(type, 8, 8)
        E = exp(M)
        T = MonteCarlo.taylor_exp(M)
        @test all(abs.(T .- E) .< 10_000eps.(abs.(E)))

        @testset "avx multiplications ($type)" begin
            A = rand(type, 8, 8)
            B = rand(type, 8, 8)
            C = rand(type, 8, 8)
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

            D = Diagonal(rand(8))
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
                v = rand(8) .+ 0.5
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
            for X in (X = rand(8, 8), kron(rand(8), rand(8)))
                U = Matrix{Float64}(undef, 8, 8)
                D = Vector{Float64}(undef, 8)
                T = copy(X)
                MonteCarlo.udt_AVX!(U, D, T)
                @test U * Diagonal(D) * T ≈ X
            end

            for X in (X = rand(type, 8, 8), kron(rand(type, 8), rand(type, 8)))
                U = Matrix{type}(undef, 8, 8)
                T = copy(X)
                pivot = Vector{Int64}(undef, 8)
                tempv = Vector{type}(undef, 8)
                udt_AVX_pivot!(U, D, T)
                @test U * Diagonal(D) * T ≈ X

                copyto!(T, X)
                pivot = Vector{Int64}(undef, 8)
                tempv = Vector{type}(undef, 8)
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



        @testset "BlockDiagonal ($type)" begin
            # setindex!
            B = BlockDiagonal(zeros(4, 4), zeros(4, 4))
            for i in 1:8, j in 1:8
                if div(i, 5) == div(j, 5) # cause 5 / 5 = 1 is where we need to jump
                    B[i, j] = 8i+j
                    @test B[i, j] == 8i+j
                else
                    @test_throws BoundsError B[i, j] = 8i+j
                end
            end

            # typing
            b1 = rand(type, 4, 4)
            b2 = rand(type, 4, 4)
            atol = 100eps(Float64)
        
            B = BlockDiagonal(b1, b2)
            @test B isa BlockDiagonal{type, 2, Matrix{type}}

            # getindex
            for i in 1:4, j in 1:4
                @test B[i, j] == b1[i, j]
                @test B[4+i, 4+j] == b2[i, j]
                @test B[4+i, j] == 0.0
                @test B[i, 4+j] == 0.0
            end
        
            # Matrix()
            B1 = copy(B)
            M1 = Matrix(B1)
            @test M1 == B1
        
            B2 = BlockDiagonal(rand(type, 4, 4), rand(type, 4, 4))
            M2 = Matrix(B2)
            B3 = BlockDiagonal(rand(type, 4, 4), rand(type, 4, 4))
            M3 = Matrix(B3)
        
            # taylor exp
            E = exp(Matrix(B))
            T = MonteCarlo.taylor_exp(B)
            @test all(abs.(T .- E) .< 10_000eps.(abs.(E)))

            # Test (avx) multiplications
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
        
            # @test log(B2) ≈ log(M2) atol = atol
        
            # @test det(B2) ≈ det(M2) atol = atol
        
            transpose!(B1, B2)
            @test B1 ≈ transpose(M2) atol = atol
        
            vmul!(B1, B2', B3')
            @test B1 ≈ M2' * M3' atol = atol
        
            # Test UDT and rdivp!
            M2 = Matrix(B2)
            D = rand(8)
            pivot = Vector{Int64}(undef, 8)
            tempv = Vector{type}(undef, 8)
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


        
    @testset "Complex StructArray" begin
        M1 = rand(ComplexF64, 8, 8)    
        C1 = StructArray(M1)
        M2 = rand(ComplexF64, 8, 8)    
        C2 = StructArray(M2)
        M3 = rand(ComplexF64, 8, 8)    
        C3 = StructArray(M3)

        # taylor exp
        E = exp(Matrix(M1))
        T = MonteCarlo.taylor_exp(C1)
        @test all(abs.(T .- E) .< 10_000eps.(abs.(E)))

        @test C1 isa CMat64
        @test M1 == C1

        # Test avx multiplications (adjoint)
        vmul!(C1, C2, C3)
        vmul!(M1, M2, M3)
        @test M1 ≈ C1

        vmul!(C1, C2, adjoint(C3))
        vmul!(M1, M2, adjoint(M3))
        @test M1 ≈ C1

        vmul!(C1, adjoint(C2), C3)
        vmul!(M1, adjoint(M2), M3)
        @test M1 ≈ C1

        vmul!(C1, adjoint(C2), adjoint(C3))
        vmul!(M1, adjoint(M2), adjoint(M3))
        @test M1 ≈ C1

        # Diagonal
        D = Diagonal(rand(8))
        vmul!(C1, C2, D)
        vmul!(M1, M2, D)
        @test M1 == C1

        DC = Diagonal(rand(ComplexF64, 8))
        DCSA = Diagonal(StructArray(DC.diag))
        vmul!(C1, C2, DCSA)
        vmul!(M1, M2, DC)
        @test M1 ≈ C1

        rvmul!(C1, D)
        rvmul!(M1, D)
        @test M1 ≈ C1

        @test_throws ErrorException rvmul!(C1, DCSA)

        lvmul!(D, C1)
        lvmul!(D, M1)
        @test M1 ≈ C1

        @test_throws ErrorException lvmul!(DCSA, C1)

        copyto!(M1, M2)
        copyto!(C1, C2)
        rvadd!(C1, D)
        rvadd!(M1, D)
        @test C1 ≈ M1

        copyto!(M1, M2)
        copyto!(C1, C2)
        rvadd!(C1, C3)
        rvadd!(M1, M3)
        @test C1 ≈ M1

        vsub!(C1, C2, I)
        vsub!(M1, M2, I)
        @test C1 ≈ M1

        # Test UDT backend

        # indmax
        i1, sqn1 = MonteCarlo.indmaxcolumn(C2)
        i2 = LinearAlgebra.indmaxcolumn(M2)
        sqn2 = norm(M2[:, i2])^2
        @test i1 == i2
        @test sqn1 ≈ sqn2

        # reflector
        x1 = MonteCarlo.reflector!(C2, norm(C2[:, 1])^2, 1, size(C2, 1), C1)
        x2 = LinearAlgebra.reflector!(view(M2, :, 1))
        @test x1 ≈ x2
        @test C2 ≈ M2

        # reflectorApply
        C2 = StructArray(M2)
        @views LinearAlgebra.reflectorApply!(M2[:, 1], norm(M2[:, 1]), M2[:, 2:end])
        MonteCarlo.reflectorApply!(norm(C2[:, 1]), C2, 1, size(C2, 1))
        @test M2 ≈ C2

        # Test UDT's, rdivp!

        M2 = rand(ComplexF64, 8, 8)
        C2 = StructArray(M2)
        D = rand(Float64, 8)
        pivot = Vector{Int64}(undef, 8)
        tempv = Vector{ComplexF64}(undef, 8)
        udt_AVX_pivot!(C1, D, C2, pivot, tempv, Val(false))
        P = zeros(length(pivot), length(pivot))
        for (i, j) in enumerate(pivot)
            P[i, j] = 1.0
        end
        @test Matrix(C1) * Diagonal(D) * UpperTriangular(Matrix(C2)) * P ≈ M2

        # rdivp!
        M1 = Matrix(C1)
        M2 = Matrix(C2)
        rdivp!(C1, C2, C3, pivot)
        @test C1 ≈ M1 * P' / UpperTriangular(M2)

        C2 = StructArray(M2)
        udt_AVX_pivot!(C1, D, C2, pivot, tempv, Val(true))
        @test Matrix(C1) * Diagonal(D) * Matrix(C2) ≈ M2

    end
end