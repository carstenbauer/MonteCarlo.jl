@testset "range index search" begin
    m = HubbardModel(2, 2)
    mc = DQMC(m, beta = 2.3, safe_mult = 10, delta_tau = 0.1)
    MonteCarlo.init!(mc)
    
    @test MonteCarlo._find_range_with_value(mc, -81273) == 0
    @test MonteCarlo._find_range_with_value(mc, 0) == 0
    for i in 1:23
        idx = MonteCarlo._find_range_with_value(mc, i)
        @test i in mc.stack.ranges[idx]
    end
    @test MonteCarlo._find_range_with_value(mc, 24) == length(mc.stack.ranges)+1
    @test MonteCarlo._find_range_with_value(mc, 1239874) == length(mc.stack.ranges)+1
end

# let to keep this out of the global scope
let
    m = HubbardModel(6, 1);
    dqmc = DQMC(m; beta=15.0, safe_mult=5)
    MonteCarlo.init!(dqmc)
    MonteCarlo.initialize_stack(dqmc, dqmc.ut_stack)
    MonteCarlo.build_stack(dqmc, dqmc.stack)

    @testset "lazy build_stack forward/backward" begin
        for i in 2:length(dqmc.ut_stack.forward_u_stack)
            dqmc.ut_stack.forward_u_stack[i]  .= 0.0
            dqmc.ut_stack.forward_d_stack[i]  .= 0.0
            dqmc.ut_stack.forward_t_stack[i]  .= 0.0
        end
        for i in 1:length(dqmc.ut_stack.backward_d_stack)-1
            dqmc.ut_stack.backward_u_stack[i] .= 0.0
            dqmc.ut_stack.backward_d_stack[i] .= 0.0
            dqmc.ut_stack.backward_t_stack[i] .= 0.0
        end

        for upto in (4, 6)
            MonteCarlo.lazy_build_forward!(dqmc, dqmc.ut_stack, upto)
            for i in 1:upto
                @test dqmc.stack.u_stack[i] ≈ dqmc.ut_stack.forward_u_stack[i]
                @test dqmc.stack.d_stack[i] ≈ dqmc.ut_stack.forward_d_stack[i]
                @test dqmc.stack.t_stack[i] ≈ dqmc.ut_stack.forward_t_stack[i]
            end
            for i in upto+1:length(dqmc.ut_stack.forward_u_stack)
                @test all(dqmc.ut_stack.forward_u_stack[i] .== 0)
                @test all(dqmc.ut_stack.forward_d_stack[i] .== 0)
                @test all(dqmc.ut_stack.forward_t_stack[i] .== 0)
            end
        end

        while dqmc.stack.direction == -1
            MonteCarlo.propagate(dqmc)
        end

        for downto in (8, 6)
            MonteCarlo.lazy_build_backward!(dqmc, dqmc.ut_stack, downto)
            for i in length(dqmc.ut_stack.backward_d_stack):-1:downto
                @test dqmc.stack.u_stack[i] ≈ dqmc.ut_stack.backward_u_stack[i]
                @test dqmc.stack.d_stack[i] ≈ dqmc.ut_stack.backward_d_stack[i]
                @test dqmc.stack.t_stack[i] ≈ dqmc.ut_stack.backward_t_stack[i]
            end
            for i in downto-1:-1:1
                @test all(dqmc.ut_stack.backward_u_stack[i] .== 0)
                @test all(dqmc.ut_stack.backward_d_stack[i] .== 0)
                @test all(dqmc.ut_stack.backward_t_stack[i] .== 0)
            end
        end
    end

    @testset "build_stack forward/backward" begin
        MonteCarlo.build_stack(dqmc, dqmc.stack)
        MonteCarlo.build_stack(dqmc, dqmc.ut_stack)

        # test B(τ, 1) / B_l1 stacks
        @test dqmc.stack.u_stack ≈ dqmc.ut_stack.forward_u_stack
        @test dqmc.stack.d_stack ≈ dqmc.ut_stack.forward_d_stack
        @test dqmc.stack.t_stack ≈ dqmc.ut_stack.forward_t_stack

        # generate B(β, τ) / B_Nl stacks
        while dqmc.stack.direction == -1
            MonteCarlo.propagate(dqmc)
        end

        # test B(β, τ) / B_Nl stacks
        # Note: dqmc.stack doesn't generate the full stack here
        @test dqmc.stack.u_stack[2:end] ≈ dqmc.ut_stack.backward_u_stack[2:end]
        @test dqmc.stack.d_stack[2:end] ≈ dqmc.ut_stack.backward_d_stack[2:end]
        @test dqmc.stack.t_stack[2:end] ≈ dqmc.ut_stack.backward_t_stack[2:end]
    end

    while !(MonteCarlo.current_slice(dqmc) == 1 && dqmc.stack.direction == -1)
        MonteCarlo.propagate(dqmc)
    end

    @testset "equal time Greens function" begin
        # Check equal time greens functions from equal time and unequal time
        # calculation against each other
        for slice in 0:MonteCarlo.nslices(dqmc)
            G1 = deepcopy(MonteCarlo.calculate_greens(dqmc, slice))
            G2 = deepcopy(MonteCarlo.calculate_greens(dqmc, slice, slice))
            @test maximum(abs.(G1 .- G2)) < 1e-14
        end

        # Check G(t, 0) + G(0, beta - t) = 0
        for slice in 0:MonteCarlo.nslices(dqmc)-1
            G1 = MonteCarlo.greens(dqmc, slice, 0).val
            G2 = MonteCarlo.greens(dqmc, slice, MonteCarlo.nslices(dqmc)).val
            @test G1 ≈ -G2 atol = 1e-13
        end
    end


    # Check Iterators
    # For some reason Gkl/Gk0 errors jump in the last 10 steps
    # when using UnequalTimeStack

    # As in G(τ = Δτ * k, 0)
    # Note: G(0, l) = -G(M-l, 0)
    Gk0s = [deepcopy(MonteCarlo.greens(dqmc, slice, 0).val) for slice in 0:MonteCarlo.nslices(dqmc)]

    @testset "GreensIterator" begin
        # Calculated from UnequalTimeStack (high precision)
        it = MonteCarlo.GreensIterator(dqmc, :, 0, dqmc.parameters.safe_mult)
        for (i, G) in enumerate(it)
            @test maximum(abs.(G.val .- Gk0s[i])) < 1e-14
        end

        # Calculated from mc.stack.greens using UDT decompositions (lower precision)
        it = MonteCarlo.GreensIterator(dqmc, :, 0, 4dqmc.parameters.safe_mult)
        for (i, G) in enumerate(it)
            @test maximum(abs.(G.val .- Gk0s[i])) < 1e-11
        end
    end

    Gkks = map(0:MonteCarlo.nslices(dqmc)) do slice
        g = MonteCarlo.calculate_greens(dqmc, slice)
        deepcopy(MonteCarlo._greens!(dqmc, dqmc.stack.greens_temp, g))
    end
    G0ks = [deepcopy(MonteCarlo.greens(dqmc, 0, slice).val) for slice in 0:MonteCarlo.nslices(dqmc)]
    MonteCarlo.calculate_greens(dqmc, 0) # restore mc.stack.greens

    @testset "CombinedGreensIterator" begin
        # high precision
        it = MonteCarlo.CombinedGreensIterator(dqmc,  start = 0, 
            stop = MonteCarlo.nslices(dqmc), recalculate = dqmc.parameters.safe_mult
        )
        @test it isa CombinedGreensIterator
        @test it.recalculate == dqmc.parameters.safe_mult
        @test it.start == 0
        @test it.stop == MonteCarlo.nslices(dqmc)
        @test length(it) == it.stop - it.start + 1

        iter = MonteCarlo.init(dqmc, it)
        @test iter isa MonteCarlo._CombinedGreensIterator
        @test iter.mc == dqmc
        @test iter.spec == it
        @test length(iter) == length(it)

        for (i, (G0k, Gk0, Gkk)) in enumerate(iter)
            @test maximum(abs.(Gk0.val .- Gk0s[i])) < 1e-14
            @test maximum(abs.(G0k.val .- G0ks[i])) < 1e-14
            @test maximum(abs.(Gkk.val .- Gkks[i])) < 1e-14
        end

        # low precision
        it = MonteCarlo.CombinedGreensIterator(dqmc, recalculate = 4dqmc.parameters.safe_mult)
        for (i, (G0k, Gk0, Gkk)) in enumerate(MonteCarlo.init(dqmc, it))
            @test maximum(abs.(Gk0.val .- Gk0s[i])) < 1e-10
            @test maximum(abs.(G0k.val .- G0ks[i])) < 1e-10
            @test maximum(abs.(Gkk.val .- Gkks[i])) < 1e-10
        end
    end
end

using MonteCarlo

@testset "BigFloat comparisons" begin
    # We want at least 5 safe_mult blocks, so that the forward, backward and 
    # inverse B chains can use precomputed blocks.
    # | block | block | block | block | block |
    #             ↑               ↑
    #             k               l

    old_precision = precision(BigFloat)
    setprecision(BigFloat, 128)

    m = HubbardModel(6, 1);
    mc = DQMC(m; beta=5.0, safe_mult=10)
    MonteCarlo.init!(mc)
    MonteCarlo.initialize_stack(mc, mc.ut_stack)
    MonteCarlo.build_stack(mc, mc.stack)
    MonteCarlo.build_stack(mc, mc.ut_stack)

    @testset "stack values" begin
        # Build stack values w/o QR decompositions with BigFloat
        GMT = Matrix{BigFloat}
        N = length(lattice(mc))
        flv = MonteCarlo.nflavors(mc)
        M = mc.stack.n_elements
        
        forward_stack = [GMT(undef, flv*N, flv*N) for _ in 1:M]
        backward_stack = [GMT(undef, flv*N, flv*N) for _ in 1:M]
        inv_stack = [GMT(undef, flv*N, flv*N) for _ in 1:M-1]

        copyto!(forward_stack[1], I)
        copyto!(backward_stack[end], I)
    
        eT2 = BigFloat.(Matrix(mc.stack.hopping_matrix_exp_squared))
        eT2inv = BigFloat.(Matrix(mc.stack.hopping_matrix_exp_inv_squared))

        # forward
        @inbounds for idx in 1:length(mc.stack.ranges)
            U = BigFloat.(forward_stack[idx])
            for slice in mc.stack.ranges[idx]
                MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, 1.0)
                U = eT2 * BigFloat.(mc.stack.eV) * U
            end
            forward_stack[idx+1] .= U
        end

        # backward
        @inbounds for idx in length(mc.stack.ranges):-1:1
            U = BigFloat.(backward_stack[idx + 1])
            for slice in reverse(mc.stack.ranges[idx])
                MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, 1.0)
                U = adjoint(eT2 * BigFloat.(mc.stack.eV)) * U
            end
            backward_stack[idx] .= U
        end

        # inverse
        @inbounds for idx in 1:length(mc.stack.ranges)
            U = GMT(I, flv*N, flv*N)
            for slice in reverse(mc.stack.ranges[idx])
                MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, -1.0)
                U = BigFloat.(mc.stack.eV) * eT2inv * U
            end
            copyto!(inv_stack[idx], U)
        end

        # compare
        s = mc.ut_stack
        for l in 1:M
            # From general testing: rel. error < 1e-13
            # From testing here: < 1e-14
            @test check(forward_stack[l], s.forward_u_stack[l] * Diagonal(s.forward_d_stack[l]) * s.forward_t_stack[l], 0, 1e-14)
            @test check(backward_stack[l], s.backward_u_stack[l] * Diagonal(s.backward_d_stack[l]) * s.backward_t_stack[l], 0, 1e-14)
            if l > M
                @test check(inv_stack[l], s.inv_u_stack[l] * Diagonal(s.inv_d_stack[l]) * s.inv_t_stack[l], 0, 1e-14)
            end
        end
    end


    @testset "Time displaced Greens function" begin
        function calc_greens(mc, slice1, slice2)
            # stack = [0, Δτ, 2Δτ, ..., β] = [0, safe_mult, 2safe_mult, ... N]
            # @assert slice1 ≥ slice2
            GMT = Matrix{BigFloat}
            N = length(lattice(mc))
            flv = MonteCarlo.nflavors(mc)
            M = mc.parameters.slices
            
            eT2 = BigFloat.(Matrix(mc.stack.hopping_matrix_exp_squared))
            eT2inv = BigFloat.(Matrix(mc.stack.hopping_matrix_exp_inv_squared))
        
            if slice1 ≥ slice2
                # k ≥ l or slice1 ≥ slice2
                # B_{l+1}^-1 B_{l+2}^-1 ⋯ B_{k-1}^-1 B_k^-1
                inv_B = GMT(I, flv*N, flv*N)
                for slice in slice1:-1:slice2+1
                    MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, -1.0)
                    inv_B = BigFloat.(mc.stack.eV) * eT2inv * inv_B
                end
                
                # B(slice2, 1) = Ul Dl Tl
                forward_B = GMT(I, flv*N, flv*N)
                for slice in 1:slice2
                    MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, 1.0)
                    forward_B = eT2 * BigFloat.(mc.stack.eV) * forward_B
                end
                
                # B(N, slice1) = (Ur Dr Tr)^† = Tr^† Dr^† Ur^†
                backward_B = GMT(I, flv*N, flv*N)
                for slice in M:-1:slice1+1
                    MonteCarlo.interaction_matrix_exp!(mc, mc.model, mc.field, mc.stack.eV, slice, 1.0)
                    backward_B = adjoint(eT2 * BigFloat.(mc.stack.eV)) * backward_B
                end
            else
                error("Not yet implemented")
            end
            
            # [inv + forward * dagger(backward)]^-1
            return inv(inv_B + forward_B * adjoint(backward_B))
        end

        # Not sure if it's worth testing this broadly
        G = MonteCarlo.calculate_greens_full1!(mc, mc.ut_stack, 37, 14)
        # general testing: ⪅ 1e-7
        # here: < 1e-13
        @test check(G, calc_greens(mc, 37, 14), 0, 1e-11)
    end

    setprecision(BigFloat, old_precision)
end