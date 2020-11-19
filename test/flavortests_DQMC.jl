include("testfunctions.jl")
@testset "DQMC Parameters" begin
    P1 = MonteCarlo.DQMCParameters(beta=5.0) #defaults to delta_tau = 0.1
    @test all((P1.beta, P1.delta_tau, P1.slices) .== (5.0, 0.1, 50))

    P2 = MonteCarlo.DQMCParameters(beta=5.0, delta_tau=0.01)
    @test all((P2.beta, P2.delta_tau, P2.slices) .== (5.0, 0.01, 500))

    P3 = MonteCarlo.DQMCParameters(beta=50.0, slices=20)
    @test all((P3.beta, P3.delta_tau, P3.slices) .== (50.0, 2.5, 20))

    P4 = MonteCarlo.DQMCParameters(delta_tau=0.1, slices=50)
    @test all((P4.beta, P4.delta_tau, P4.slices) .== (5.0, 0.1, 50))
end

@testset "DQMC" begin
    m = HubbardModelAttractive(dims=1, L=8);

    # constructors
    dqmc = DQMC(m; beta=5.0)
    @test m.L == 8 && m.dims == 1
    @test typeof(m) == MonteCarlo.HubbardModelAttractive{MonteCarlo.Chain}
    d = Dict(:dims=>2,:L=>3)
    m = HubbardModelAttractive(d)
    @test typeof(m) == MonteCarlo.HubbardModelAttractive{MonteCarlo.SquareLattice}
    @test m.L == 3 && m.dims == 2

    # generic checkerboard
    sq = MonteCarlo.SquareLattice(4);
    @test MonteCarlo.build_checkerboard(sq) == ([1.0 3.0 5.0 7.0 9.0 11.0 13.0 15.0 1.0 2.0 4.0 6.0 9.0 10.0 12.0 14.0 2.0 3.0 4.0 5.0 8.0 10.0 11.0 16.0 6.0 7.0 8.0 12.0 13.0 14.0 15.0 16.0; 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 5.0 3.0 8.0 7.0 13.0 11.0 16.0 15.0 6.0 7.0 1.0 9.0 12.0 14.0 15.0 13.0 10.0 11.0 5.0 9.0 1.0 2.0 3.0 4.0; 1.0 5.0 9.0 13.0 17.0 21.0 25.0 29.0 2.0 3.0 8.0 11.0 18.0 19.0 24.0 27.0 4.0 6.0 7.0 10.0 16.0 20.0 22.0 31.0 12.0 14.0 15.0 23.0 26.0 28.0 30.0 32.0], UnitRange[1:8, 9:16, 17:24, 25:32], 4)

    m = HubbardModelAttractive(dims=2, L=8, mu=0.5)
    mc1 = DQMC(m, beta=5.0)
    mc2 = DQMC(m, beta=5.0, checkerboard=false)
    mc2.conf = deepcopy(mc1.conf)
    MonteCarlo.init_hopping_matrices(mc1, m)
    MonteCarlo.init_hopping_matrices(mc2, m)
    MonteCarlo.build_stack(mc1, mc1.s)
    MonteCarlo.build_stack(mc2, mc2.s)
    @test MonteCarlo.slice_matrix(mc1, m, 1, 1.) == MonteCarlo.slice_matrix(mc2, m, 1, 1.)

    mc = DQMC(m, beta=5.0, checkerboard=true, delta_tau=0.1)
    MonteCarlo.init_hopping_matrices(mc, m)
    hop_mat_exp_chkr = foldl(*,mc.s.chkr_hop_half) * sqrt.(mc.s.chkr_mu)
    r = MonteCarlo.effreldiff(mc.s.hopping_matrix_exp,hop_mat_exp_chkr)
    r[findall(x -> x==zero(x), hop_mat_exp_chkr)] .= 0.
    @test maximum(MonteCarlo.absdiff(mc.s.hopping_matrix_exp,hop_mat_exp_chkr)) <= mc.p.delta_tau

    # initial greens test
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.build_stack(mc, mc.s)
    MonteCarlo.propagate(mc)
    # With this we effectively test calculate_greens without wrap_greens
    greens, = calculate_greens_and_logdet(mc, mc.s.current_slice)
    MonteCarlo.wrap_greens!(mc, greens, mc.s.current_slice+1, -1)
    @test greens ≈ mc.s.greens
    # here with a single implied wrap
    greens, = calculate_greens_and_logdet(mc, mc.s.current_slice-1)
    @test maximum(MonteCarlo.absdiff(greens, mc.s.greens)) < 1e-12

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.s.greens, mc.s.current_slice - k, -1)
    end
    greens, = calculate_greens_and_logdet(mc, mc.s.current_slice-11)
    @test maximum(MonteCarlo.absdiff(greens, mc.s.greens)) < 1e-9

    # Check greens reconstruction used in replay
    mc = DQMC(m, beta=5.0, safe_mult=5)
    # Make sure this works with any values
    for k in shuffle(0:MonteCarlo.nslices(mc))
        G1, _ = calculate_greens_and_logdet(mc, k)
        G2 = MonteCarlo.calculate_greens(mc, k)
        @test G1 ≈ G2
    end

    include("slice_matrices.jl")
end

@testset "Unequal Time Stack" begin
    m = HubbardModelAttractive(dims=1, L=6);
    dqmc = DQMC(m; beta=15.0, safe_mult=5)
    MonteCarlo.initialize_stack(dqmc, dqmc.ut_stack)

    MonteCarlo.build_stack(dqmc, dqmc.ut_stack)
    MonteCarlo.build_stack(dqmc, dqmc.s)
        
    # test B(τ, 1) / B_l1 stacks
    @test dqmc.s.u_stack ≈ dqmc.ut_stack.forward_u_stack
    @test dqmc.s.d_stack ≈ dqmc.ut_stack.forward_d_stack
    @test dqmc.s.t_stack ≈ dqmc.ut_stack.forward_t_stack

    # generate B(β, τ) / B_Nl stacks
    while dqmc.s.direction == -1
        MonteCarlo.propagate(dqmc)
    end

    # test B(β, τ) / B_Nl stacks
    # Note: dqmc.s doesn't generate the full stack here
    @test dqmc.s.u_stack[2:end] ≈ dqmc.ut_stack.backward_u_stack[2:end]
    @test dqmc.s.d_stack[2:end] ≈ dqmc.ut_stack.backward_d_stack[2:end]
    @test dqmc.s.t_stack[2:end] ≈ dqmc.ut_stack.backward_t_stack[2:end]

    while !(
            MonteCarlo.current_slice(dqmc) == 1 &&
            dqmc.s.direction == -1
        )
        MonteCarlo.propagate(dqmc)
    end

    # Check equal time greens functions from equal time and unequal time
    # calculation against each other
    for slice in 0:MonteCarlo.nslices(dqmc)
        G1 = deepcopy(MonteCarlo.calculate_greens(dqmc, slice))
        G2 = deepcopy(MonteCarlo.calculate_greens(dqmc, slice, slice))
        @test maximum(abs.(G1 .- G2)) < 1e-14
    end

    # Check G(t, 0) + G(0, beta - t) = 0
    for slice in 0:MonteCarlo.nslices(dqmc)-1
        G1 = MonteCarlo.greens(dqmc, slice, 0)
        G2 = MonteCarlo.greens(dqmc, slice, MonteCarlo.nslices(dqmc))
        @test G1 ≈ -G2 atol = 1e-14
    end


    # Check Iterators
    # For some reason Gkl/Gk0 errors jump in the last 10 steps
    # when using UnequalTimeStack
    
    # As in G(τ = Δτ * k, 0)
    # Note: G(0, l) = G(M-l, 0)
    Gk0s = [deepcopy(MonteCarlo.greens(dqmc, slice, 0)) for slice in 0:MonteCarlo.nslices(dqmc)]
    
    # Calculated from UnequalTimeStack (high precision)
    it = MonteCarlo.GreensIterator(dqmc, :, 0, dqmc.p.safe_mult)
    for (i, G) in enumerate(it)
        @test maximum(abs.(G .- Gk0s[i])) < 1e-14
    end

    # Calculated from mc.s.greens using UDT decompositions (lower precision)
    it = MonteCarlo.GreensIterator(dqmc, :, 0, 4dqmc.p.safe_mult)
    for (i, G) in enumerate(it)
        @test maximum(abs.(G .- Gk0s[i])) < 1e-11
    end

    Gkks = map(0:MonteCarlo.nslices(dqmc)) do slice
        g = MonteCarlo.calculate_greens(dqmc, slice)
        deepcopy(MonteCarlo._greens!(dqmc, dqmc.s.greens_temp, g))
    end
    G0ks = [deepcopy(MonteCarlo.greens(dqmc, 0, slice)) for slice in 0:MonteCarlo.nslices(dqmc)]
    MonteCarlo.calculate_greens(dqmc, 0) # restore mc.s.greens

    # high precision
    it = MonteCarlo.CombinedGreensIterator(dqmc, dqmc.p.safe_mult)
    for (i, (G0k, Gk0, Gkk)) in enumerate(it)
        @test maximum(abs.(Gk0 .- Gk0s[i+1])) < 1e-14
        @test maximum(abs.(G0k .- G0ks[i+1])) < 1e-14
        @test maximum(abs.(Gkk .- Gkks[i+1])) < 1e-14
    end

    # low precision
    it = MonteCarlo.CombinedGreensIterator(dqmc, 4dqmc.p.safe_mult)
    for (i, (G0k, Gk0, Gkk)) in enumerate(it)
        @test maximum(abs.(Gk0 .- Gk0s[i+1])) < 1e-10
        @test maximum(abs.(G0k .- G0ks[i+1])) < 1e-10
        @test maximum(abs.(Gkk .- Gkks[i+1])) < 1e-10
    end
end