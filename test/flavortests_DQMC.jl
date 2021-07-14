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

    @test parameters(P4) == (beta = 5.0, delta_tau = 0.1, thermalization = 100, sweeps = 100)
end

@testset "Magnitude Stats" begin
    m = MonteCarlo.MagnitudeStats()
    @test m.max == -Inf
    @test m.min == Inf
    @test m.sum == 0.0
    @test m.count == 0

    @test min(m) == 0.0
    @test max(m) == 0.0
    @test mean(m) == 0.0
    @test length(m) == 0

    push!(m, 1e-7)
    @test m.max == log10(1e-7)
    @test m.min == log10(1e-7)
    @test m.sum == log10(1e-7)
    @test m.count == 1

    push!(m, 1e-5)
    @test m.max == log10(1e-5)
    @test m.min == log10(1e-7)
    @test m.sum == log10(1e-7) + log10(1e-5)
    @test m.count == 2

    @test min(m) ≈ 1e-7
    @test max(m) ≈ 1e-5
    @test mean(m) ≈ 10^(0.5*(log10(1e-7) + log10(1e-5)))
    @test length(m) == 2
end

@testset "DQMC utilities " begin
    m = HubbardModelAttractive(8, 1);

    # constructors
    dqmc = DQMC(m; beta=5.0)

    @test MonteCarlo.beta(dqmc) == dqmc.parameters.beta
    @test MonteCarlo.nslices(dqmc) == dqmc.parameters.slices
    @test MonteCarlo.current_slice(dqmc) == dqmc.stack.current_slice
    @test MonteCarlo.configurations(dqmc) == dqmc.recorder
    @test MonteCarlo.parameters(dqmc) == merge(
        parameters(dqmc.parameters), parameters(dqmc.model)
    )

    io = IOBuffer()
    show(io, dqmc)
    @test String(take!(io)) == "Determinant quantum Monte Carlo simulation\nModel: attractive Hubbard model, 8 sites\nBeta: 5.0 (T ≈ 0.2)\nMeasurements: 0 (0 + 0)"

    # DQMC mandatory
    m = DummyModel(SquareLattice(2))
    @test_throws MethodError rand(DQMC, m, 4)
    @test_throws ErrorException MonteCarlo.nflavors(m)
    @test_throws MethodError MonteCarlo.hopping_matrix(dqmc, m)
    @test_throws MethodError MonteCarlo.interaction_matrix_exp!(dqmc, m, zeros(2,2), zeros(2,2), 1, 1.0)
    @test_throws MethodError MonteCarlo.propose_local(dqmc, m, 1, 1, zeros(2,2))
    @test_throws MethodError MonteCarlo.accept_local!(dqmc, m, 1, 1, zeros(2,2), 1.0, 1.0, nothing)

    MonteCarlo.nflavors(::DummyModel) = 2
    MonteCarlo.lattice(m::DummyModel) = m.lattice

    # DQMC optional
    @test MonteCarlo.greenseltype(DQMC, m) == ComplexF64
    @test MonteCarlo.interaction_matrix_type(DQMC, m) == Matrix{ComplexF64}
    @test MonteCarlo.init_interaction_matrix(m) == zeros(ComplexF64, 8, 8)
    @test_throws MethodError MonteCarlo.energy_boson(dqmc, m, MonteCarlo.conf(dqmc))
    @test parameters(m) == NamedTuple()
end

@testset "DQMC stack" begin
    m = HubbardModelAttractive(8, 1);

    # constructors
    dqmc = DQMC(m; beta=5.0)

    # generic checkerboard
    sq = MonteCarlo.SquareLattice(4);
    @test MonteCarlo.build_checkerboard(sq) == ([1.0 3.0 5.0 7.0 9.0 11.0 13.0 15.0 1.0 2.0 4.0 6.0 9.0 10.0 12.0 14.0 2.0 3.0 4.0 5.0 8.0 10.0 11.0 16.0 6.0 7.0 8.0 12.0 13.0 14.0 15.0 16.0; 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 5.0 3.0 8.0 7.0 13.0 11.0 16.0 15.0 6.0 7.0 1.0 9.0 12.0 14.0 15.0 13.0 10.0 11.0 5.0 9.0 1.0 2.0 3.0 4.0; 1.0 5.0 9.0 13.0 17.0 21.0 25.0 29.0 2.0 3.0 8.0 11.0 18.0 19.0 24.0 27.0 4.0 6.0 7.0 10.0 16.0 20.0 22.0 31.0 12.0 14.0 15.0 23.0 26.0 28.0 30.0 32.0], UnitRange[1:8, 9:16, 17:24, 25:32], 4)

    m = HubbardModelAttractive(8, 2, mu=0.5)
    mc1 = DQMC(m, beta=5.0)
    mc2 = DQMC(m, beta=5.0, checkerboard=false)
    mc2.conf = deepcopy(mc1.conf)
    MonteCarlo.init_hopping_matrices(mc1, m)
    MonteCarlo.init_hopping_matrices(mc2, m)
    MonteCarlo.build_stack(mc1, mc1.stack)
    MonteCarlo.build_stack(mc2, mc2.stack)
    @test MonteCarlo.slice_matrix(mc1, m, 1, 1.) == MonteCarlo.slice_matrix(mc2, m, 1, 1.)

    mc = DQMC(m, beta=5.0, checkerboard=true, delta_tau=0.1)
    MonteCarlo.init_hopping_matrices(mc, m)
    hop_mat_exp_chkr = foldl(*,mc.stack.chkr_hop_half) * sqrt.(mc.stack.chkr_mu)
    r = MonteCarlo.effreldiff(mc.stack.hopping_matrix_exp,hop_mat_exp_chkr)
    r[findall(x -> x==zero(x), hop_mat_exp_chkr)] .= 0.
    @test maximum(MonteCarlo.absdiff(mc.stack.hopping_matrix_exp,hop_mat_exp_chkr)) <= mc.parameters.delta_tau

    # initial greens test
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.build_stack(mc, mc.stack)
    MonteCarlo.propagate(mc)
    # With this we effectively test calculate_greens without wrap_greens
    greens, = calculate_greens_and_logdet(mc, mc.stack.current_slice)
    MonteCarlo.wrap_greens!(mc, greens, mc.stack.current_slice+1, -1)
    @test greens ≈ mc.stack.greens
    # here with a single implied wrap
    greens, = calculate_greens_and_logdet(mc, mc.stack.current_slice-1)
    @test maximum(MonteCarlo.absdiff(greens, mc.stack.greens)) < 1e-12

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.stack.greens, mc.stack.current_slice - k, -1)
    end
    greens, = calculate_greens_and_logdet(mc, mc.stack.current_slice-11)
    @test maximum(MonteCarlo.absdiff(greens, mc.stack.greens)) < 1e-9

    # Check greens reconstruction used in replay
    mc = DQMC(m, beta=5.0, safe_mult=5)
    # Make sure this works with any values
    for k in shuffle(0:MonteCarlo.nslices(mc))
        G1, _ = calculate_greens_and_logdet(mc, k)
        G2 = MonteCarlo.calculate_greens(mc, k)
        @test G1 ≈ G2
    end

    # check reverse_build_stack
    m = HubbardModelAttractive(8, 1);
    mc1 = DQMC(m; beta=5.0)
    MonteCarlo.init!(mc1)
    MonteCarlo.build_stack(mc1, mc1.stack)
    MonteCarlo.propagate(mc1)
    for t in 1:MonteCarlo.nslices(mc1)
        MonteCarlo.propagate(mc1)
    end

    m = HubbardModelAttractive(8, 1);
    mc2 = DQMC(m; beta=5.0)
    mc2.conf .= mc1.conf
    MonteCarlo.init!(mc2)
    MonteCarlo.reverse_build_stack(mc2, mc2.stack)
    MonteCarlo.propagate(mc2)

    @test mc1.stack.current_slice == mc2.stack.current_slice
    @test mc1.stack.direction == mc2.stack.direction
    @test mc1.stack.greens ≈ mc2.stack.greens
    for field in (:u_stack, :d_stack, :t_stack)
        for i in 1:mc1.stack.n_elements
            @test getfield(mc1.stack, field)[i] ≈ getfield(mc2.stack, field)[i]
        end
    end

    for field in (:Ul, :Ur, :Dl, :Dr, :Tl, :Tr)
        @test getfield(mc1.stack, field) ≈ getfield(mc2.stack, field)
    end
    
end

@testset "LinAlg and Slice Matrices" begin
    include("slice_matrices.jl")
end

@testset "Unequal Time Stack" begin
    m = HubbardModelAttractive(6, 1);
    dqmc = DQMC(m; beta=15.0, safe_mult=5)
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
    # Note: G(0, l) = G(M-l, 0)
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
        it = MonteCarlo.CombinedGreensIterator(dqmc, recalculate = dqmc.parameters.safe_mult)
        @test it isa CombinedGreensIterator
        @test it.recalculate == dqmc.parameters.safe_mult
        @test it.start == 1
        @test it.stop == MonteCarlo.nslices(dqmc)
        @test length(it) == it.stop - it.start + 1

        iter = MonteCarlo.init(dqmc, it)
        @test iter isa MonteCarlo._CombinedGreensIterator
        @test iter.mc == dqmc
        @test iter.spec == it
        @test length(iter) == length(it)

        for (i, (G0k, Gk0, Gkk)) in enumerate(iter)
            @test maximum(abs.(Gk0.val .- Gk0s[i+1])) < 1e-14
            @test maximum(abs.(G0k.val .- G0ks[i+1])) < 1e-14
            @test maximum(abs.(Gkk.val .- Gkks[i+1])) < 1e-14
        end

        # low precision
        it = MonteCarlo.CombinedGreensIterator(dqmc, recalculate = 4dqmc.parameters.safe_mult)
        for (i, (G0k, Gk0, Gkk)) in enumerate(MonteCarlo.init(dqmc, it))
            @test maximum(abs.(Gk0.val .- Gk0s[i+1])) < 1e-10
            @test maximum(abs.(G0k.val .- G0ks[i+1])) < 1e-10
            @test maximum(abs.(Gkk.val .- Gkks[i+1])) < 1e-10
        end
    end
end


@testset "Exact Greens comparison (Analytic, medium systems)" begin
    # These are theoretically the same but their implementation differs on
    # some level. To make sure both are correct it makes sense to check both here.
    models = (
        HubbardModelRepulsive(7, 2, U = 0.0, t = 1.0),
        HubbardModelAttractive(8, 2, U = 0.0, mu = 1.0, t = 1.0)
    )

    @info "Exact Greens comparison"
    for model in models, beta in (1.0, 10.0)
        @testset "$(typeof(model))" begin
            Random.seed!(123)
            dqmc = DQMC(
                model, beta=beta, delta_tau = 0.1, safe_mult=5, recorder = Discarder, 
                thermalization = 1, sweeps = 2, measure_rate = 1
            )
            @info "Running DQMC ($(typeof(model).name)) β=$(dqmc.parameters.beta)"

            dqmc[:G] = greens_measurement(dqmc, model)
            @time run!(dqmc, verbose=false)
            
            # error tolerance
            atol = 1e-13
            rtol = 1e-13
            N = length(lattice(model))

            # Direct calculation similar to what DQMC should be doing
            T = Matrix(MonteCarlo.hopping_matrix(dqmc, model))
            # Doing an eigenvalue decomposition makes this pretty stable
            vals, U = eigen(exp(-T))
            D = Diagonal(vals)^(dqmc.parameters.beta)

            # Don't believe "Quantum Monte Carlo Methods", this is the right
            # formula (believe dos Santos DQMC review instead)
            G = U * inv(I + D) * adjoint(U)

            @test check(mean(dqmc[:G]), G, atol, rtol)
        end
    end
end