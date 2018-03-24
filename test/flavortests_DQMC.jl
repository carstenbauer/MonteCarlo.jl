@testset "DQMC" begin
    m = HubbardModelAttractive(dims=1, L=8);

    # constructors
    dqmc = DQMC(m; beta=5.0)
    @test m.L == 8 && m.dims == 1
    @test typeof(m) == MonteCarlo.HubbardModelAttractive{MonteCarlo.Chain}
    d = Dict{String,Any}(Pair{String,Any}("dims", 2),Pair{String,Any}("L", 3))
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
    MonteCarlo.build_stack(mc1)
    MonteCarlo.build_stack(mc2)
    @test MonteCarlo.slice_matrix(mc1, m, 1, 1.) == MonteCarlo.slice_matrix(mc2, m, 1, 1.)

    mc = DQMC(m, beta=5.0, checkerboard=true, delta_tau=0.1)
    MonteCarlo.init_hopping_matrices(mc, m)
    hop_mat_exp_chkr = foldl(*,mc.s.chkr_hop_half) * sqrt.(mc.s.chkr_mu)
    r = MonteCarlo.effreldiff(mc.s.hopping_matrix_exp,hop_mat_exp_chkr)
    r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
    @test maximum(MonteCarlo.absdiff(mc.s.hopping_matrix_exp,hop_mat_exp_chkr)) <= mc.p.delta_tau

    # initial greens test
    mc = DQMC(m, beta=5.0, safe_mult=1)
    MonteCarlo.build_stack(mc)
    MonteCarlo.propagate(mc)
    greens, = MonteCarlo.calculate_greens_and_logdet(mc, mc.s.current_slice, 1)
    @test maximum(MonteCarlo.absdiff(greens, mc.s.greens)) < 1e-13

    # wrap greens test
    for k in 0:9
        MonteCarlo.wrap_greens!(mc, mc.s.greens, mc.s.current_slice - k, -1)
    end
    greens, = MonteCarlo.calculate_greens_and_logdet(mc, mc.s.current_slice-10, 1)
    @test maximum(MonteCarlo.absdiff(greens, mc.s.greens)) < 1e-9

end
