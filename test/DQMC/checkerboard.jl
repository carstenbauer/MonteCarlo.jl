using BinningAnalysis

@testset "Square lattice" begin
    # TODO:
    # make these test properties for different lattices:
    # - no duplication (first can have both src -> trg and trg -> src though)
    # - no missing bonds
    # - no douplicate attachments to the same site
    # - equal bond direction within a group
    # - no staggering within a group
    sq = MonteCarlo.SquareLattice(4);
    @test MonteCarlo.build_checkerboard(sq) == ([1 3 5 7 9 11 13 15 2 4 6 8 10 12 14 16 1 2 3 4 9 10 11 12 5 6 7 8 13 14 15 16 1 3 5 7 9 11 13 15 2 4 6 8 10 12 14 16 1 2 3 4 9 10 11 12 5 6 7 8 13 14 15 16; 2 4 6 8 10 12 14 16 3 1 7 5 11 9 15 13 5 6 7 8 13 14 15 16 9 10 11 12 1 2 3 4 4 2 8 6 12 10 16 14 1 3 5 7 9 11 13 15 13 14 15 16 5 6 7 8 1 2 3 4 9 10 11 12; 4294967296 139689615082304 139689615082304 4 4294967296 139689615082304 139689615082304 7 4294967296 139689615082304 139689615082304 1 8589934592 139689615082304 139689615082304 0 12884901888 139689615082304 139689615082304 3 12884901888 139689615082304 139689615082304 6 12884901888 139689615082304 139689615082304 9 12884901888 139689615082304 139689615082304 12 12884901888 139689615082304 139689615082304 15 12884901888 139689615082304 139689615082304 1 21474836480 139689615082304 139689615082304 1 25769803776 139689615082304 139689615082304 4 25769803776 139689615082304 139689615082304 7 25769803776 139689615082304 139689615082304 1 30064771072 139689615082304 139689615082304 4 30064771072 139689615082304 139689852103344 139689852106480], UnitRange{Int64}[1:8, 9:16, 17:24, 25:32, 33:40, 41:48, 49:56, 57:64], 8)
    @test MonteCarlo.build_checkerboard2(sq) == ([1 3 5 7 9 11 13 15 2 4 6 8 10 12 14 16 1 2 3 4 9 10 11 12 5 6 7 8 13 14 15 16; 2 4 6 8 10 12 14 16 3 1 7 5 11 9 15 13 5 6 7 8 13 14 15 16 9 10 11 12 1 2 3 4; 139689612432272 139689612432272 139689612432272 139687558314112 139689612432272 139689612432272 139689612432272 139689612454752 139689612454752 139689613930608 139689612432272 139687558465440 139689614017648 139687559457776 139689612432272 139687559458384 139689612432272 139689612454752 1 1 139687559549328 139689613930608 139689612432272 139689612454752 139689612454752 139689612432272 139687559679392 139689613930608 139687559680128 139689612454752 139689612454752 139687559710384], UnitRange{Int64}[1:8, 9:16, 17:24, 25:32], 4) 

    # TODO: rerun these tests for a couple of lattices

    # Run a normal simulation at fairly high beta because that can mess things 
    # up ... apparently?
    m = HubbardModel(L=4, dims = 2, mu = 1.3, t = 0.3)
    mc1 = DQMC(m, beta = 24.0, checkerboard = false)
    mc1[:G] = greens_measurement(mc1, m, obs = FullBinner(Matrix{Float64}))
    mc1[:CDC] = charge_density_correlation(mc1, m, obs = FullBinner(Array{Float64, 3}))
    mc1[:CCS] = current_current_susceptibility(mc1, m, obs = FullBinner(Array{Float64, 4}))
    run!(mc1, verbose = false)

    # replay the configs without checkerboard decomposition with the decomposition
    mc2 = DQMC(m, beta = 24.0, checkerboard = true)
    mc2[:G] = greens_measurement(mc2, m, obs = FullBinner(Matrix{Float64}))
    mc2[:CDC] = charge_density_correlation(mc2, m, obs = FullBinner(Array{Float64, 3}))
    mc2[:CCS] = current_current_susceptibility(mc2, m, obs = FullBinner(Array{Float64, 4}))
    replay!(mc2, mc1.recorder, verbose = false)

    # All of these should work with float precision 10^-15 ~ 10^-17 diff

    # verify hopping matrices
    for name in (Symbol(), :_inv, :_squared, :_inv_squared)
        fullname = Symbol(:hopping_matrix_exp, name)
        @test getfield(mc1.stack, fullname) ≈ Matrix(getfield(mc2.stack, fullname))
    end

    # verify working greens matrix
    @test mc1.stack.greens ≈ mc2.stack.greens

    # verify measurements
    for key in (:G, :CDC, :CCS)
        vals1 = mc1[key].observable.x
        vals2 = mc2[key].observable.x
        @test all(vals1 .≈ vals2)
    end
end