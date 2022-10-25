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
    @test MonteCarlo.build_checkerboard(sq) == [
        [1 => 2, 3 => 4, 5 => 6, 7 => 8, 9 => 10, 11 => 12, 13 => 14, 15 => 16], 
        [2 => 3, 4 => 1, 6 => 7, 8 => 5, 10 => 11, 12 => 9, 14 => 15, 16 => 13], 
        [1 => 5, 2 => 6, 3 => 7, 4 => 8, 9 => 13, 10 => 14, 11 => 15, 12 => 16], 
        [5 => 9, 6 => 10, 7 => 11, 8 => 12, 13 => 1, 14 => 2, 15 => 3, 16 => 4]
    ]

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