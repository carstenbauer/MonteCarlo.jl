include("ED.jl")

@testset "ED checks" begin
    void = state_from_integer(0, 1, 2)
    up = state_from_integer(1, 1, 2)
    down = state_from_integer(2, 1, 2)
    updown = state_from_integer(3, 1, 2)

    # create up at site 1
    _sign, s = create(void, 1, 1)
    @test _sign == 1.0 &&  s == up
    _sign, s = create(up, 1, 1)
    @test _sign == 0 &&  s == 0
    _sign, s = create(down, 1, 1)
    @test _sign == 1.0 &&  s == updown
    _sign, s = create(updown, 1, 1)
    @test _sign == 0 &&  s == 0

    # create down at site 1
    _sign, s = create(void, 1, 2)
    @test _sign == 1.0 &&  s == down
    _sign, s = create(up, 1, 2)
    @test _sign == -1.0 &&  s == updown
    _sign, s = create(down, 1, 2)
    @test _sign == 0 &&  s == 0
    _sign, s = create(updown, 1, 2)
    @test _sign == 0 &&  s == 0

    # annihilate up at site 1
    _sign, s = annihilate(void, 1, 1)
    @test _sign == 0 &&  s == 0
    _sign, s = annihilate(up, 1, 1)
    @test _sign == 1.0 &&  s == void
    _sign, s = annihilate(down, 1, 1)
    @test _sign == 0 &&  s == 0
    _sign, s = annihilate(updown, 1, 1)
    @test _sign == 1.0 &&  s == down

    # annihilate down at site 1
    _sign, s = annihilate(void, 1, 2)
    @test _sign == 0 &&  s == 0
    _sign, s = annihilate(up, 1, 2)
    @test _sign == 0 &&  s == 0
    _sign, s = annihilate(down, 1, 2)
    @test _sign == 1.0 &&  s == void
    _sign, s = annihilate(updown, 1, 2)
    @test _sign == -1.0 &&  s == up

    # check Greens consistency
    model = HubbardModelAttractive(
        L = 2, dims = 2,
        U = rand(), mu = rand(), t = rand()
    )
    H = HamiltonMatrix(model)
    for substate1 in 1:2, substate2 in 1:2
        for site1 in 1:model.l.sites, site2 in 1:model.l.sites
            G = expectation_value(
                Greens(site1, site2, substate1, substate2),
                H,
                N_sites = model.l.sites,
            )
            G_perm = expectation_value(
                Greens_permuted(site1, site2, substate1, substate2),
                H,
                N_sites = model.l.sites,
            )
            @test G ≈ G_perm
        end
    end
end


@testset "Attractive Hubbard Model (ED)" begin
    model = HubbardModelAttractive(
        L = 2,
        dims = 2,
        U = 1.0,
        mu = 1.0,
        t = 1.0
    )

    @info "Running DQMC β=1.0, 100k + 100k sweeps, ≈1min"
    dqmc = DQMC(model, beta=1.0)
    run!(dqmc, thermalization = 100_000, sweeps = 100_000, verbose=false)
    G_DQMC = mean(dqmc.obs["greens"])

    @info "Running ED"
    H = HamiltonMatrix(model)
    G_ED = calculate_Greens_matrix(H, model.l, beta=1.0)

    # G_DQMC is smaller because it doesn't differentiate between spin up/down
    for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
        @test isapprox(G_DQMC[i, j], G_ED[i, j], atol=0.025, rtol=0.1)
        # @test isapprox(G_DQMC[i, j], G_ED[i, j])
    end
end
