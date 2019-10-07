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

    @info "Running DQMC β=1.0, 10k + 20k sweeps, ≈1min"
    dqmc = DQMC(model, beta=1.0, delta_tau = 0.025, measurements = Dict{Symbol, AbstractMeasurement}())
    push!(dqmc, :Greens => MonteCarlo.GreensMeasurement)
    push!(dqmc, :CDC => MonteCarlo.ChargeDensityCorrelationMeasurement)
    push!(dqmc, :Magn => MonteCarlo.MagnetizationMeasurement)
    push!(dqmc, :SDC => MonteCarlo.SpinDensityCorrelationMeasurement)
    push!(dqmc, :PC => MonteCarlo.PairingCorrelationMeasurement)
    @time run!(dqmc, thermalization = 10_000, sweeps = 50_000, verbose=false)

    @info "Running ED"
    H = HamiltonMatrix(model)

    atol = 0.025
    rtol = 0.1

    # G_DQMC is smaller because it doesn't differentiate between spin up/down
    @testset "Greens" begin
        G_DQMC = mean(dqmc.measurements[:Greens].obs)
        G_ED = calculate_Greens_matrix(H, model.l, beta=1.0)
        for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
            @test isapprox(G_DQMC[i, j], G_ED[i, j], atol=atol, rtol=rtol)
        end
    end

    @testset "Charge Density Correlation" begin
        CDC = mean(dqmc.measurements[:CDC].obs)
        for site1 in 1:size(CDC, 1), site2 in 1:size(CDC, 2)
            ED_CDC = expectation_value(
                charge_density_correlation(site1, site2),
                H, beta = 1.0, N_sites = MonteCarlo.nsites(model)
            )
            @test ED_CDC ≈ CDC[site1, site2] atol=atol rtol=rtol
        end
    end

    @testset "Magnetization x" begin
        Mx = mean(dqmc.measurements[:Magn].x)
        for site in 1:length(Mx)
            ED_Mx = expectation_value(m_x(site), H, beta = 1.0, N_sites = MonteCarlo.nsites(model))
            @test ED_Mx ≈ Mx[site] atol=atol rtol=rtol
        end
    end
    @testset "Magnetization y" begin
        My = mean(dqmc.measurements[:Magn].y)
        for site in 1:length(My)
            ED_My = expectation_value(m_y(site), H, beta = 1.0, N_sites = MonteCarlo.nsites(model))
            @test ED_My ≈ My[site] atol=atol rtol=rtol
        end
    end
    @testset "Magnetization z" begin
        Mz = mean(dqmc.measurements[:Magn].z)
        for site in 1:length(Mz)
            ED_Mz = expectation_value(m_z(site), H, beta = 1.0, N_sites = MonteCarlo.nsites(model))
            @test ED_Mz ≈ Mz[site] atol=atol rtol=rtol
        end
    end

    @testset "Spin density correlation x" begin
        SDCx = mean(dqmc.measurements[:SDC].x)
        for site1 in 1:size(SDCx, 1), site2 in 1:size(SDCx, 2)
            ED_SDCx = expectation_value(
                spin_density_correlation_x(site1, site2),
                H, beta = 1.0, N_sites = MonteCarlo.nsites(model)
            )
            @test ED_SDCx ≈ SDCx[site1, site2] atol=atol rtol=rtol
        end
    end
    @testset "Spin density correlation y" begin
        SDCy = mean(dqmc.measurements[:SDC].y)
        for site1 in 1:size(SDCy, 1), site2 in 1:size(SDCy, 2)
            ED_SDCy = expectation_value(
                spin_density_correlation_y(site1, site2),
                H, beta = 1.0, N_sites = MonteCarlo.nsites(model)
            )
            @test ED_SDCy ≈ SDCy[site1, site2] atol=atol rtol=rtol
        end
    end
    @testset "Spin density correlation z" begin
        SDCz = mean(dqmc.measurements[:SDC].z)
        for site1 in 1:size(SDCz, 1), site2 in 1:size(SDCz, 2)
            ED_SDCz = expectation_value(
                spin_density_correlation_z(site1, site2),
                H, beta = 1.0, N_sites = MonteCarlo.nsites(model)
            )
            @test ED_SDCz ≈ SDCz[site1, site2] atol=atol rtol=rtol
        end
    end

    @testset "Pairing Correlation" begin
        PC = mean(dqmc.measurements[:PC].mat)
        for site1 in 1:size(PC, 1), site2 in 1:size(PC, 2)
            ED_PC = expectation_value(
                pairing_correlation(site1, site2),
                H, beta = 1.0, N_sites = MonteCarlo.nsites(model)
            )
            @test ED_PC ≈ PC[site1, site2] atol=atol rtol=rtol
        end
    end
end
