using Random
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
    @test _sign == 0
    _sign, s = create(down, 1, 1)
    @test _sign == 1.0 &&  s == updown
    _sign, s = create(updown, 1, 1)
    @test _sign == 0

    # create down at site 1
    _sign, s = create(void, 1, 2)
    @test _sign == 1.0 &&  s == down
    _sign, s = create(up, 1, 2)
    @test _sign == -1.0 &&  s == updown
    _sign, s = create(down, 1, 2)
    @test _sign == 0
    _sign, s = create(updown, 1, 2)
    @test _sign == 0

    # annihilate up at site 1
    _sign, s = annihilate(void, 1, 1)
    @test _sign == 0
    _sign, s = annihilate(up, 1, 1)
    @test _sign == 1.0 &&  s == void
    _sign, s = annihilate(down, 1, 1)
    @test _sign == 0
    _sign, s = annihilate(updown, 1, 1)
    @test _sign == 1.0 &&  s == down

    # annihilate down at site 1
    _sign, s = annihilate(void, 1, 2)
    @test _sign == 0 
    _sign, s = annihilate(up, 1, 2)
    @test _sign == 0
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
            UTG = expectation_value(
                s -> annihilate!(s, site2, substate2),
                s -> create!(s, site1, substate1),
                H, 0.1, 0.1, N_sites=model.l.sites
            )
            @test UTG ≈ real(G) atol=1e-14
            UTG = expectation_value(
                s -> annihilate!(s, site2, substate2),
                s -> create!(s, site1, substate1),
                H, 0.7, 0.7, N_sites=model.l.sites
            )
            @test UTG ≈ real(G) atol=1e-14
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
    mask = MonteCarlo.DistanceMask(model.l)

    @info "Running DQMC β=1.0, 10k + 50k sweeps, ≈4s"
    Random.seed!(123)
    dqmc = DQMC(model, beta=5.0, delta_tau = 0.1, safe_mult=5, measurements = Dict{Symbol, MonteCarlo.AbstractMeasurement}())
    push!(dqmc, :Greens => MonteCarlo.GreensMeasurement)
    push!(dqmc, :Occs => MonteCarlo.OccupationMeasurement)
    MonteCarlo.unsafe_push!(dqmc, :CDC => MonteCarlo.ChargeDensityCorrelationMeasurement(dqmc, model, mask=mask))
    push!(dqmc, :Magn => MonteCarlo.MagnetizationMeasurement)
    MonteCarlo.unsafe_push!(dqmc, :SDC => MonteCarlo.SpinDensityCorrelationMeasurement(dqmc, model, mask=mask))
    MonteCarlo.unsafe_push!(dqmc, :PC => MonteCarlo.PairingCorrelationMeasurement(dqmc, model, mask=mask))
    
    # Unequal time
    # N = MonteCarlo.nslices(dqmc) # 10
    l1s = [0, 4, 5, 6, 0]
    l2s = [1, 5, 5, 7, MonteCarlo.nslices(dqmc)]
    dqmc.ut_stack = MonteCarlo.UnequalTimeStack(dqmc)
    # Why is UTGreensMeasurement not defined?
    UTG = MonteCarlo.UTGreensMeasurement
    MonteCarlo.unsafe_push!(dqmc, :UTG1 => UTG(dqmc, model, slice1=l1s[1], slice2=l2s[1]))
    MonteCarlo.unsafe_push!(dqmc, :UTG2 => UTG(dqmc, model, slice1=l1s[2], slice2=l2s[2]))
    MonteCarlo.unsafe_push!(dqmc, :UTG3 => UTG(dqmc, model, slice1=l1s[3], slice2=l2s[3]))
    MonteCarlo.unsafe_push!(dqmc, :UTG4 => UTG(dqmc, model, slice1=l1s[4], slice2=l2s[4]))
    MonteCarlo.unsafe_push!(dqmc, :UTG5 => UTG(dqmc, model, slice1=l1s[5], slice2=l2s[5]))

    # MonteCarlo.enable_benchmarks()

    @time run!(dqmc, thermalization = 10_000, sweeps = 50_000, verbose=false)

    # MonteCarlo.disable_benchmarks()
    # MonteCarlo.print_timer()

    @info "Running ED"
    @time begin
    H = HamiltonMatrix(model)

    # Absolute tolerance from Trotter decompositon
    atol = dqmc.p.delta_tau^2
    rtol = 0.01

    # G_DQMC is smaller because it doesn't differentiate between spin up/down
    @testset "Greens" begin
        G_DQMC = mean(dqmc.measurements[:Greens].obs)
        occs = mean(dqmc.measurements[:Occs].obs)                                   # measuring
        occs2 = mean(MonteCarlo.occupations(dqmc.measurements[:Greens]))            # wrapping
        occs3 = mean(MonteCarlo.OccupationMeasurement(dqmc.measurements[:Greens]))  # copying
        G_ED = calculate_Greens_matrix(H, model.l, beta=dqmc.p.beta)
        for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
            @test isapprox(G_DQMC[i, j], G_ED[i, j], atol=atol, rtol=rtol)
        end
        for i in 1:size(G_DQMC, 1)
            @test isapprox(occs[i], G_ED[i, i], atol=atol, rtol=rtol)
            @test isapprox(occs2[i], G_ED[i, i], atol=atol, rtol=rtol)
            @test isapprox(occs3[i], G_ED[i, i], atol=atol, rtol=rtol)
        end
    end

    @testset "Charge Density Correlation" begin
        CDC = mean(dqmc.measurements[:CDC].obs)
        N = MonteCarlo.nsites(model)
        for dir in 1:length(mask)
            ED_CDC = 0.0
            for (src, trg) in MonteCarlo.getdirorder(mask, dir)
                ED_CDC += expectation_value(
                    charge_density_correlation(trg, src),
                    H, beta = dqmc.p.beta, N_sites = N
                )
            end
            @test ED_CDC/N ≈ CDC[dir] atol=atol rtol=rtol
        end
    end

    @testset "Magnetization x" begin
        Mx = mean(dqmc.measurements[:Magn].x)
        for site in 1:length(Mx)
            ED_Mx = expectation_value(m_x(site), H, beta = dqmc.p.beta, N_sites = MonteCarlo.nsites(model))
            @test ED_Mx ≈ Mx[site] atol=atol rtol=rtol
        end
    end
    @testset "Magnetization y" begin
        My = mean(dqmc.measurements[:Magn].y)
        for site in 1:length(My)
            ED_My = expectation_value(m_y(site), H, beta = dqmc.p.beta, N_sites = MonteCarlo.nsites(model))
            @test ED_My ≈ My[site] atol=atol rtol=rtol
        end
    end
    @testset "Magnetization z" begin
        Mz = mean(dqmc.measurements[:Magn].z)
        for site in 1:length(Mz)
            ED_Mz = expectation_value(m_z(site), H, beta = dqmc.p.beta, N_sites = MonteCarlo.nsites(model))
            @test ED_Mz ≈ Mz[site] atol=atol rtol=rtol
        end
    end

    @testset "Spin density correlation x" begin
        SDCx = mean(dqmc.measurements[:SDC].x)
        N = MonteCarlo.nsites(model)
        for offset in 1:length(mask)
            ED_SDCx = 0.0
            for (src, trg) in MonteCarlo.getdirorder(mask, offset)
                ED_SDCx += expectation_value(
                    spin_density_correlation_x(trg, src),
                    H, beta = dqmc.p.beta, N_sites = N
                )
            end
            @test ED_SDCx/N ≈ SDCx[offset] atol=atol rtol=rtol
        end
    end
    @testset "Spin density correlation y" begin
        SDCy = mean(dqmc.measurements[:SDC].y)
        N = MonteCarlo.nsites(model)
        for offset in 1:length(mask)
            ED_SDCy = 0.0
            for (src, trg) in MonteCarlo.getdirorder(mask, offset)
                ED_SDCy += expectation_value(
                    spin_density_correlation_y(trg, src),
                    H, beta = dqmc.p.beta, N_sites = N
                )
            end
            @test ED_SDCy/N ≈ SDCy[offset] atol=atol rtol=rtol
        end
    end
    @testset "Spin density correlation z" begin
        SDCz = mean(dqmc.measurements[:SDC].z)
        N = MonteCarlo.nsites(model)
        for offset in 1:length(mask)
            ED_SDCz = 0.0
            for (src, trg) in MonteCarlo.getdirorder(mask, offset)
                ED_SDCz += expectation_value(
                    spin_density_correlation_z(trg, src),
                    H, beta = dqmc.p.beta, N_sites = N
                )
            end
            @test ED_SDCz/N ≈ SDCz[offset] atol=atol rtol=rtol
        end
    end

    @testset "Pairing Correlation" begin
        PC = mean(dqmc.measurements[:PC])
        N = MonteCarlo.nsites(model)
        for offset in 1:length(mask)
            ED_PC = 0.0
            for (src1, trg1) in MonteCarlo.getdirorder(mask, offset)
                for (src2, trg2) in MonteCarlo.getdirorder(mask, offset)
                    ED_PC += expectation_value(
                        pairing_correlation(
                            trg2, src2, trg1, src1
                        ), H, beta = dqmc.p.beta, N_sites = N
                    )
                end
            end
            @test ED_PC/N^2 ≈ PC[offset] atol=atol rtol=rtol
        end
    end
    end
    # @info "Unequal time, oh boy"
    @time begin
        @testset "Unequal Time Greens" begin
            for (i, tau1, tau2) in zip(1:5, 0.1l1s, 0.1l2s)
                UTG = mean(dqmc.measurements[Symbol(:UTG, i)])
                N = MonteCarlo.nsites(model)
                ED_UTG = calculate_Greens_matrix(H, tau2, tau1, model.l, beta=dqmc.p.beta)
                # @info maximum(abs.(UTG .- ED_UTG[1:4, 1:4]))
                @test isapprox(UTG, ED_UTG[1:4, 1:4], atol=atol, rtol=rtol)
            end
        end
    end
end
