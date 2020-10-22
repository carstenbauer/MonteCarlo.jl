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


@testset "Repulsive/Attractive Hubbard Model (ED)" begin
    models = (
        HubbardModelRepulsive(L = 2, dims = 2, U = 1.0, t = 1.0),
        HubbardModelAttractive(L = 2, dims = 2, U = 1.0, mu = 1.0, t = 1.0)
    )

    for model in models
        @testset "$(typeof(model))" begin
            @info "Running DQMC ($(typeof(model).name)) β=5.0, 10k + 10k sweeps"
            Random.seed!(123)
            dqmc = DQMC(
                model, beta=5.0, delta_tau = 0.1, safe_mult=5, recorder = Discarder, 
                thermalization = 10_000, sweeps = 10_000
            )

            dqmc[:G]    = greens_measurement(dqmc, model)
            dqmc[:Occs] = occupation_measurement(dqmc, model)
            dqmc[:CDC]  = CDC_measurement(dqmc, model)
            dqmc[:Mx]   = magnetization_measurement(dqmc, model, :x)
            dqmc[:My]   = magnetization_measurement(dqmc, model, :y)
            dqmc[:Mz]   = magnetization_measurement(dqmc, model, :z)
            dqmc[:SDCx] = SDC_measurement(dqmc, model, :x)
            dqmc[:SDCy] = SDC_measurement(dqmc, model, :y)
            dqmc[:SDCz] = SDC_measurement(dqmc, model, :z)
            dqmc[:PC]   = PC_measurement(dqmc, model, K=4)

            # # Unequal time
            # # N = MonteCarlo.nslices(dqmc) # 10
            # l1s = [0, 4, 5, 6, 0]
            # l2s = [1, 5, 5, 7, MonteCarlo.nslices(dqmc)]
            # # Why is UTGreensMeasurement not defined?
            # UTG = MonteCarlo.UTGreensMeasurement
            # dqmc[:UTG1] = UTG(dqmc, model, slice1=l1s[1], slice2=l2s[1])
            # dqmc[:UTG2] = UTG(dqmc, model, slice1=l1s[2], slice2=l2s[2])
            # dqmc[:UTG3] = UTG(dqmc, model, slice1=l1s[3], slice2=l2s[3])
            # dqmc[:UTG4] = UTG(dqmc, model, slice1=l1s[4], slice2=l2s[4])
            # dqmc[:UTG5] = UTG(dqmc, model, slice1=l1s[5], slice2=l2s[5])
            # MonteCarlo.initialize_stack(dqmc, dqmc.ut_stack)

            # MonteCarlo.enable_benchmarks()

            @time run!(dqmc, verbose=false)
            
            # Absolute tolerance from Trotter decompositon
            atol = 2dqmc.p.delta_tau^2
            rtol = 2dqmc.p.delta_tau^2
            N = length(lattice(model))
        
            @info "Running ED"
            @time begin
                H = HamiltonMatrix(model)
            
                # G_DQMC is smaller because it doesn't differentiate between spin up/down
                @testset "Greens" begin
                    G_DQMC = mean(dqmc.measurements[:G])
                    occs = mean(dqmc.measurements[:Occs])                                   # measuring
                    # occs2 = mean(MonteCarlo.occupations(dqmc.measurements[:Greens]))            # wrapping
                    # occs3 = mean(MonteCarlo.OccupationMeasurement(dqmc.measurements[:Greens]))  # copying
                    G_ED = calculate_Greens_matrix(H, model.l, beta = dqmc.p.beta)
                    for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
                        @test isapprox(G_DQMC[i, j], G_ED[i, j], atol=atol, rtol=rtol)
                    end
                    for i in 1:size(G_DQMC, 1)
                        @test isapprox(occs[i],  1 - G_ED[i, i], atol=atol, rtol=rtol)
                        # @test isapprox(occs2[i], 1 - G_ED[i, i], atol=atol, rtol=rtol)
                        # @test isapprox(occs3[i], 1 - G_ED[i, i], atol=atol, rtol=rtol)
                    end
                end

                @testset "Charge Density Correlation" begin
                    CDC = mean(dqmc.measurements[:CDC])
                    ED_CDC = zeros(size(CDC))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:CDC], dqmc, model)
                        ED_CDC[dir] += expectation_value(
                            charge_density_correlation(trg, src),
                            H, beta = dqmc.p.beta, N_sites = N
                        )
                    end
                    @test ED_CDC/N ≈ CDC atol=atol rtol=rtol
                end

                @testset "Magnetization x" begin
                    Mx = mean(dqmc.measurements[:Mx])
                    for site in 1:length(Mx)
                        ED_Mx = expectation_value(m_x(site), H, beta = dqmc.p.beta, N_sites = N)
                        @test ED_Mx ≈ Mx[site] atol=atol rtol=rtol
                    end
                end
                @testset "Magnetization y" begin
                    My = mean(dqmc.measurements[:My])
                    for site in 1:length(My)
                        ED_My = expectation_value(m_y(site), H, beta = dqmc.p.beta, N_sites = N)
                        @test ED_My ≈ My[site] atol=atol rtol=rtol
                    end
                end
                @testset "Magnetization z" begin
                    Mz = mean(dqmc.measurements[:Mz])
                    ΔMz = std_error(dqmc.measurements[:Mz])
                    for site in 1:length(Mz)
                        ED_Mz = expectation_value(m_z(site), H, beta = dqmc.p.beta, N_sites = N)
                        @test ED_Mz ≈ Mz[site] atol=atol+ΔMz[site] rtol=rtol
                    end
                end

                @testset "Spin density correlation x" begin
                    SDCx = mean(dqmc.measurements[:SDCx])
                    ED_SDCx = zeros(ComplexF64, size(SDCx))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCx], dqmc, model)
                        ED_SDCx[dir] += expectation_value(
                            spin_density_correlation_x(trg, src),
                            H, beta = dqmc.p.beta, N_sites = N
                        )
                    end
                    @test ED_SDCx/N ≈ SDCx atol=atol rtol=rtol
                end
                @testset "Spin density correlation y" begin
                    SDCy = mean(dqmc.measurements[:SDCy])
                    ED_SDCy = zeros(ComplexF64, size(SDCy))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCy], dqmc, model)
                        ED_SDCy[dir] += expectation_value(
                            spin_density_correlation_y(trg, src),
                            H, beta = dqmc.p.beta, N_sites = N
                        )
                    end
                    @test ED_SDCy/N ≈ SDCy atol=atol rtol=rtol
                end
                @testset "Spin density correlation z" begin
                    SDCz = mean(dqmc.measurements[:SDCz])
                    ED_SDCz = zeros(ComplexF64, size(SDCz))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCz], dqmc, model)
                        ED_SDCz[dir] += expectation_value(
                            spin_density_correlation_z(trg, src),
                            H, beta = dqmc.p.beta, N_sites = N
                        )
                    end
                    @test ED_SDCz/N ≈ SDCz atol=atol rtol=rtol
                end

                @testset "Pairing Correlation" begin
                    PC = mean(dqmc.measurements[:PC])
                    ED_PC = zeros(ComplexF64, size(PC))
                    for (dir12, dir1, dir2, src1, trg1, src2, trg2) in 
                            MonteCarlo.EachLocalQuadByDistance{4}(dqmc, model)
                        ED_PC[dir12, dir1, dir2] += expectation_value(
                            pairing_correlation(src1, trg1, src2, trg2), 
                            H, beta = dqmc.p.beta, N_sites = N
                        )
                    end
                    # for (dir_idx, src1, src2) in MonteCarlo.getorder(mask)
                    #     for (i, trg1) in MonteCarlo.getorder(rsm, src1)
                    #         for (j, trg2) in MonteCarlo.getorder(rsm, src2)
                    #             ED_PC[dir_idx, i, j] += expectation_value(
                    #                 pairing_correlation(
                    #                     src1, trg1, src2, trg2
                    #                 ), H, beta = dqmc.p.beta, N_sites = N
                    #             )
                    #         end
                    #     end
                    # end
                    @test ED_PC/N ≈ PC atol=atol rtol=rtol
                    # println("PC")
                    # println(ED_PC/N, "  ", PC)
                end
                # @testset "Unequal Time Greens" begin
                # for (i, tau1, tau2) in zip(1:5, 0.1l1s, 0.1l2s)
                #         UTG = mean(dqmc.measurements[Symbol(:UTG, i)])
                #         # ΔUTG = std_error(dqmc.measurements[Symbol(:UTG, i)])
                #         M = size(UTG, 1)
                #         ED_UTG = calculate_Greens_matrix(H, tau2, tau1, model.l, beta=dqmc.p.beta)
                #         for k in 1:M, l in 1:M
                #             @test isapprox(UTG[k, l], ED_UTG[k, l], atol=atol, rtol=rtol)
                #         end
                #     end
                # end
            end
        end
    end
end
