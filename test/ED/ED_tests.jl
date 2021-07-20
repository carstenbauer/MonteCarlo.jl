using Random
include("ED.jl")


@testset "ED checks" begin
    void = State(0)
    up = State(1)
    down = State(2)
    updown = State(3)

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
    model = HubbardModelAttractive(2, 2, U = rand(), mu = rand(), t = rand()
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

            # Test G(t, t) = G(0, 0) = G
            UTG = expectation_value(
                s -> annihilate(s, site2, substate2),
                s -> create(s, site1, substate1),
                H, 0.1, 0.1, N_sites=model.l.sites
            )
            @test check(UTG, real(G), 1e-14, 0.0)
            UTG = expectation_value(
                s -> annihilate(s, site2, substate2),
                s -> create(s, site1, substate1),
                H, 0.7, 0.7, N_sites=model.l.sites
            )
            @test check(UTG, real(G), 1e-14, 0.0)
        end
    end

    # Test G(t, 0) + G(0, beta - t) = 0
    UTG1 = calculate_Greens_matrix(H, 0.7, 0.0, model.l, beta=1.0)
    UTG2 = calculate_Greens_matrix(H, 0.0, 1.0 - 0.7, model.l, beta=1.0)
    @test UTG1 ≈ -UTG2 atol=1e-13
end


@testset "Exact Greens comparison (ED, tiny systems)" begin
    # These are theoretically the same but their implementation differs on
    # some level. To make sure both are correct it makes sense to check both here.
    models = (
        HubbardModelRepulsive(2, 2, U = 0.0, t = 1.0),
        HubbardModelAttractive(2, 2, U = 0.0, mu = 1.0, t = 1.0)
    )

    @info "Exact Greens comparison (ED)"
    for model in models, beta in (1.0, 10.0)
        @testset "$(typeof(model))" begin
            Random.seed!(123)
            dqmc = DQMC(
                model, beta=beta, delta_tau = 0.1, safe_mult=5, recorder = Discarder, 
                thermalization = 1, sweeps = 2, measure_rate=1
            )
            @info "Running DQMC ($(typeof(model).name)) β=$(dqmc.parameters.beta)"

            dqmc[:G]    = greens_measurement(dqmc, model)
            l1s = [0, 3, 5, 7, 3, 0, MonteCarlo.nslices(dqmc), 0]
            l2s = [1, 7, 5, 2, 1, MonteCarlo.nslices(dqmc), 0, 0]
            for i in eachindex(l1s)
                dqmc[Symbol(:UTG, i)] = greens_measurement(dqmc, model, GreensAt(l2s[i], l1s[i]))
            end

            @time run!(dqmc, verbose=false)
            
            # error tolerance
            atol = 1e-13
            rtol = 1e-13
            N = length(lattice(model))

            # Direct calculation simialr to what DQMC should be doing
            T = Matrix(MonteCarlo.hopping_matrix(dqmc, model))
            # Doing an eigenvalue decomposition makes this pretty stable
            vals, U = eigen(exp(-T))
            D = Diagonal(vals)^(dqmc.parameters.beta)

            # G = I - U * inv(I + D) * adjoint(U)
            G = U * inv(I + D) * adjoint(U)
        
            @info "Running ED"
            @time begin
                H = HamiltonMatrix(model)
            
                # G_DQMC is smaller because it doesn't differentiate between spin up/down
                @testset "Greens" begin
                    G_DQMC = mean(dqmc.measurements[:G])
                    G_ED = calculate_Greens_matrix(H, model.l, beta = dqmc.parameters.beta)
                    @test check(G_DQMC, G_ED[1:size(G_DQMC, 1), 1:size(G_DQMC, 2)], atol, rtol)
                    @test check(G, G_ED[1:size(G_DQMC, 1), 1:size(G_DQMC, 2)], atol, rtol)
                    @test check(G_DQMC, G, atol, rtol)
                end

                for (i, tau1, tau2) in zip(eachindex(l1s), 0.1l1s, 0.1l2s)
                    UTG = mean(dqmc.measurements[Symbol(:UTG, i)])
                    M = size(UTG, 1)
                    ED_UTG = calculate_Greens_matrix(H, tau2, tau1, model.l, beta=dqmc.parameters.beta)

                    @testset "[$i] $tau1 -> $tau2" begin
                        for k in 1:M, l in 1:M
                            @test check(UTG[k, l], ED_UTG[k, l], atol, rtol)
                        end
                    end
                end
            end

        end
    end
end


@testset "Repulsive/Attractive Hubbard Model (ED)" begin
    models = (
        HubbardModelRepulsive(2, 2, U = 1.0, t = 1.0),
        HubbardModelAttractive(2, 2, U = 1.0, mu = 1.0, t = 1.0)
    )

    for model in models
        @testset "$(typeof(model))" begin
            Random.seed!(123)
            dqmc = DQMC(
                model, beta=1.0, delta_tau = 0.1, safe_mult=5, recorder = Discarder, 
                thermalization = 10_000, sweeps = 10_000, print_rate=1000,
                # scheduler = AdaptiveScheduler(
                #     (LocalSweep(10), Adaptive(),), (GlobalShuffle(), GlobalFlip())
                # )
            )
            @info "Running DQMC ($(typeof(model).name)) β=$(dqmc.parameters.beta), 10k + 10k sweeps"

            dqmc[:G]    = greens_measurement(dqmc, model)
            dqmc[:E]    = total_energy(dqmc, model)
            dqmc[:Occs] = occupation(dqmc, model)
            dqmc[:CDC]  = charge_density_correlation(dqmc, model)
            dqmc[:Mx]   = magnetization(dqmc, model, :x)
            dqmc[:My]   = magnetization(dqmc, model, :y)
            dqmc[:Mz]   = magnetization(dqmc, model, :z)
            dqmc[:SDCx] = spin_density_correlation(dqmc, model, :x)
            dqmc[:SDCy] = spin_density_correlation(dqmc, model, :y)
            dqmc[:SDCz] = spin_density_correlation(dqmc, model, :z)
            dqmc[:PC]   = pairing_correlation(dqmc, model, K=4)

            # Unequal time
            l1s = [0, 3, 5, 7, 3, 0]
            l2s = [1, 7, 5, 2, 1, MonteCarlo.nslices(dqmc)]
            # l1s = [0, 3, 5, 0]
            # l2s = [1, 7, 5, MonteCarlo.nslices(dqmc)]
            dqmc[:UTG1] = greens_measurement(dqmc, model, GreensAt(l2s[1], l1s[1]))
            dqmc[:UTG2] = greens_measurement(dqmc, model, GreensAt(l2s[2], l1s[2]))
            dqmc[:UTG3] = greens_measurement(dqmc, model, GreensAt(l2s[3], l1s[3]))
            dqmc[:UTG4] = greens_measurement(dqmc, model, GreensAt(l2s[4], l1s[4]))
            dqmc[:UTG5] = greens_measurement(dqmc, model, GreensAt(l2s[5], l1s[5]))
            dqmc[:UTG6] = greens_measurement(dqmc, model, GreensAt(l2s[6], l1s[6]))

            dqmc[:CDS]  = charge_density_susceptibility(dqmc, model)
            dqmc[:SDSx] = spin_density_susceptibility(dqmc, model, :x)
            dqmc[:SDSy] = spin_density_susceptibility(dqmc, model, :y)
            dqmc[:SDSz] = spin_density_susceptibility(dqmc, model, :z)
            dqmc[:PS]   = pairing_susceptibility(dqmc, model, K=4)
            dqmc[:CCS]  = current_current_susceptibility(dqmc, model, K=4)

            # MonteCarlo.enable_benchmarks()

            @time run!(dqmc, verbose=!true)
            
            # Absolute tolerance from Trotter decompositon
            atol = 2.5dqmc.parameters.delta_tau^2
            rtol = 2dqmc.parameters.delta_tau^2
            N = length(lattice(model))
        
            @info "Running ED"
            @time begin
                H = HamiltonMatrix(model)

                @testset "(total) energy" begin
                    dqmc_E = mean(dqmc[:E])
                    ED_E = energy(H, beta = dqmc.parameters.beta)
                    @test dqmc_E ≈ ED_E atol=atol rtol=rtol
                end
            
                # G_DQMC is smaller because it doesn't differentiate between spin up/down
                @testset "Greens" begin
                    G_DQMC = mean(dqmc.measurements[:G])
                    occs = mean(dqmc.measurements[:Occs])                                   # measuring
                    # occs2 = mean(MonteCarlo.occupations(dqmc.measurements[:Greens]))            # wrapping
                    # occs3 = mean(MonteCarlo.OccupationMeasurement(dqmc.measurements[:Greens]))  # copying
                    G_ED = calculate_Greens_matrix(H, model.l, beta = dqmc.parameters.beta)
                    for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
                        @test check(G_DQMC[i, j], G_ED[i, j], atol, rtol)
                    end
                    for i in 1:size(G_DQMC, 1)
                        @test check(occs[i],  1 - G_ED[i, i], atol, rtol)
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
                            H, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_CDC/N, CDC, atol, rtol)
                end

                @testset "Magnetization x" begin
                    Mx = mean(dqmc.measurements[:Mx])
                    for site in 1:length(Mx)
                        ED_Mx = expectation_value(
                            m_x(site), H, beta = dqmc.parameters.beta, N_sites = N
                        )
                        @test check(ED_Mx, Mx[site], atol, rtol)
                    end
                end
                @testset "Magnetization y" begin
                    My = mean(dqmc.measurements[:My])
                    for site in 1:length(My)
                        ED_My = expectation_value(
                            m_y(site), H, beta = dqmc.parameters.beta, N_sites = N
                        ) |> imag
                        @test check(ED_My, My[site], atol, rtol)
                    end
                end
                @testset "Magnetization z" begin
                    Mz = mean(dqmc.measurements[:Mz])
                    for site in 1:length(Mz)
                        ED_Mz = expectation_value(
                            m_z(site), H, beta = dqmc.parameters.beta, N_sites = N
                        )
                        @test check(ED_Mz, Mz[site], atol, rtol)
                    end
                end

                @testset "Spin density correlation x" begin
                    SDCx = mean(dqmc.measurements[:SDCx])
                    ED_SDCx = zeros(ComplexF64, size(SDCx))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCx], dqmc, model)
                        ED_SDCx[dir] += expectation_value(
                            spin_density_correlation_x(trg, src),
                            H, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDCx/N, SDCx, atol, rtol)
                end
                @testset "Spin density correlation y" begin
                    SDCy = mean(dqmc.measurements[:SDCy])
                    ED_SDCy = zeros(ComplexF64, size(SDCy))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCy], dqmc, model)
                        ED_SDCy[dir] += expectation_value(
                            spin_density_correlation_y(trg, src),
                            H, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDCy/N, SDCy, atol, rtol)
                end
                @testset "Spin density correlation z" begin
                    SDCz = mean(dqmc.measurements[:SDCz])
                    ED_SDCz = zeros(ComplexF64, size(SDCz))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDCz], dqmc, model)
                        ED_SDCz[dir] += expectation_value(
                            spin_density_correlation_z(trg, src),
                            H, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDCz/N, SDCz, atol, rtol)
                end

                @testset "Pairing Correlation" begin
                    PC = mean(dqmc.measurements[:PC])
                    ED_PC = zeros(ComplexF64, size(PC))
                    for (dirs, src1, trg1, src2, trg2) in 
                            MonteCarlo.EachLocalQuadByDistance{4}(dqmc, model)
                        ED_PC[dirs] += expectation_value(
                            pairing_correlation(src1, trg1, src2, trg2), 
                            H, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_PC/N, PC, atol, rtol)
                end

                ################################################################
                ### Unequal Time
                ################################################################

                @testset "Unequal Time Greens" begin
                    dt = dqmc.parameters.delta_tau
                    for (i, tau1, tau2) in zip(eachindex(l1s), dt * l1s, dt * l2s)
                        UTG = mean(dqmc.measurements[Symbol(:UTG, i)])
                        # ΔUTG = std_error(dqmc.measurements[Symbol(:UTG, i)])
                        M = size(UTG, 1)
                        ED_UTG = calculate_Greens_matrix(H, tau2, tau1, model.l, beta=dqmc.parameters.beta)

                        @testset "[$i] $tau1 -> $tau2" begin
                            for k in 1:M, l in 1:M
                                @test check(UTG[k, l], ED_UTG[k, l], atol, rtol)
                            end
                            # println("[$i] $tau1 -> $tau2")
                            # display(UTG)
                            # println()
                            # display(ED_UTG)
                            # println()
                        end
                    end
                end

                # CDC
                @testset "Charge Density Susceptibility" begin
                    CDS = mean(dqmc.measurements[:CDS])
                    ED_CDS = zeros(size(CDS))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:CDS], dqmc, model)
                        ED_CDS[dir] += expectation_value_integrated(
                            number_operator(trg), number_operator(src), H, 
                            step = dqmc.parameters.delta_tau, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_CDS/N, CDS, atol, rtol)
                end
                
                # SDS
                @testset "Spin density Susceptibility x" begin
                    SDSx = mean(dqmc.measurements[:SDSx])
                    ED_SDSx = zeros(ComplexF64, size(SDSx))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDSx], dqmc, model)
                        ED_SDSx[dir] += expectation_value_integrated(
                            m_x(trg), m_x(src), H, step = dqmc.parameters.delta_tau, 
                            beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDSx/N, SDSx, atol, rtol)
                end
                @testset "Spin density Susceptibility y" begin
                    SDSy = mean(dqmc.measurements[:SDSy])
                    ED_SDSy = zeros(ComplexF64, size(SDSy))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDSy], dqmc, model)
                        ED_SDSy[dir] += expectation_value_integrated(
                            m_y(trg), m_y(src), H, step = dqmc.parameters.delta_tau, 
                            beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDSy/N, SDSy, atol, rtol)
                end
                @testset "Spin density Susceptibility z" begin
                    SDSz = mean(dqmc.measurements[:SDSz])
                    ED_SDSz = zeros(ComplexF64, size(SDSz))
                    for (dir, src, trg) in MonteCarlo.lattice_iterator(dqmc[:SDSz], dqmc, model)
                        ED_SDSz[dir] += expectation_value_integrated(
                            m_z(trg), m_z(src), H, step = dqmc.parameters.delta_tau, 
                            beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_SDSz/N, SDSz, atol, rtol)
                end

                @testset "Pairing Susceptibility" begin
                    PS = mean(dqmc.measurements[:PS])
                    ED_PS = zeros(Float64, size(PS))
                    for (dirs, src1, trg1, src2, trg2) in 
                            MonteCarlo.EachLocalQuadByDistance{4}(dqmc, model)
                        ED_PS[dirs] += expectation_value_integrated(
                            state -> begin
                                sign1, _state = annihilate(state, trg1, DOWN)
                                sign2, _state = annihilate(_state, src1, UP)
                                p = sign1*sign2
                                p, _state
                            end,
                            state -> begin
                                sign1, _state = create(state, src2, UP)
                                sign2, _state = create(_state, trg2, DOWN)
                                p = sign1*sign2
                                p, _state
                            end,
                            H, step = dqmc.parameters.delta_tau, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_PS/N, PS, atol, rtol)
                end
                
                @testset "Current Current Susceptibility" begin
                    CCS = mean(dqmc.measurements[:CCS])
                    ED_CCS = zeros(Float64, size(CCS))
                    T = dqmc.stack.hopping_matrix
                    for (dirs, src1, trg1, src2, trg2) in 
                            MonteCarlo.EachLocalQuadBySyncedDistance{4}(dqmc, model)
                        ED_CCS[dirs] -= expectation_value_integrated(
                            # actually the order of this doesn't seem to matter
                            current_density(src2, trg2, T), 
                            current_density(src1, trg1, T),
                            H, step = dqmc.parameters.delta_tau, beta = dqmc.parameters.beta, N_sites = N
                        )
                    end
                    @test check(ED_CCS/N, CCS, atol, rtol)
                end
            end
        end
    end
end