@testset "Lookup tables" begin
    param = MonteCarlo.DQMCParameters(beta = 1.0)
    model = HubbardModelRepulsive(2, 2)
    field = MagneticGHQField(param, model)

    @test field.α == sqrt(-0.5 * 0.1 * model.U)
    
    # See ALF Documentation for the formulas of η and γ
    @test field.γ[3 + -2] ≈ Float64(BigFloat(1) - sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[3 + -1] ≈ Float64(BigFloat(1) + sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[3 + +1] ≈ Float64(BigFloat(1) + sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    @test field.γ[3 + +2] ≈ Float64(BigFloat(1) - sqrt(BigFloat(6)) / BigFloat(3))  rtol=1e-15 
    
    @test field.η[3 + -2] ≈ Float64(- sqrt(BigFloat(2) * (BigFloat(3) + sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[3 + -1] ≈ Float64(- sqrt(BigFloat(2) * (BigFloat(3) - sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[3 + +1] ≈ Float64(+ sqrt(BigFloat(2) * (BigFloat(3) - sqrt(BigFloat(6)))))  rtol=1e-15
    @test field.η[3 + +2] ≈ Float64(+ sqrt(BigFloat(2) * (BigFloat(3) + sqrt(BigFloat(6)))))  rtol=1e-15

    @test field.choices[3 + -2, :] == [-1, +1, +2]
    @test field.choices[3 + -1, :] == [-2, +1, +2]
    @test field.choices[3 + +1, :] == [-2, -1, +2]
    @test field.choices[3 + +2, :] == [-2, -1, +1]
end

@testset "Exact Greens test" begin
    models = (
        HubbardModelRepulsive(l = SquareLattice(5), U = 0.0),
        # HubbardModelAttractive(l = SquareLattice(5), U = 0.0)
    )
    for model in models
        for beta in (1.0, 8.9)
            @testset "$(typeof(model).name.name) β=$(beta)" begin
                Random.seed!(123)
                dqmc = DQMC(
                    model, beta=beta, delta_tau = 0.1, safe_mult=5, recorder = Discarder(), 
                    thermalization = 1, sweeps = 2, measure_rate = 1, field = MagneticGHQField
                )
                # @info "Running DQMC ($(typeof(model).name.name)) β=$(dqmc.parameters.beta)"

                dqmc[:G] = greens_measurement(dqmc, model)
                run!(dqmc, verbose=false)
                
                # error tolerance
                atol = 1e-12
                rtol = 1e-12
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
end

include("ED/ED.jl")

@testset "ED Comparison" begin
    models = (
        HubbardModelRepulsive(l = SquareLattice(2), U = -1.0),
        # HubbardModelAttractive(l = SquareLattice(2), U = 1.0)
    )
    for model in models
        @testset "$(typeof(model).name.name)" begin
            dqmc = DQMC(
                model, beta=1.0, delta_tau = 0.1, safe_mult=5, recorder = Discarder(), 
                thermalization = 10_000, sweeps = 10_000, print_rate=1000, 
                field = MagneticGHQField
            )
            print(
                "  Running DQMC ($(typeof(model).name.name)) " * 
                "β=$(dqmc.parameters.beta), 10k + 10k sweeps\n    "
            )

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
            dqmc[:PC]   = pairing_correlation(
                dqmc, model, lattice_iterator = EachLocalQuadByDistance(1:4)
            )

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
            dqmc[:PS]   = pairing_susceptibility(
                dqmc, model, lattice_iterator = EachLocalQuadByDistance(1:4)
            )
            dqmc[:CCS]  = current_current_susceptibility(
                dqmc, model, lattice_iterator = EachLocalQuadByDistance(1:4)
            )

            # MonteCarlo.enable_benchmarks()

            @time run!(dqmc, verbose = false)
            
            # Absolute tolerance from Trotter decompositon
            atol = 2.5dqmc.parameters.delta_tau^2
            rtol = 2dqmc.parameters.delta_tau^2
            N = length(lattice(model))

            print("    Running ED and checking (tolerance: $atol, $(100rtol)%)\n    ")
            @time begin
                H = HamiltonMatrix(model)

                @testset "(total) energy" begin
                    dqmc_E = mean(dqmc[:E])
                    ED_E = energy(H, beta = dqmc.parameters.beta)
                    @test dqmc_E ≈ ED_E atol=atol rtol=rtol
                end
            
                @testset "Greens" begin
                    G_DQMC = mean(dqmc.measurements[:G])
                    occs = mean(dqmc.measurements[:Occs])
                    G_ED = calculate_Greens_matrix(H, model.l, beta = dqmc.parameters.beta)
                    for i in 1:size(G_DQMC, 1), j in 1:size(G_DQMC, 2)
                        @test check(G_DQMC[i, j], G_ED[i, j], atol, rtol)
                    end
                    for i in 1:size(G_DQMC, 1)
                        @test check(occs[i],  1 - G_ED[i, i], atol, rtol)
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
                            MonteCarlo.lattice_iterator(dqmc[:PC], dqmc, model)
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
                        ED_UTG = calculate_Greens_matrix(
                            H, tau2, tau1, model.l, beta=dqmc.parameters.beta
                        )

                        @testset "[$i] $tau1 -> $tau2" begin
                            for k in 1:M, l in 1:M
                                @test check(UTG[k, l], ED_UTG[k, l], atol, rtol)
                            end
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
                            MonteCarlo.lattice_iterator(dqmc[:PS], dqmc, model)
                        ED_PS[dirs] += expectation_value_integrated(
                            state -> begin
                                sign1, _state  = annihilate(state, trg1, DOWN)
                                sign2, _state1 = annihilate(_state, src1, UP)
                                sign3, _state  = create(state, src1, UP)
                                sign4, _state2 = create(_state, trg1, DOWN)
                                (sign1*sign2, sign3*sign4), (_state1, _state2)
                            end,
                            state -> begin
                                sign1, _state  = create(state, src2, UP)
                                sign2, _state1 = create(_state, trg2, DOWN)
                                sign3, _state  = annihilate(state, trg2, DOWN)
                                sign4, _state2 = annihilate(_state, src2, UP)
                                (sign1*sign2, sign3*sign4), (_state1, _state2)
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
                            MonteCarlo.lattice_iterator(dqmc[:CCS], dqmc, model)
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