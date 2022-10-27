using BinningAnalysis

@testset "Square lattice" begin
    # TODO:
    # make these test properties for different lattices:
    # - [x] no duplication (first can have both src -> trg and trg -> src though)
    # - [] no missing bonds
    # - [x] no douplicate attachments to the same site
    # - [x] equal bond direction within a group
    # - [] no staggering within a group
    lattices = [
        SquareLattice(2),
        SquareLattice(3),
        SquareLattice(4),
        Honeycomb(2),
        SquareLattice(6),
        TriangularLattice(5),
        Honeycomb(4)
    ]
    rtols = (1e-11, 1e-3, 1e-14, 1e-15, 1e-3, 1e-3, 1e-3)
    for (rtol, l) in zip(rtols, lattices)
        @testset "$l" begin
            pos = collect(MonteCarlo.positions(l))
            
            groups = MonteCarlo.build_checkerboard(l)
            wrap = MonteCarlo.generate_combinations(l)
            for group in groups
                # no more than one bond per site in each group
                sites = vcat(first.(group), last.(group))
                @test allunique(sites)

                # check if all bond directions match
                src, trg = first(group)
                dir = pos[trg] .- pos[src]
                perp = [dir[2], -dir[1]]
                @test mapreduce((a, b) -> a && b, group) do (src, trg)
                    dir = pos[trg] .- pos[src]
                    for shift in wrap
                        if abs(dot(dir .+ shift, perp)) < 1e-6
                            return true
                        end
                    end
                    return false
                end
            end

            # no duplicates
            all_pairs = vcat(groups...)
            @test allunique(all_pairs)

            m = HubbardModel(l, mu = 1.3, t = 0.3)
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
                @test getfield(mc1.stack, fullname) ≈ Matrix(getfield(mc2.stack, fullname)) atol = 1e-14 rtol = rtol
            end

            # verify working greens matrix
            @test mc1.stack.greens ≈ mc2.stack.greens atol = 1e-6 rtol = rtol

            # verify measurements
            for key in (:G, :CDC, :CCS)
                vals1 = mc1[key].observable.x
                vals2 = mc2[key].observable.x
                if Base.Sys.isapple()
                    println(key)
                    for delta in vals1 .- vals2
                        println(extrema(abs.(delta)))
                    end
                end
                @test all(isapprox.(vals1, vals2, atol = 1e-6, rtol = rtol))
            end
        end
    end
end