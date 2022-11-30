using BinningAnalysis

function sanitized_ratio(X, ref, atol = 1e-6)
    map(X, ref) do x, r
        ifelse(abs(x) < atol, 0, (x-r)/r)
    end
end

@testset "Checkerboard Decomposition" begin
    lattices = [
        SquareLattice(2), # check case where reverse bond = bond in second grou
        SquareLattice(3), # check odd
        SquareLattice(4), # check standard
        Honeycomb(2),     # check basis
        SquareLattice(6),     # check larger lattices
        TriangularLattice(5), # (where this becomes an approximation)
        Honeycomb(4)
    ]
    # Trotter error should be around Δτ^2 (somehow scaling with [X, Y] and G)

    # These are based on threshholds I'm reaching (as Δτ * tol). 
    # This is not the expected scaling :(
    # Note that the tiny tolerances do not scale with Δτ (float precision)
    # Note that triangular scales harder than usual (0.36 -> 0.0058)
    tols = (1e-13, 0.6, 1e-13, 1e-13, 1.5, 4.0, 2.0)
    for (tol, l) in zip(tols, lattices)
        atol = max.(1e-9, tol)
        rtol = tol

        @testset "$l" begin
            pos = collect(MonteCarlo.positions(l))
            groups = MonteCarlo.build_checkerboard(l)
            wrap = MonteCarlo.generate_combinations(l)
            bs = unique(map(b -> (b.from, b.to), bonds(l, Val(true))))
            checklist = fill(false, length(bs))
            
            for group in vcat(groups...)
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
                
                x0, y0 = MonteCarlo._ind2sub(l, src)
                for (src, trg) in group
                    # mark bond as used
                    idx = findfirst(p -> p == (src, trg), bs)::Int
                    @assert !checklist[idx]
                    checklist[idx] = true
                    idx = findfirst(p -> p == (trg, src), bs)::Int
                    @assert !checklist[idx]
                    checklist[idx] = true
                end
            end

            # all bonds used
            @test all(checklist)

            # no duplicates
            all_pairs = vcat(groups...)
            @test allunique(all_pairs)

            # Scaling with checkerboard is O(Δτ) ~ 0.1Δτ

            m = HubbardModel(l, mu = 0.3, t = 0.7, U = 0.0)
            mc = DQMC(
                m, beta = 24.0, delta_tau = 0.1, checkerboard = true, 
                thermalization = 1, sweeps = 2, measure_rate = 1
            )
            mc[:G] = greens_measurement(mc, m)
            run!(mc, verbose = false)

            # Direct calculation, see flavortests_DQMC / testfunctions
            G = analytic_greens(mc)
            # println(l.unitcell.name)
            # println(extrema(sanitized_ratio(mean(mc[:G]), G)))
            # println(norm(mean(mc[:G]) - G))
            @test mean(mc[:G]) ≈ G() atol = 1e-6 rtol = 0.1rtol

            # TODO: test time displaced
            # TODO: test d²(diag(Gl0)) / dl² > 0
            @testset "Time Displaced Greens" begin
                MonteCarlo.initialize_stack(mc, mc.ut_stack)
                vals = Float64[]
                for l in 0:MonteCarlo.nslices(mc)
                    g = MonteCarlo.greens(mc, l, 0)
                    # println(0.1rtol)
                    @test g ≈ G(l, 0) atol = 0.01 rtol = 0.1rtol
                    # println(norm(g - G(l, 0)), "   ", norm(g), "   ", norm(G(l, 0)))
                    push!(vals, mean(diag(g)))
                end
                # positive second derivative
                @test all((vals[3:end] .- 2 * vals[2:end-1] .+ vals[1:end-2]) .> 0)
            end


            m = HubbardModel(l, mu = -0.3, t = 0.7, U = 0.0)
            mc = DQMC(
                m, beta = 24.0, delta_tau = 0.01, checkerboard = true, 
                thermalization = 1, sweeps = 2, measure_rate = 1
            )
            mc[:G] = greens_measurement(mc, m)
            run!(mc, verbose = false)
            G = analytic_greens(mc)
            # println(extrema(sanitized_ratio(mean(mc[:G]), G)))
            # println(norm(mean(mc[:G]) - G))
            # println()
            @test mean(mc[:G]) ≈ G() atol = 1e-6 rtol = 0.01rtol

            @testset "Time Displaced Greens" begin
                MonteCarlo.initialize_stack(mc, mc.ut_stack)
                vals = Float64[]
                for l in 0:MonteCarlo.nslices(mc)
                    g = MonteCarlo.greens(mc, l, 0)
                    # println(0.1rtol)
                    @test g ≈ G(l, 0) atol = 0.001 rtol = 0.02rtol # errors too big?
                    # println(norm(g - G(l, 0)), "   ", norm(g), "   ", norm(G(l, 0)))
                    push!(vals, mean(diag(g)))
                end
                # positive second derivative
                @test all((vals[3:end] .- 2 * vals[2:end-1] .+ vals[1:end-2]) .> 0)
            end
        end
    end
end