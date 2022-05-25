using MonteCarlo: lattice_vectors

@testset "Lattices" begin
    @testset "Chain" begin
        l = Chain(4)

        @test length(l) == 4
        @test size(l) == (4,)
        @test lattice_vectors(l) == ([1.0],)
        @test eachindex(l) == 1:4

        ps = collect(positions(l))
        @test size(ps) == (1, 4) 
        @test ps[:] == [[1.0], [2.0], [3.0], [4.0]]

        bs = collect(bonds(l))
        @test MonteCarlo.from.(bs) == [1,2,3,4]
        @test MonteCarlo.to.(bs) == [2,3,4,1]
        @test MonteCarlo.label.(bs) == [1,1,1,1]

        bs = collect(bonds(l, Val(true)))
        @test MonteCarlo.from.(bs) == [1,1,2,2,3,3,4,4]
        @test MonteCarlo.to.(bs) == [2,4,3,1,4,2,1,3]
        @test MonteCarlo.label.(bs) == ones(8)

        bs = collect(bonds(l, 3))
        @test MonteCarlo.from.(bs) == [3,3]
        @test MonteCarlo.to.(bs) == [4,2]
        @test MonteCarlo.label.(bs) == ones(2)
    end

    @testset "Square" begin
        l = SquareLattice(3)

        @test length(l) == 9
        @test size(l) == (3,3)
        @test lattice_vectors(l) == ([1.0, 0.0], [0.0, 1.0])
        @test eachindex(l) == 1:9

        ps = collect(positions(l))
        @test size(ps) == (1, 3, 3) 
        @test ps[:] == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]]

        bs = collect(bonds(l))
        @test MonteCarlo.from.(bs) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
        @test MonteCarlo.to.(bs) == [2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 4, 9, 8, 1, 9, 2, 7, 3]
        @test MonteCarlo.label.(bs) == ones(18)

        bs = collect(bonds(l, Val(true)))
        @test MonteCarlo.from.(bs) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]
        @test MonteCarlo.to.(bs) == [2, 4, 3, 7, 3, 5, 1, 8, 1, 6, 2, 9, 5, 7, 6, 1, 6, 8, 4, 2, 4, 9, 5, 3, 8, 1, 9, 4, 9, 2, 7, 5, 7, 3, 8, 6]
        @test MonteCarlo.label.(bs) == ones(36)

        bs = collect(bonds(l, 3))
        @test MonteCarlo.from.(bs) == [3,3,3,3]
        @test MonteCarlo.to.(bs) == [1,6,2,9]
        @test MonteCarlo.label.(bs) == ones(4)
    end

    @testset "Cubic" begin
        l = CubicLattice(2)

        @test length(l) == 8
        @test size(l) == (2,2,2)
        @test lattice_vectors(l) == ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        @test eachindex(l) == 1:8

        ps = collect(positions(l))
        @test size(ps) == (1, 2, 2, 2) 
        @test ps[:] == [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 2.0], [1.0, 2.0, 2.0], [2.0, 2.0, 2.0]]

        bs = collect(bonds(l))
        @test MonteCarlo.from.(bs) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
        @test MonteCarlo.to.(bs) == [2, 3, 5, 1, 4, 6, 4, 1, 7, 3, 2, 8, 6, 7, 1, 5, 8, 2, 8, 5, 3, 7, 6, 4]
        @test MonteCarlo.label.(bs) == ones(24)

        bs = collect(bonds(l, Val(true)))
        @test MonteCarlo.from.(bs) == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
        @test MonteCarlo.to.(bs) == [2, 3, 5, 2, 3, 5, 1, 4, 6, 1, 4, 6, 4, 1, 7, 4, 1, 7, 3, 2, 8, 3, 2, 8, 6, 7, 1, 6, 7, 1, 5, 8, 2, 5, 8, 2, 8, 5, 3, 8, 5, 3, 7, 6, 4, 7, 6, 4]
        @test MonteCarlo.label.(bs) == ones(48)

        bs = collect(bonds(l, 3))
        @test MonteCarlo.from.(bs) == [3,3,3,3,3,3]
        @test MonteCarlo.to.(bs) == [4,1,7,4,1,7]
        @test MonteCarlo.label.(bs) == ones(6)
    end

    @testset "Honeycomb" begin
        l = Honeycomb(2)

        @test length(l) == 8
        @test size(l) == (2,2)
        @test lattice_vectors(l) == ([0.8660254037844386, -0.5], [0.8660254037844386, 0.5])
        @test eachindex(l) == 1:8

        ps = collect(positions(l))
        @test size(ps) == (2, 2, 2) 
        @test ps[:] == [[1.7320508075688772, 0.0], [2.309401076758503, 0.0], [2.598076211353316, -0.5], [3.1754264805429413, -0.5], [2.598076211353316, 0.5], [3.1754264805429417, 0.5], [3.4641016151377544, 0.0], [4.04145188432738, 0.0]]

        bs = collect(bonds(l))
        @test MonteCarlo.from.(bs) == [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7]
        @test MonteCarlo.to.(bs) == [2, 4, 6, 4, 2, 8, 6, 8, 2, 8, 6, 4]
        @test MonteCarlo.label.(bs) == ones(12)

        bs = collect(bonds(l, Val(true)))
        @test MonteCarlo.from.(bs) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
        @test MonteCarlo.to.(bs) == [2, 4, 6, 1, 3, 5, 4, 2, 8, 3, 1, 7, 6, 8, 2, 5, 7, 1, 8, 6, 4, 7, 5, 3]
        @test MonteCarlo.label.(bs) == ones(24)

        bs = collect(bonds(l, 3))
        @test MonteCarlo.from.(bs) == [3,3,3]
        @test MonteCarlo.to.(bs) == [4,2,8]
        @test MonteCarlo.label.(bs) == ones(3)
    end
end

using MonteCarlo: directed_norm

@testset "Lattice Iterators" begin
    # Set up a few test models/DQMCs    
    m = HubbardModel(3, 3)
    dqmc1 = DQMC(m, beta=1.0)
    
    m = HubbardModel(10, 1, U = -1.0)
    dqmc2 = DQMC(m, beta=1.0)

    l = Honeycomb(2, 3)
    m = HubbardModel(l)
    dqmc3 = DQMC(m, beta=1.0)

    dqmcs = (dqmc1, dqmc2, dqmc3)


    @testset "Meta" begin
        for dqmc in dqmcs
            dirs = directions(lattice(dqmc))
            for i in 2:length(dirs)
                @test norm(dirs[i-1]) < norm(dirs[i]) + 1e-5
            end
        end

        dirs = [[cos(x), sin(x)] for x in range(0, 2pi-10eps(2pi), length=10)]
        norms = directed_norm.(dirs, 1e-6)
        for i in 2:length(norms)
            @test norms[i-1] < norms[i] < norms[i-1] + 1e-5
            @test 1.0 - 1e-5 < norms[i] < 1.0 + 1e-5
        end
    end

    @testset "EachSiteAndFlavor" begin
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachSiteAndFlavor(dqmc), lattice(dqmc))
            Nsites = length(lattice(dqmc))
            Nflavors = MonteCarlo.nflavors(MonteCarlo.field(dqmc))
            @test collect(iter) == 1:Nsites*Nflavors
            @test length(iter) == Nsites*Nflavors
            @test eltype(iter) == Int64
            @test Base.IteratorSize(EachSiteAndFlavor) == Base.HasLength()
            @test Base.IteratorEltype(EachSiteAndFlavor) == Base.HasEltype()
        end
    end

    @testset "EachSite" begin
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachSite(), lattice(dqmc))
            Nsites = length(lattice(dqmc))
            @test collect(iter) == 1:Nsites
            @test length(iter) == Nsites
            @test eltype(iter) == Int64
            @test Base.IteratorSize(EachSite) == Base.HasLength()
            @test Base.IteratorEltype(EachSite) == Base.HasEltype()
        end
    end

    @testset "OnSite" begin # TODO FAILS
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(OnSite(), lattice(dqmc))
            Nsites = length(lattice(dqmc))
            @test collect(iter) == collect(zip(1:Nsites, 1:Nsites, 1:Nsites))
            @test length(iter) == Nsites
            @test eltype(iter) == NTuple{3, Int} # TODO ANY
            @test Base.IteratorSize(OnSite) == Base.HasLength()
            @test Base.IteratorEltype(OnSite) == Base.HasEltype()
        end
    end

    @testset "EachSitePair" begin
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachSitePair(), lattice(dqmc))
            Nsites = length(lattice(dqmc))
            @test collect(iter) == [(i, j) for i in 1:Nsites for j in 1:Nsites]
            @test length(iter) == Nsites^2
            @test eltype(iter) == Tuple{Int64, Int64}
            @test Base.IteratorSize(EachSitePair) == Base.HasLength()
            @test Base.IteratorEltype(EachSitePair) == Base.HasEltype()
        end
    end

    @testset "EachSitePairByDistance" begin # TODO FAILS
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachSitePairByDistance(), lattice(dqmc))
            Nsites = length(lattice(dqmc))
            @test length(iter) == Nsites^2
            @test eltype(iter) == NTuple{3, Int64}
            @test Base.IteratorSize(EachSitePairByDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachSitePairByDistance) == Base.HasEltype()

            dirs = directions(Bravais(lattice(dqmc)))
            pos = collect(positions(lattice(dqmc)))
            uc_pos = positions(unitcell(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(lattice(dqmc))
            
            B = length(lattice(dqmc).unitcell.sites)
            ind2sub = CartesianIndices((B, B, length(dirs)))

            # Let's summarize these tests...
            check = true
            N = 0

            for (combined, src, trg) in iter
                N += 1
                b1, b2, idx = Tuple(ind2sub[combined])
                # raw distance between points - wrapped Bravais distance - raw uc distance
                _d = pos[src] - pos[trg] - dirs[idx] - uc_pos[b1] + uc_pos[b2]
                
                # Redo wrapping around periodic bounds. Note that the wrapping
                # in the Bravais lattice may differ from what would happen with
                # unit cells. Because of that wrapping on just 
                # (pos[src] - pos[trg]) would fail - but it's fine for the full
                # difference vector
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                
                # Some slack for float precision
                check = check && all(d .< 1e-15)
            end

            @test length(iter) == N
            @test check
        end
    end

    @testset "EachLocalQuadByDistance" begin
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachLocalQuadByDistance(6), lattice(dqmc))
            dir2srctrg = lattice(dqmc)[:dir2srctrg]
            iter_length = mapreduce(dir -> length(dir2srctrg[dir]), +, iter.iter.directions)^2
            @test length(iter) == iter_length
            @test eltype(iter) == NTuple{5, Int64}
            @test Base.IteratorSize(EachLocalQuadByDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachLocalQuadByDistance) == Base.HasEltype()

            dirs = directions(Bravais(lattice(dqmc)))
            full_dirs = directions(lattice(dqmc))
            pos = collect(positions(Bravais(lattice(dqmc))))
            uc_pos = positions(unitcell(lattice(dqmc)))
            full_pos = collect(positions(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(lattice(dqmc))

            # Let's summarize these tests...
            check12 = true
            check1 = true
            check2 = true

            B = length(lattice(dqmc).unitcell.sites)
            idxs = CartesianIndices((B, B, length(dirs), 6, 6))
            N = 0

            for (lin, src1, trg1, src2, trg2) in iter
                N += 1
                uc1, uc2, idx12, idx1, idx2 = Tuple(idxs[lin])

                # src1 -- idx12 -- src2
                _d = full_pos[src1] - full_pos[src2] - dirs[idx12] + uc_pos[uc2] - uc_pos[uc1]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check12 = check12 && all(d .< 1e-15)

                # src1 -- idx1 -- trg1
                _d = full_pos[src1] - full_pos[trg1]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check1 = check1 && (full_dirs[idx1] ≈ d)

                # src2 -- idx2 -- trg2
                _d = full_pos[src2] - full_pos[trg2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check2 = check2 && (full_dirs[idx2] ≈ d)
            end

            @test length(iter) == N
            @test check12
            @test check1
            @test check2
        end
    end

    @testset "EachLocalQuadBySyncedDistance" begin
        for dqmc in dqmcs
            iter = MonteCarlo.with_lattice(EachLocalQuadBySyncedDistance(6), lattice(dqmc))
            dir2srctrg = lattice(dqmc)[:dir2srctrg]
            iter_length = mapreduce(dir -> length(dir2srctrg[dir])^2, +, iter.iter.directions)
            @test length(iter) == iter_length
            @test eltype(iter) == NTuple{5, Int64}
            @test Base.IteratorSize(EachLocalQuadBySyncedDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachLocalQuadBySyncedDistance) == Base.HasEltype()

            dirs = directions(Bravais(lattice(dqmc)))
            full_dirs = directions(lattice(dqmc))
            pos = collect(positions(Bravais(lattice(dqmc))))
            uc_pos = positions(unitcell(lattice(dqmc)))
            full_pos = collect(positions(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(lattice(dqmc))

            # Let's summarize these tests...
            check12 = true
            check1 = true
            check2 = true

            B = length(lattice(dqmc).unitcell.sites)
            idxs = CartesianIndices((B, B, length(dirs), 6))
            N = 0

            for (lin, src1, trg1, src2, trg2) in iter
                N += 1
                uc1, uc2, idx12, idx = Tuple(idxs[lin])

                # src1 -- idx12 -- src2
                _d = full_pos[src1] - full_pos[src2] - dirs[idx12] - uc_pos[uc1] + uc_pos[uc2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check12 = check12 && all(d .< 1e-15)

                # src1 -- idx -- trg1
                _d = full_pos[src1] - full_pos[trg1]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check1 = check1 && (full_dirs[idx] ≈ d)

                # src2 -- idx -- trg2
                _d = full_pos[src2] - full_pos[trg2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check2 = check2 && (full_dirs[idx] ≈ d)
            end

            @test length(iter) == N
            @test check12
            @test check1
            @test check2
        end
    end

    @testset "Sum Wrapper" begin
        for dqmc in dqmcs
            wrapped = MonteCarlo.with_lattice(Sum(EachSitePairByDistance()), lattice(dqmc))

            N = length(lattice(dqmc))
            @test eltype(wrapped) == NTuple{3, Int}
            @test length(wrapped) == N^2
        
            counter = zeros(Int, N, N)
            check = true
            
            for (i, src, trg) in wrapped
                counter[src, trg] += 1
                check = check && (i == 1)
            end
            @test check
            @test all(1 .== counter)
        end
    end

    # @testset "Symmetry Wrapper" begin
    #     iter = EachSitePairByDistance()(dqmc1, dqmc1.model)
    #     wrapped = ApplySymmetries(
    #         EachSitePairByDistance(), 
    #         ([1], [0, 1, 1])
    #     )(dqmc1, dqmc1.model)

    #     @test eltype(wrapped) == eltype(iter)
    #     @test length(wrapped) == length(iter)
        
    #     check = true
    #     vals = collect(iter)
    #     wals = collect(wrapped)
    #     for (v, w) in zip(vals, wals)
    #         check = check && (v == w)
    #     end
    #     @test check
    # end

    # @testset "Superfluid Stiffness Wrapper" begin
    #     iter = EachLocalQuadByDistance([2, 3, 4, 5])(dqmc3, dqmc3.model)
    #     wrapped = MonteCarlo.SuperfluidStiffness(
    #         EachLocalQuadByDistance([2, 3, 4, 5]), [1.0, 0.0]
    #     )(dqmc3, dqmc3.model)

    #     @test eltype(wrapped) == eltype(iter)
    #     @test length(wrapped) == length(iter)
        
    #     check = true
    #     vals = collect(iter)
    #     wals = collect(wrapped)
    #     for (v, w) in zip(vals, wals)
    #         check = check && (v == w)
    #     end
    #     @test check

    #     all_dirs = directions(lattice(dqmc3))
    #     fs = map(idx -> dot([1.0, 0.0], all_dirs[idx]), [2,3,4,5])
    #     weights = Array{ComplexF64}(undef, length(all_dirs), 4, 4)
    #     for (i, dr12) in enumerate(all_dirs)
    #         for (j, dj) in enumerate([2,3,4,5]), (k, dk) in enumerate([2,3,4,5])
    #             weights[i,j,k] = 0.25 * fs[j] * fs[k] *
    #                 (cis(-dot([1.0, 0.0], dr12 + 0.5 * (all_dirs[dj] - all_dirs[dk]))) - 1)
    #         end
    #     end

    #     @test weights ≈ wrapped.weights
    # end


    # TODO
    # SymmetryWrapper, SuperfluidDensity wrapper
end