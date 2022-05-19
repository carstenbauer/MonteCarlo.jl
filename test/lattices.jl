using MonteCarlo: lattice_vectors

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


using MonteCarlo: directed_norm

@testset "Lattice Iterators" begin
    # Set up a few test models/DQMCs    
    m = HubbardModel(3, 3)
    dqmc1 = DQMC(m, beta=1.0)
    
    m = HubbardModel(10, 1, U = -1.0)
    dqmc2 = DQMC(m, beta=1.0)

    l = SquareLattice(2, 6)
    m = HubbardModel(l)
    dqmc3 = DQMC(m, beta=1.0)

    dqmcs = (dqmc1, dqmc2, dqmc3)


    @testset "Meta" begin
        for dqmc in dqmcs
            dirs = directions(dqmc)
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
            iter = EachSiteAndFlavor()(dqmc, dqmc.model)
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
            iter = EachSite()(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            @test collect(iter) == 1:Nsites
            @test length(iter) == Nsites
            @test eltype(iter) == Int64
            @test Base.IteratorSize(EachSite) == Base.HasLength()
            @test Base.IteratorEltype(EachSite) == Base.HasEltype()
        end
    end

    @testset "OnSite" begin
        for dqmc in dqmcs
            iter = OnSite()(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            @test collect(iter) == collect(zip(1:Nsites, 1:Nsites))
            @test length(iter) == Nsites
            @test eltype(iter) == Tuple{Int64, Int64}
            @test Base.IteratorSize(OnSite) == Base.HasLength()
            @test Base.IteratorEltype(OnSite) == Base.HasEltype()
        end
    end

    @testset "EachSitePair" begin
        for dqmc in dqmcs
            iter = EachSitePair()(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            @test collect(iter) == [(i, j) for i in 1:Nsites for j in 1:Nsites]
            @test length(iter) == Nsites^2
            @test eltype(iter) == Tuple{Int64, Int64}
            @test Base.IteratorSize(EachSitePair) == Base.HasLength()
            @test Base.IteratorEltype(EachSitePair) == Base.HasEltype()
        end
    end

    @testset "EachSitePairByDistance" begin
        for dqmc in dqmcs
            iter = EachSitePairByDistance()(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            @test length(iter) == Nsites^2
            @test eltype(iter) == NTuple{3, Int64}
            @test Base.IteratorSize(EachSitePairByDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachSitePairByDistance) == Base.HasEltype()

            dirs = directions(dqmc)
            pos = collect(positions(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(
                size(lattice(dqmc)) .* MonteCarlo.lattice_vectors(lattice(dqmc)))

            # Let's summarize these tests...
            check = true

            for (idx, src, trg) in iter
                _d = pos[src] - pos[trg]
                # Find lowest distance w/ periodic bounds
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check = check && (dirs[idx] ≈ d)
            end

            @test check

            _dirs = directions(iter, lattice(dqmc))
            check = true
            for (v, u) in zip(dirs, _dirs)
                check = check && (v ≈ u)
            end
            @test check
        end
    end

    @testset "EachLocalQuadByDistance" begin
        for dqmc in dqmcs
            iter = EachLocalQuadByDistance(6)(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            @test length(iter) == 6^2 * Nsites^2 # TODO x3
            @test eltype(iter) == NTuple{5, Int64}
            @test Base.IteratorSize(EachLocalQuadByDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachLocalQuadByDistance) == Base.HasEltype()

            dirs = directions(dqmc)
            pos = collect(positions(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(
                size(lattice(dqmc)) .* MonteCarlo.lattice_vectors(lattice(dqmc))
            )

            # Let's summarize these tests...
            check12 = true
            check1 = true
            check2 = true

            idxs = CartesianIndices((length(dirs), 6, 6))

            for (lin, src1, trg1, src2, trg2) in iter
                idx12, idx1, idx2 = Tuple(idxs[lin])

                # src1 -- idx12 -- src2
                _d = pos[src1] - pos[src2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check12 = check12 && (dirs[idx12] ≈ d)

                # src1 -- idx1 -- trg1
                _d = pos[src1] - pos[trg1]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check1 = check1 && (dirs[idx1] ≈ d)

                # src2 -- idx2 -- trg2
                _d = pos[src2] - pos[trg2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check2 = check2 && (dirs[idx2] ≈ d)
            end

            @test check12
            @test check1
            @test check2
        end
    end

    @testset "EachLocalQuadBySyncedDistance" begin
        for dqmc in dqmcs
            iter = EachLocalQuadBySyncedDistance(6)(dqmc, dqmc.model)
            Nsites = length(lattice(dqmc))
            # These are wrong on a lattice with a basis, because {6}
            # does not mean 6 surrounding sites, but rather 6 smallest directions (globally).
            # (A site directions may not apply to B sites)
            #@test length(iter) == 6^2 * Nsites^2
            #@test length(collect(iter)) == 6^2 * Nsites^2
            @test eltype(iter) == NTuple{5, Int64}
            @test Base.IteratorSize(EachLocalQuadBySyncedDistance) == Base.HasLength()
            @test Base.IteratorEltype(EachLocalQuadBySyncedDistance) == Base.HasEltype()

            dirs = directions(dqmc)
            pos = collect(positions(lattice(dqmc)))
            wrap = MonteCarlo.generate_combinations(
                size(lattice(dqmc)) .* MonteCarlo.lattice_vectors(lattice(dqmc)))

            # Let's summarize these tests...
            check12 = true
            check1 = true
            check2 = true

            idxs = CartesianIndices((length(dirs), 6))

            for (lin, src1, trg1, src2, trg2) in iter
                idx12, idx = Tuple(idxs[lin])

                # src1 -- idx12 -- src2
                _d = pos[src1] - pos[src2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check12 = check12 && (dirs[idx12] ≈ d)

                # src1 -- idx -- trg1
                _d = pos[src1] - pos[trg1]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check1 = check1 && (dirs[idx] ≈ d)

                # src2 -- idx -- trg2
                _d = pos[src2] - pos[trg2]
                d = _d .+ wrap[1]
                for v in wrap[2:end]
                    new_d = _d .+ v
                    if directed_norm(new_d, 1e-6) + 1e-6 < directed_norm(d, 1e-6)
                        d .= new_d
                    end
                end
                check2 = check2 && (dirs[idx] ≈ d)
            end

            @test check12
            @test check1
            @test check2
        end
    end

    @testset "Sum Wrapper" begin
        iter = EachSitePair()(dqmc1, dqmc1.model)
        wrapped = Sum(EachSitePairByDistance())(dqmc1, dqmc1.model)

        @test eltype(wrapped) == eltype(iter)
        @test length(wrapped) == length(iter)
        
        check = true
        vals = collect(iter)
        wals = collect(wrapped)
        for (v, w) in zip(vals, wals)
            check = check && (v == w)
        end
        @test check
    end

    @testset "Symmetry Wrapper" begin
        iter = EachSitePairByDistance()(dqmc1, dqmc1.model)
        wrapped = ApplySymmetries(
            EachSitePairByDistance(), 
            ([1], [0, 1, 1])
        )(dqmc1, dqmc1.model)

        @test eltype(wrapped) == eltype(iter)
        @test length(wrapped) == length(iter)
        
        check = true
        vals = collect(iter)
        wals = collect(wrapped)
        for (v, w) in zip(vals, wals)
            check = check && (v == w)
        end
        @test check
    end

    @testset "Superfluid Stiffness Wrapper" begin
        iter = EachLocalQuadByDistance([2, 3, 4, 5])(dqmc3, dqmc3.model)
        wrapped = MonteCarlo.SuperfluidStiffness(
            EachLocalQuadByDistance([2, 3, 4, 5]), [1.0, 0.0]
        )(dqmc3, dqmc3.model)

        @test eltype(wrapped) == eltype(iter)
        @test length(wrapped) == length(iter)
        
        check = true
        vals = collect(iter)
        wals = collect(wrapped)
        for (v, w) in zip(vals, wals)
            check = check && (v == w)
        end
        @test check

        all_dirs = directions(lattice(dqmc3))
        fs = map(idx -> dot([1.0, 0.0], all_dirs[idx]), [2,3,4,5])
        weights = Array{ComplexF64}(undef, length(all_dirs), 4, 4)
        for (i, dr12) in enumerate(all_dirs)
            for (j, dj) in enumerate([2,3,4,5]), (k, dk) in enumerate([2,3,4,5])
                weights[i,j,k] = 0.25 * fs[j] * fs[k] *
                    (cis(-dot([1.0, 0.0], dr12 + 0.5 * (all_dirs[dj] - all_dirs[dk]))) - 1)
            end
        end

        @test weights ≈ wrapped.weights
    end


    # TODO
    # SymmetryWrapper, SuperfluidDensity wrapper
end


# # L = 3 hardcoded Honeycomb
# struct HoneycombTestLattice <: AbstractLattice
#     positions::Vector{Vector{Float64}}
# end

# function HoneycombTestLattice()
#     # Taken from LatticePhysics (L = 3)
#     pos = [[0.0, 0.0], [0.5773502691896258, 0.0], [0.8660254037844386, 0.5], [1.4433756729740645, 0.5], [1.7320508075688772, 1.0], [2.309401076758503, 1.0], [0.8660254037844386, -0.5], [1.4433756729740645, -0.5], [1.7320508075688772, 0.0], [2.309401076758503, 0.0], [2.598076211353316, 0.5], [3.1754264805429417, 0.5], [1.7320508075688772, -1.0], [2.309401076758503, -1.0], [2.598076211353316, -0.5], [3.1754264805429413, -0.5], [3.4641016151377544, 0.0], [4.04145188432738, 0.0]]
#     HoneycombTestLattice(pos)
# end
# Base.length(l::HoneycombTestLattice) = length(l.positions)
# MonteCarlo.positions(l::HoneycombTestLattice) = l.positions
# function MonteCarlo.DistanceMask(lattice::HoneycombTestLattice)
#     L = 3
#     wrap = MonteCarlo.generate_combinations(
#         [L * [0.8660254037844386, -0.5], L * [0.8660254037844386, 0.5]]
#     )
#     MonteCarlo.VerboseDistanceMask(lattice, wrap)
# end
# function MonteCarlo.directions(mask::MonteCarlo.VerboseDistanceMask, lattice::HoneycombTestLattice)
#     pos = MonteCarlo.positions(lattice)
#     dirs = [pos[trg] - pos[src] for (src, trg) in first.(mask.targets)]
#     # marked = Set{Int64}()
#     # dirs = Vector{eltype(pos)}(undef, maximum(first(x) for x in mask.targets))
#     # for src in 1:size(mask.targets, 1)
#     #     for (idx, trg) in mask.targets[src, :]
#     #         if !(idx in marked)
#     #             push!(marked, idx)
#     #             dirs[idx] = pos[trg] - pos[src]
#     #         end
#     #     end
#     # end
#     L = 3
#     wrap = MonteCarlo.generate_combinations(
#         [L * [0.8660254037844386, -0.5], L * [0.8660254037844386, 0.5]]
#     )
#     map(dirs) do _d
#         d = round.(_d .+ wrap[1], digits=6)
#         for v in wrap[2:end]
#             new_d = round.(_d .+ v, digits=6)
#             if norm(new_d) < norm(d)
#                 d .= new_d
#             end
#         end
#         d
#     end
# end

# @testset "Mask with basis" begin
#     l = HoneycombTestLattice()
#     L = 3
#     positions = MonteCarlo.positions(l)
#     mask = DistanceMask(l)
#     @test mask isa MonteCarlo.VerboseDistanceMask
#     @test length(mask) == 27
#     @test allunique(MonteCarlo.getorder(mask))
#     # for i in 1:length(l)
#     #     @test allunique(mask[i, :])
#     #     @test allunique(mask[:, i])
#     # end
#     wrap = MonteCarlo.generate_combinations(
#         [L * [0.8660254037844386, -0.5], L * [0.8660254037844386, 0.5]]
#     )

#     # in the same direction
#     dirs = MonteCarlo.directions(mask, l)
#     for dir_idx in 1:length(mask)
#         _dirs = map(MonteCarlo.getdirorder(mask, dir_idx)) do (src, trg)
#             d = round.(positions[trg] .- positions[src] .+ wrap[1], digits=6)
#             for v in wrap[2:end]
#                 new_d = round.(positions[trg] .- positions[src] .+ v, digits=6)
#                 if norm(new_d) < norm(d)
#                     d .= new_d
#                 end
#             end
#             d
#         end
#         @test all(==(dirs[dir_idx]), _dirs)
#     end
#     # for src in 1:length(l)
#     #     for (idx, trg) in MonteCarlo.getorder(mask, src)
#     #         d = round.(positions[trg] .- positions[src] .+ wrap[1], digits=6)
#     #         for v in wrap[2:end]
#     #             new_d = round.(positions[trg] .- positions[src] .+ v, digits=6)
#     #             if norm(new_d) < norm(d)
#     #                 d .= new_d
#     #             end
#     #         end
#     #         @test dirs[idx] ≈ d
#     #     end
#     # end
# end

# @testset "2D Honeycomb" begin
#     for L in (2, 3)
#         l = MonteCarlo.HoneycombLattice(L)
#         @test length(l) == (2L)^2
#
#         bonds = collect(MonteCarlo.bonds(l, Val(true)))
#         @test length(bonds) == 3 * (2L)^2
#         @test allunique(bonds)
#         for i in 0:length(l)-1
#             # same source
#             @test all(bonds[3i+1][1] == x[1] for x in bonds[3i .+ (2:3)])
#             # different target
#             @test allunique(x[2] for x in bonds[3i .+ (1:3)])
#         end
#
#         reduced_bonds = collect(MonteCarlo.bonds(l, Val(false)))
#         @test length(reduced_bonds) == round(Int64, 1.5 * (2L)^2)
#         # If directed is false, only one of (i, j) and (j, i)
#         # should be kept.
#         mirrored_bonds = [[trg, src] for (src, trg) in reduced_bonds]
#         all_bonds = vcat(reduced_bonds, mirrored_bonds)
#         @test sort(all_bonds) == sort(bonds)
#
#         # For L < 3 NNN wrap around the lattice
#         L < 3 && continue
#         l = MonteCarlo.HoneycombLattice(L, true)
#         @test size(l.NNNs) == (6, length(l))
#         for i in 1:length(l)
#             @test allunique(l.NNNs[:, i])
#         end
#
#         positions = MonteCarlo.positions(l)
#         mask = MonteCarlo.DistanceMask(l)
#         @test mask isa MonteCarlo.VerboseDistanceMask
#         for i in 1:length(l)
#             @test allunique(mask[i, :])
#             @test allunique(mask[:, i])
#         end
#         wrap = MonteCarlo.generate_combinations(
#             [L * [0.8660254037844386, -0.5], L * [0.8660254037844386, 0.5]]
#         )
#
#         # in the same direction
#         dirs = MonteCarlo.directions(mask, l)
#         for src in 1:length(l)
#             for (idx, trg) in MonteCarlo.getorder(mask, src)
#                 d = round.(positions[trg] .- positions[src] .+ wrap[1], digits=6)
#                 for v in wrap[2:end]
#                     new_d = round.(positions[trg] .- positions[src] .+ v, digits=6)
#                     if norm(new_d) < norm(d)
#                         d .= new_d
#                     end
#                 end
#                 @test dirs[idx] ≈ d
#             end
#         end
#     end
# end


# # TODO: maybe if LatticePhysics gets registered one day?
# @testset "LatPhys" begin
#     for uc in [
#             getUnitcellSquare(), getUnitcellTriangular(),
#             getUnitcellKagome(), getUnitcellHoneycomb(),
#             getUnitcellSC(), getUnitcellFCC(),
#             getUnitcellDiamond(),
#             getUnitcell_9_3_a() # <- This one takes a bit longer
#         ]
#         l = getLatticePeriodic(uc, 4)
#         @info l
#         lattice = MonteCarlo.LatPhysLattice(l)
#
#         @test length(lattice) == numSites(l)
#         @test ndims(lattice) == ndims(l)
#
#         # ...
#
#         mask = MonteCarlo.DistanceMask(lattice)
#         for i in 1:length(lattice)
#             @test allunique(mask[i, :])
#             @test allunique(mask[:, i])
#         end
#
#         positions = point.(sites(l))
#         wrap = MonteCarlo.generate_combinations(latticeVectors(l))
#
#         # in the same direction
#         dirs = MonteCarlo.directions(mask, lattice)
#         for src in 1:length(lattice)
#             for (idx, trg) in MonteCarlo.getorder(mask, src)
#                 d = round.(positions[trg] .- positions[src] .+ wrap[1], digits=6)
#                 for v in wrap[2:end]
#                     new_d = round.(positions[trg] .- positions[src] .+ v, digits=6)
#                     if norm(new_d) < norm(d)
#                         d .= new_d
#                     end
#                 end
#                 @test dirs[idx] ≈ d
#             end
#         end
#     end
# end