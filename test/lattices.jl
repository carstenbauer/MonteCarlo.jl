for (d, lattype) in enumerate((
        MonteCarlo.Chain,
        MonteCarlo.SquareLattice,
        MonteCarlo.CubicLattice
    ))
    constructor = d < 3 ? L -> lattype(L) : L -> lattype(d, L)

    @testset "$(d)D $lattype" begin
        for L in (3, 4)
            l = constructor(L)
            @test length(l) == L^d

            bonds = collect(MonteCarlo.neighbors(l, Val(true)))
            @test length(bonds) == 2*d*L^d
            @test allunique(bonds)
            for i in 0:length(l)-1
                # same source
                @test all(bonds[2d*i + 1][1] == x[1] for x in bonds[2d*i .+ (2:2d)])
                # different target
                @test allunique(x[2] for x in bonds[2d*i .+ (1:2d)])
            end

            reduced_bonds = collect(MonteCarlo.neighbors(l, Val(false)))
            @test length(reduced_bonds) == d*L^d
            # If directed is false, only one of (i, j) and (j, i)
            # should be kept.
            mirrored_bonds = [[trg, src] for (src, trg) in reduced_bonds]
            all_bonds = vcat(reduced_bonds, mirrored_bonds)
            @test sort(all_bonds) == sort(bonds)
        end
    end
end

@testset "2D Honeycomb" begin
    for L in (2, 3)
        l = MonteCarlo.HoneycombLattice(L)
        @test length(l) == (2L)^2

        bonds = collect(MonteCarlo.neighbors(l, Val(true)))
        @test length(bonds) == 3 * (2L)^2
        @test allunique(bonds)
        for i in 0:length(l)-1
            # same source
            @test all(bonds[3i+1][1] == x[1] for x in bonds[3i .+ (2:3)])
            # different target
            @test allunique(x[2] for x in bonds[3i .+ (1:3)])
        end

        reduced_bonds = collect(MonteCarlo.neighbors(l, Val(false)))
        @test length(reduced_bonds) == round(Int64, 1.5 * (2L)^2)
        # If directed is false, only one of (i, j) and (j, i)
        # should be kept.
        mirrored_bonds = [[trg, src] for (src, trg) in reduced_bonds]
        all_bonds = vcat(reduced_bonds, mirrored_bonds)
        @test sort(all_bonds) == sort(bonds)

        # For L < 3 NNN wrap around the lattice
        L < 3 && continue
        l = MonteCarlo.HoneycombLattice(L, true)
        @test size(l.NNNs) == (6, length(l))
        for i in 1:length(l)
            @test allunique(l.NNNs[:, i])
        end
    end
end
