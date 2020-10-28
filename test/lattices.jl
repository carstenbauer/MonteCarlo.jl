for (d, lattype) in enumerate((
        Chain,
        SquareLattice,
        CubicLattice
    ))
    constructor = d < 3 ? L -> lattype(L) : L -> lattype(d, L)

    @testset "$(d)D $lattype" begin
        for L in (3, 4)
            l = constructor(L)
            @test length(l) == L^d

            bonds = collect(neighbors(l, Val(true)))
            @test length(bonds) == 2*d*L^d
            @test allunique(bonds)
            for i in 0:length(l)-1
                # same source
                @test all(bonds[2d*i + 1][1] == x[1] for x in bonds[2d*i .+ (2:2d)])
                # different target
                @test allunique(x[2] for x in bonds[2d*i .+ (1:2d)])
            end

            reduced_bonds = collect(neighbors(l, Val(false)))
            @test length(reduced_bonds) == d*L^d
            # If directed is false, only one of (i, j) and (j, i)
            # should be kept.
            mirrored_bonds = [[trg, src] for (src, trg) in reduced_bonds]
            all_bonds = vcat(reduced_bonds, mirrored_bonds)
            @test sort(all_bonds) == sort(bonds)

            # if L == 4
            #     positions = if lattype == Chain
            #         [[i] for i in 1:L]
            #     else
            #         l.lattice |> CartesianIndices .|> Tuple .|> collect
            #     end
            #     mask = DistanceMask(l)
            #     @test allunique(MonteCarlo.getorder(mask))
            #     # for i in 1:length(l)
            #     #     @test allunique(mask[i, :])
            #     #     @test allunique(mask[:, i])
            #     # end
            #     # The point of DistanceMask is for all i -> mask[i, j] to point
            #     # in the same direction
            #     dirs = directions(mask, l)
            #     for dir_idx in 1:length(mask)
            #         _dirs = [
            #             mod.(positions[trg] .- positions[src], L) 
            #             for (src, trg) in MonteCarlo.getdirorder(mask, dir_idx)
            #         ]
            #         @test all(==(first(_dirs)), _dirs)
            #         @test mod.(dirs[dir_idx], L) == _dirs[1]
            #     end
            # end
        end
    end
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
#         bonds = collect(MonteCarlo.neighbors(l, Val(true)))
#         @test length(bonds) == 3 * (2L)^2
#         @test allunique(bonds)
#         for i in 0:length(l)-1
#             # same source
#             @test all(bonds[3i+1][1] == x[1] for x in bonds[3i .+ (2:3)])
#             # different target
#             @test allunique(x[2] for x in bonds[3i .+ (1:3)])
#         end
#
#         reduced_bonds = collect(MonteCarlo.neighbors(l, Val(false)))
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
