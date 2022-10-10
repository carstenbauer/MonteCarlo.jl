"""
    DistanceBased(iter::AbstractLatticeIterator)

Wraps a lattice iterator as `Restructure(iter)`, marking that it should use a 
distance based Greens function `G[src, dir]`.

Measurements using `DistanceBased` lattice iterator require a custom 
implementation of the lattice iteration scheme which is guaranteed to exist if 
`DistanceBased(iter)` is defined. Furthermore the kernel needs to be adjusted to 
work with a different set of indices and Greens matrices.
"""
DistanceBased(iter::EachBondPairByBravaisDistance) = Restructure(iter)


function _save(file::FileLike, key::String, r::Restructure)
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", "Restructure")
    _save(file, "$key/wrapped", r.wrapped)
    return
end

function _load(data, ::Val{:Restructure})
    return Restructure(_load(data["wrapped"]))
end


output_size(r::Restructure, l::AbstractLattice) = output_size(r.wrapped, l)
function output_size(r::Restructure{<: EachBondPairByBravaisDistance}, l::Lattice)
    B = length(l.unitcell.bonds)
    return (l.Ls..., B, B)
end

prepare!(r::Restructure, m, mc) = prepare!(r.wrapped, m, mc)
finish!(r::Restructure, m, mc) = finish!(r.wrapped, m, mc)



################################################################################
### Matrix restructuring
################################################################################



"""
    restructure!(mc, G[; temp = mc.stack.curr_U, target = G])

Takes a matrix `G` with indexing format `[source, target]` and restructures it 
to `[source, direction]`.

In more detail, `source` and `target` are assumed to be be linear Bravais 
lattice indices dressed with other indices at higher dimension. For example 
`source = x + Lx * (y-1) + Lx * Ly * (uc-1 + B * (flv - 1))` in a 2d lattice 
with Bravais lattice indices `1 ≤ x ≤ Lx`, `1 ≤ y ≤ Ly`, Basis/unitcell index 
`1 ≤ uc ≤ B` and flavor index `1 ≤ flv`. In this case the matrix could be 
reshaped to `[src_x, src_y, src_uc, src_flv, trg_x, trg_y, trg_uc, trg_flv]`.
The restructuring only affects the Bravais lattice indices `trg_x` and `trg_y`, 
such that the full indexing becomes
`[src_x, src_y, src_uc, src_flv, mod1(trg_x-src_x+1, Lx), mod1(trg_y-src_y, Ly), trg_uc, trg_flv]`.

Note that `restructure!` works regardless of what the higher dimensions are as 
long as the first indices match the Bravais lattice size.

This function works with tuples of matrices, `GreensMatrix`, `Matrix` and all 
the matrix types implemented in MonteCarlo.jl.
"""
@bm function restructure!(mc, Gs::Tuple; kwargs...)
    restructure!.((mc,), Gs; kwargs...)
    return
end
@bm function restructure!(mc, G::GreensMatrix; kwargs...)
    restructure!(mc, G.val; kwargs...)
    return
end
function restructure!(mc, G::BlockDiagonal; temp = mc.stack.curr_U, kwargs...)
    for i in eachindex(G.blocks)
        restructure!(mc, G.blocks[i]; temp = temp.blocks[i], kwargs...)
    end
    return
end
function restructure!(mc, G::StructArray; temp = mc.stack.curr_U, kwargs...)
    restructure!(mc, G.re; temp = temp.re, kwargs...)
    restructure!(mc, G.im; temp = temp.im, kwargs...)
    return
end
function restructure!(mc, G::DiagonallyRepeatingMatrix; kwargs...)
    restructure!(mc, G.val; kwargs...)
    return
end
function restructure!(mc, G::Matrix; target = G, temp = mc.stack.curr_U)
    N = length(Bravais(lattice(mc)))
    K = div(size(G, 1), N)
    dir2srctrg = lattice(mc)[:Bravais_dir2srctrg]::Vector{Vector{Int}}

    if target === G
        for k1 in 0:N:N*K-1, k2 in 0:N:N*K-1 # flv and basis
            for dir in eachindex(dir2srctrg)
                for (src, trg) in enumerate(dir2srctrg[dir])
                    temp[src+k1, dir+k2] = G[src+k1, trg+k2]
                end
            end
        end

        copyto!(target, temp)
    else
        for k1 in 0:N:N*K-1, k2 in 0:N:N*K-1 # flv and basis
            for dir in eachindex(dir2srctrg)
                for (src, trg) in enumerate(dir2srctrg[dir])
                    target[src+k1, dir+k2] = G[src+k1, trg+k2]
                end
            end
        end
    end

    return
end



################################################################################
### Custom lattice iteration schemes
################################################################################


function apply!(
        temp::Array, ::Restructure{<: EachSitePairByDistance}, measurement, mc::DQMC, 
        packed_greens, weight = 1.0
    )

    # Sketch:
    #      --dir-->
    #  (src) --- (trg)
    #     <--rev---
    
    # With src using basis index b1 and trg using b2 (or shifts uc1, uc2)

    @timeit_debug "apply!(::Restructure{EachSitePairByDistance}, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        N = length(Bravais(l))
        indices = CartesianIndices(l.Ls) # one based matrix indices
        
        @inbounds @fastmath for σ in measurement.flavor_iterator
            for b2 in eachindex(l.unitcell.sites), b1 in eachindex(l.unitcell.sites)
                uc1 = N * (b1-1)
                uc2 = N * (b2-1)
                for cdir in indices
                    # one based: mod1.(l.Ls + 2 - distance, l.Ls) 
                    # maps (1 -> 1, 2 -> L, 3 -> L-1, ...)
                    crev = @. ifelse(cdir.I > 1, l.Ls + 2 - cdir.I, 1)

                    # map to linear indices (1-based)
                    _dir = _sub2ind(l.Ls, cdir.I)
                    dir = _dir + uc2
                    rev = _sub2ind(l.Ls, crev) + uc1

                    @simd for csrc in indices
                        # cdir is one based so -1
                        # results is 0 < idx < 2L, so we can use ifelse as mod1
                        ctrg = csrc.I .+ cdir.I .- 1
                        ctrg = @. ifelse(ctrg > l.Ls, ctrg - l.Ls, ctrg)

                        # to linear index
                        src = _sub2ind(l.Ls, csrc.I) + uc1
                        trg = _sub2ind(l.Ls, ctrg) + uc2

                        # measure
                        temp[_dir, b1, b2] += weight * measurement.kernel(
                            mc, mc.model, (src, trg), (dir, rev), (uc1, uc2), packed_greens, σ
                        )
                    end # src loop
                end # dir loop
            end # basis loops
        end # flavor loop
    end # benchmark

    nothing
end


function construct_dir2sub(l::AbstractLattice)
    return map(_dir2srctrg(l)) do srctrgs
        map(srctrgs) do (src, trg)
            sub_src = _ind2sub(l, src)
            sub_trg = _ind2sub(l, trg)
            dir = sub_trg[1:end-1] .- sub_src[1:end-1]
            dir = @. ifelse(dir < 0, l.Ls + dir, dir)
            (dir, sub_src[end], sub_trg[end])
        end |> unique
    end
end

function generate_iterable(dirs::BondDirections, l::Lattice)
    Iterators.map(l.unitcell.bonds) do b
        uc1 = from(b)
        uc2 = to(b)
        sub_d = @. ifelse(b.uc_shift < 0, l.Ls + b.uc_shift, b.uc_shift)
        ((sub_d, uc1, uc2),)
    end |> enumerate
end

function generate_iterable(dirs::Union{Vector, Tuple}, l::Lattice)
    D = length(l.Ls)
    dir2sub = get!(l, :dir2sub, construct_dir2sub)::Vector{Vector{Tuple{NTuple{D, Int}, Int, Int}}}
    return (idx => dir2sub[dir] for (idx, dir) in dirs)
end


function apply!(
        temp::Array, iter::Restructure{<: EachLocalQuadByDistance}, 
        measurement::DQMCMeasurement, mc::DQMC, packed_greens, weight = 1.0
    )

    #           x, y
    # trg1 ---- src1 ---- src2 ---- trg2
    #    dx1, dy1   dx, dy   dx2, dy2
    #    sub_dir1            sub_dir2
    # uc12      uc11      uc21      uc22

    @timeit_debug "apply!(::Restructure{EachLocalQuadByDistance}, ::$(typeof(measurement.kernel)))" begin
        lat = lattice(mc)
        N = length(Bravais(lat))
        Ls = lat.Ls
        dirs = generate_iterable(iter.wrapped.directions, lat) # is this a type problem?

        T0 = zero(eltype(temp))
        indices0 = CartesianIndices(map(L -> 0:L-1, Ls)) # zero based
        indices1 = CartesianIndices(Ls) # one based

        @inbounds @fastmath for σ in measurement.flavor_iterator
            # Assuming 2D
            for (dir_idx2, sub_dirs2) in dirs
                for (dir_idx1, sub_dirs1) in dirs
                    for (sub_d2, uc21, uc22) in sub_dirs2
                        b2 = uc21
                        uc21 = N * (uc21 - 1)
                        uc22 = N * (uc22 - 1)
                        Δkl = _sub2ind0(Ls, sub_d2) + 1 + uc22
                        sub_r2 = @. ifelse.(sub_d2 > 0, Ls .- sub_d2, 0)
                        Δlk = _sub2ind0(Ls, sub_r2) + 1 + uc21

                        for (sub_d1, uc11, uc12) in sub_dirs1
                            b1 = uc11
                            uc11 = N * (uc11 - 1)
                            uc12 = N * (uc12 - 1)
                            Δij = _sub2ind0(Ls, sub_d1) + 1 + uc12
                            sub_r1 = @. ifelse.(sub_d1 > 0, Ls .- sub_d1, 0)
                            Δji = _sub2ind0(Ls, sub_r1) + 1 + uc21

                            for sub_d in indices0 # zero based
                                val = T0

                                # Inlined mod1, see comment below
                                # src2 -> trg1
                                # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                                sub_p1 = @. sub_d1 - sub_d.I
                                sub_p1 = @. ifelse(sub_p1 < 0, sub_p1 + Ls, sub_p1) 
                                Δkj = _sub2ind0(Ls, sub_p1) + 1 + uc12
                                sub_r = @. ifelse(sub_p1 > 0, Ls - sub_p1, 0)
                                Δjk = _sub2ind0(Ls, sub_r) + 1 + uc21


                                # src1 -> trg2
                                # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                                sub_p2 = @. sub_d2 + sub_d.I
                                sub_p2 = @. ifelse(sub_p2 ≥ Ls, sub_p2 - Ls, sub_p2) 
                                Δil = _sub2ind0(Ls, sub_p2) + 1 + uc22
                                sub_r = @. ifelse(sub_p2 > 0, Ls - sub_p2, 0)
                                Δli = _sub2ind0(Ls, sub_r) + 1 + uc11

                                Δik = _sub2ind0(Ls, sub_d.I) + 1 + uc21
                                sub_r = @. ifelse(sub_d.I > 0, Ls - sub_d.I, 0)
                                Δki = _sub2ind0(Ls, sub_r) + 1 + uc11

                                sub = @. sub_p2 - sub_d1
                                sub = @. ifelse(sub < 0, sub + Ls, sub) 
                                Δjl = _sub2ind0(Ls, sub) + 1 + uc22
                                sub_r = @. ifelse(sub > 0, Ls - sub, 0)
                                Δlj = _sub2ind0(Ls, sub_r) + 1 + uc12


                                @simd for sub_src in indices1
                                    # linear src index
                                    i = _sub2ind(Ls, sub_src.I) + uc11
                                    # mod1(src + dir, Ls)
                                    sub = @. sub_src.I + sub_d1
                                    sub = @. ifelse(sub > Ls, sub - Ls, sub)
                                    j = _sub2ind(Ls, sub) + uc12

                                    sub = @. sub_src.I + sub_d.I
                                    sub = @. ifelse(sub > Ls, sub - Ls, sub)
                                    k = _sub2ind(Ls, sub) + uc21
                                    # l follows from k
                                    sub = @. sub + sub_d2
                                    sub = @. ifelse(sub > Ls, sub - Ls, sub)
                                    l = _sub2ind(Ls, sub) + uc22

                                    val += measurement.kernel(
                                        mc, mc.model, 
                                        (i, j, k, l), 
                                        (Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk),
                                        (uc11, uc12, uc21, uc22),
                                        packed_greens, σ
                                    )
                                end # source site loop

                                # 0-based index to 1 based index
                                # need += for time integrals
                                temp[Δik - uc21, dir_idx1, dir_idx2, b1, b2] += weight * val
                            end # main distance loop

                        end # iterate lattice indices in direction
                    end # iterate lattice indices in direction
                end # iterate directions
            end # iterate directions

        end # flavor
    end # benchmark

    return 
end


function apply!(
        temp::Array, iter::Restructure{<: EachLocalQuadBySyncedDistance}, 
        measurement::DQMCMeasurement, mc::DQMC, packed_greens, weight = 1.0
    )

    #           x, y
    # trg1 ---- src1 ---- src2 ---- trg2
    #    dx1, dy1   dx, dy   dx2, dy2
    #    sub_dir1            sub_dir2
    # uc12      uc11      uc21      uc22

    @timeit_debug "apply!(::Restructure{EachLocalQuadBySyncedDistance}, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        N = length(Bravais(l))
        dirs = generate_iterable(iter.directions, l) # is this a type problem?

        T0 = zero(eltype(temp))
        indices0 = CartesianIndices(map(L -> 0:L-1, l.Ls)) # zero based
        indices1 = CartesianIndices(l.Ls) # one based
        cart1 = CartesianIndex(map(_ -> 1, l.Ls))

        @inbounds @fastmath for σ in measurement.flavor_iterator
            # Assuming 2D
            for (dir_idx2, sub_dirs2) in enumerate(dirs)
                for (dir_idx1, sub_dirs1) in enumerate(dirs)
                    for (sub_d2, uc21, uc22) in sub_dirs2
                        uc21 = N * (uc21 - 1)
                        uc22 = N * (uc22 - 1)
                        d2 = _sub2ind0(l.Ls, sub_d2) + 1

                        for (sub_d1, uc11, uc12) in sub_dirs1
                            uc11 = N * (uc11 - 1)
                            uc12 = N * (uc12 - 1)
                            d1 = _sub2ind0(l.Ls, sub_d1) + 1


                            for sub_d in indices0 # zero based
                                val = T0

                                # Inlined mod1, see comment below
                                # src2 -> trg1
                                # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                                sub_p1 = @. sub_d1 - sub_d.I
                                sub_p1 = @. ifelse(sub_p1 < 0, sub_p1 + l.Ls, sub_p1) 
                                p1 = _sub2ind0(l.Ls, sub_p1) + 1

                                # src1 -> trg2
                                # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                                sub_p2 = @. sub_d2 + sub_d.I
                                sub_p2 = @. ifelse(sub_p2 ≥ l.Ls, sub_p2 - l.Ls, sub_p2) 
                                p2 = _sub2ind0(l.Ls, sub_p2) + 1

                                @simd for sub_src in indices1
                                    # linear src index
                                    i = _sub2ind(l.Ls, sub_src.I)
                                    # mod1(src + dir, Ls)
                                    sub_src2 = @. sub_src.I + sub_d.I
                                    sub_src2 = @. ifelse(sub_src2 > l.Ls, sub_src2 - l.Ls, sub_src2)
                                    i2 = _sub2ind(l.Ls, sub_src2)

                                    val += cc_kernel(
                                        mc, mc.model, 
                                        (i, i2), (d1, d2, p1, p2),
                                        (uc11, uc12, uc21, uc22),
                                        packed_greens, σ
                                    )
                                end # source site loop

                                # 0-based index to 1 based index
                                # need += for time integrals
                                temp[sub_d + cart1, dir_idx1, dir_idx2] += weight * val
                            end # main distance loop

                        end # iterate lattice indices in direction
                    end # iterate lattice indices in direction
                end # iterate directions
            end # iterate directions

        end # flavor
    end # benchmark

    return 
end


function apply!(
        temp::Array, ::Restructure{<: EachBondPairByBravaisDistance}, 
        measurement::DQMCMeasurement, mc::DQMC, packed_greens, weight = 1.0
    )

    #           x, y
    # trg1 ---- src1 ---- src2 ---- trg2
    #    dx1, dy1   dx, dy   dx2, dy2
    #       b1                  b2
    # uc12      uc11      uc21      uc22

    @timeit_debug "apply!(::Restructure{EachBondPairByBravaisDistance}, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        N = length(Bravais(l))
        bs = l.unitcell.bonds

        T0 = zero(eltype(temp))
        directions = CartesianIndices(map(L -> 0:L-1, l.Ls)) # zero based
        indices = CartesianIndices(l.Ls) # one based
        cart1 = CartesianIndex(map(_ -> 1, l.Ls))

        @inbounds @fastmath for σ in measurement.flavor_iterator
            # Assuming 2D
            for (bi, b1) in enumerate(bs)
                # Note: 
                # we can always use this form and we require it when calculating 
                # flat indices/distances, so we're doing it early here.
                uc11 = N * (from(b1) - 1)
                uc12 = N * (to(b1) - 1)

                # mod0(bond.uc_shift, L) to map negative shifts to positive shifts 
                sub_d1 = @. ifelse(b1.uc_shift < 0, l.Ls + b1.uc_shift, b1.uc_shift)

                # cartesian 0-based index to linear 1-based index
                d1 = _sub2ind0(l.Ls, sub_d1) + 1 + uc12

                for (bj, b2) in enumerate(bs)
                    uc21 = N * (from(b2) - 1)
                    uc22 = N * (to(b2) - 1)
                    sub_d2 = @. ifelse(b2.uc_shift < 0, l.Ls + b2.uc_shift, b2.uc_shift)
                    d2 = _sub2ind0(l.Ls, sub_d2) + 1 + uc22

                    for sub_d in directions # zero based
                        val = T0

                        # Inlined mod1, see comment below
                        # src2 -> trg1
                        # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                        sub_p1 = @. sub_d1 - sub_d.I
                        sub_p1 = @. ifelse(sub_p1 < 0, sub_p1 + l.Ls, sub_p1) 
                        p1 = _sub2ind0(l.Ls, sub_p1) + 1 + uc12

                        # src1 -> trg2
                        # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                        sub_p2 = @. sub_d2 + sub_d.I
                        sub_p2 = @. ifelse(sub_p2 ≥ l.Ls, sub_p2 - l.Ls, sub_p2) 
                        p2 = _sub2ind0(l.Ls, sub_p2) + 1 + uc22

                        @simd for sub_src in indices
                            # linear src index
                            i = _sub2ind(l.Ls, sub_src.I) + uc11
                            # mod1(src + dir, Ls)
                            sub_src2 = @. sub_src.I + sub_d.I
                            sub_src2 = @. ifelse(sub_src2 > l.Ls, sub_src2 - l.Ls, sub_src2)
                            i2 = _sub2ind(l.Ls, sub_src2) + uc21

                            val += measurement.kernel(
                                mc, mc.model, (i, i2), (d1, d2, p1, p2),
                                (uc11, uc12, uc21, uc22),
                                packed_greens, σ
                            )
                        end # source site loop

                        # 0-based index to 1 based index
                        # need += for time integrals
                        temp[sub_d + cart1, bi, bj] += weight * val
                    end # main distance loop

                end # selected distance
            end # selected distance

        end # flavor
    end # benchmark

    return 
end