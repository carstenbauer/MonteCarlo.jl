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


output_size(r::Restructure) = output_size(r.wrapped)
function output_size(r::Restructure{<: EachBondPairByBravaisDistance}, l::Lattice)
    N = length(Bravais(l))
    B = length(l.unitcell.bonds)
    return (N, N, B, B)
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
function restructure!(mc, Gs::NTuple; kwargs...)
    restructure!.((mc,), Gs; kwargs...)
    return
end
function restructure!(mc, G::GreensMatrix; kwargs...)
    restructure!(mc, G.val; kwargs...)
    return
end
function restructure!(mc, G::BlockDiagonal; kwargs...)
    restructure!.((mc,), G.blocks; kwargs...)
    return
end
function restructure!(mc, G::StructArray; kwargs...)
    restructure!(mc, G.re; kwargs...)
    restructure!(mc, G.im; kwargs...)
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
        indices = CartesianIndices(l:Ls) # one based matrix indices
        
        @inbounds @fastmath for σ in measurement.flavor_iterator
            for b2 in eachindex(l.unitcell.sites), b1 in eachindex(l.unitcell.sites)
                uc1 = N * (b1-1)
                uc2 = N * (b2-1)
                for cdir in indices
                    # one based: mod1.(l.Ls + 2 - distance, l.Ls) 
                    # maps (1 -> 1, 2 -> L, 3 -> L-1, ...)
                    @. crev = ifelse(cdir.I > 1, l.Ls + 2 - cdir.I, 1)

                    # map to linear indices
                    dir = _sub2ind(l.Ls, cdir.I) + uc2
                    rev = _sub2ind(l.Ls, crev) + uc1

                    for csrc in indices
                        # cdist is one based so -1
                        # results is 0 < idx < 2L, so we can use ifelse as mod1
                        ctrg = csrc.I .+ cdist.I .- 1
                        @. ctrg = ifelse(ctrg > l.Ls, ctrg - l.Ls, ctrg)

                        # to linear index
                        src = _sub2ind(l.Ls, csrc.I) + uc1
                        trg = _sub2ind(l.Ls, ctrg) + uc2

                        # measure
                        temp[dir] += weight * measurement.kernel(
                            mc, mc.model, (src, trg), (dir, rev), packed_greens, σ
                        )
                    end # src loop
                end # dir loop
            end # basis loops
        end # flavor loop
    end # benchmark

    nothing
end

function _sub2ind0(Ns, idxs)
    idx = idxs[end]
    @inbounds for d in length(idxs)-1:-1:1
        idx = idx * Ns[d] + idxs[d]
    end
    return idx
end

@generated function _sub2ind0(Ns::T, idxs::T) where T
    if Ns <: NTuple{1}
        :(@inbounds return idxs[1])
    elseif Ns <: NTuple{2}
        :(@inbounds return muladd(Ns[1], idxs[2], idxs[1]))
    elseif Ns <: NTuple{3}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2], idxs[1])
            @inbounds return muladd(Ns[2], idxs[3], idx)
        end
    elseif Ns <: NTuple{4}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2], idxs[1])
            @inbounds idx = muladd(Ns[2], idxs[3], idx)
            @inbounds return muladd(Ns[3], idxs[4], idx)
        end
    else
        quote
            idx = idxs[end]
            @inbounds for d in length(idxs)-1:-1:1
                idx = idx * Ns[d] + idxs[d]
            end
            return idx
        end
    end
end

@generated function _sub2ind(Ns::T, idxs::T) where T
    if Ns <: NTuple{1}
        :(@inbounds return idxs[1])
    elseif Ns <: NTuple{2}
        :(@inbounds return muladd(Ns[1], idxs[2]-1, idxs[1]))
    elseif Ns <: NTuple{3}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2]-1, idxs[1])
            @inbounds return muladd(Ns[2], idxs[3]-1, idx)
        end
    elseif Ns <: NTuple{4}
        quote
            @inbounds idx = muladd(Ns[1], idxs[2]-1, idxs[1])
            @inbounds idx = muladd(Ns[2], idxs[3]-1, idx)
            @inbounds return muladd(Ns[3], idxs[4]-1, idx)
        end
    else
        quote
            #@boundscheck length(idxs)-2 < length(Ns)
            @inbounds idx = idxs[end]-1
            @inbounds for d in length(idxs)-1:-1:1
                idx = muladd(idx, Ns[d], idxs[d]-1)
            end
            return idx
        end
    end
end


# TODO: generalize to D-dimensional
function apply!(
        temp::Array, ::Restructure{<: EachBondPairByBravaisDistance}, 
        measurement::DQMCMeasurement, mc::DQMC, packed_greens, weight = 1.0
    )

    @timeit_debug "apply!(::Restructure{EachBondPairByBravaisDistance}, ::$(typeof(measurement.kernel)))" begin
        l = lattice(mc)
        Lx, Ly = l.Ls
        N = length(Bravais(l))
        bs = l.unitcell.bonds

        #           x, y
        # trg1 ---- src1 ---- src2 ---- trg2
        #    dx1, dy1   dx, dy   dx2, dy2
        #       b1                  b2
        # uc12      uc11      uc21      uc22

        G00, G0l, Gl0, Gll = packed_greens

        T0 = zero(eltype(temp))
        T1 = one(eltype(temp))
        directions = CartesianIndices(map(L -> 0:L-1, l.Ls)) # zero based
        indices = CartesianIndices(l.Ls) # one based

        @inbounds @fastmath for σ in measurement.flavor_iterator
            # Assuming 2D
            for (bi, b1) in enumerate(bs)
                # Note: 
                # we can always use this form and we require it when calculating 
                # flat indices/distances, so we're doing it early here.
                uc11 = N * (from(b1) - 1)
                uc12 = N * (to(b1) - 1)
                # dx1, dy1 = b1.uc_shift

                # mod0
                # dx1 = ifelse(dx1 < 0, Lx + dx1, dx1)
                # dy1 = ifelse(dy1 < 0, Ly + dy1, dy1)
                sub_d1 = @. ifelse(b1.uc_shift < 0, l.Ls + b1.uc_shift, b1.uc_shift)

                # one based; direction + target unitcell
                # d1 = 1 + dx1 + Lx * dy1 #+ uc12
                d1 = _sub2ind0(l.Ls, sub_d1) + 1 + uc12 # <--

                for (bj, b2) in enumerate(bs)
                    uc21 = N * (from(b2) - 1)
                    uc22 = N * (to(b2) - 1)
                    # dx2, dy2 = b2.uc_shift

                    # mod0
                    # dx2 = ifelse(dx2 < 0, Lx + dx2, dx2)
                    # dy2 = ifelse(dy2 < 0, Ly + dy2, dy2)
                    sub_d2 = @. ifelse(b2.uc_shift < 0, l.Ls + b2.uc_shift, b2.uc_shift)

                    # d2 = 1 + dx2 + Lx * dy2 #+ uc22
                    d2 = _sub2ind0(l.Ls, sub_d2) + 1 + uc22 # <--

                    # for dy in 0:Ly-1, dx in 0:Lx-1
                    for sub_d in directions # zero based
                        # dx, dy = sub_d.I
                        val = T0
                        # sub_d = CartesianIndex(dx, dy)

                        # Inlined mod1, see comment below
                        # src2 -> trg1
                        # -Lx ≤ (px2, py2) = b1 - (dx, dy) ≤ Lx
                        # px1 = dx1 - dx; px1 = ifelse(px1 < 0, px1 + Lx, px1)
                        # py1 = dy1 - dy; py1 = ifelse(py1 < 0, py1 + Ly, py1)
                        # p1 = 1 + px1 + Lx * py1 #+ uc12

                        # sub_d1 = (dx1, dy1)
                        sub_p1 = @. sub_d1 - sub_d.I
                        sub_p1 = @. ifelse(sub_p1 < 0, sub_p1 + l.Ls, sub_p1) 
                        p1 = _sub2ind0(l.Ls, sub_p1) + 1 + uc12 # <--
                        # p1 = _sub2ind(l.Ls, (px1, py1))

                        # src1 -> trg2
                        # 1 ≤ (px1, py1) = (dx, dy) + b2 ≤ 2Lx
                        # px2 = dx + dx2; px2 = ifelse(px2 ≥ Lx, px2 - Lx, px2)
                        # py2 = dy + dy2; py2 = ifelse(py2 ≥ Ly, py2 - Ly, py2)
                        # p2 = 1 + px2 + Lx * py2 #+ uc22

                        # sub_d2 = (dx2, dy2)
                        # sub_d = CartesianIndex((dx, dy))
                        sub_p2 = @. sub_d2 + sub_d.I
                        sub_p2 = @. ifelse(sub_p2 ≥ l.Ls, sub_p2 - l.Ls, sub_p2) 
                        p2 = _sub2ind0(l.Ls, sub_p2) + 1 + uc22 # <--
                        # p2 = 1 + px2 + Lx * py2 #+ uc22


                        I3 = ifelse(
                            (uc11 == uc22) && (p2 == uc22+1) && (G0l.l == G0l.k),
                            T1, T0
                        )

                        # Slow as fuck boiii
                        @simd for sub_src in indices
                            # linear src index
                            i = _sub2ind(l.Ls, sub_src.I) + uc11 # <--
                            # mod1(src + dir, Ls)
                            sub_src2 = @. sub_src.I + sub_d.I
                            sub_src2 = @. ifelse(sub_src2 > l.Ls, sub_src2 - l.Ls, sub_src2)
                            i2 = _sub2ind(l.Ls, sub_src2) + uc21 # <--

                            val += temp_kernel(packed_greens, (i, i2, d1, d2, p1, p2), I3)
                            # val += cc_kernel(
                            #     mc, mc.model, (i, i2), (d1, d2, p1, p2),
                            #     (uc11, uc12, uc21, uc22),
                            #     packed_greens, σ
                            # )
                        end

                        dx, dy = sub_d.I

                        # @simd for y in 1:Ly
                        #     # This is a repalcement for mod1(y+dy, Ly). Another 
                        #     # alternative is explicit loop splitting, which is
                        #     # used for the x loop. For the y loop it doesn't 
                        #     # seem to matter if we use ifelse or loop splitting,
                        #     # for x loop splitting is faster.
                        #     y2 = ifelse(y > Ly - dy, y - Ly, y)
                        #     for x in 1:Lx-dx
                        #         i = x + Lx * (y-1) #+ uc11
                        #         i2 = x + dx + Lx * (y2 + dy - 1) #+ uc21

                        #         # initial version:
                        #         # 4 * (I1 - Gll.val.val[i, d1]) * (I2 - G00.val.val[i2, d2]) +
                        #         # 2 * (I3 - G0l.val.val[i2, p1]) * Gl0.val.val[i, p2]

                        #         val += cc_kernel(
                        #             mc, mc.model, (i, i2), (d1, d2, p1, p2),
                        #             (uc11, uc12, uc21, uc22),
                        #             packed_greens, σ
                        #         )
                        #     end

                        #     # in this loop dx > 0
                        #     for x in Lx-dx+1:Lx
                        #         i  = x + Lx * (y-1) #+ uc11
                        #         # needs a -Lx which is here  ⤵
                        #         i2 = x + dx + Lx * (y2 + dy - 2) #+ uc21

                        #         val += cc_kernel(
                        #             mc, mc.model, (i, i2), (d1, d2, p1, p2),
                        #             (uc11, uc12, uc21, uc22),
                        #             packed_greens, σ
                        #         )
                        #     end
                        # end # source loops

                        temp[dx+1, dy+1, bi, bj] += weight * val

                    end # main distance loop

                end # selected distance
            end # selected distance

        end # flavor
    end # benchmark

    return 
end



################################################################################
### kernels
################################################################################



# Adjust this to take idxs w/o uc, uc_shifts (i.e. N* (uc-1)) and flavor
# then calculate I3 in here?
@inline Base.@propagate_inbounds function cc_kernel(
        mc::DQMC, ::Model, 
        sources::NTuple{2, Int},
        directions::NTuple{4, Int},
        uc_shifts::NTuple{4, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix},
        flavors::NTuple{2, Int},
    )
    N = length(lattice(mc))
    G00, G0l, Gl0, Gll = packed_greens
    i, k = sources
    Δij, Δkl, Δkj, Δil = directions
    uc11, uc12, uc21, uc22 = uc_shifts
    f1, f2 = flavors

    # I3 = ifelse((Δil == 0) && (uc11 == uc22) && (G0l.l == 0) && (f1 == f2), 1, 0)
    # 3 ==, 2 &&, 1 cast
    I3 = Int((Δil == 0) && (uc11 == uc22) && (G0l.l == 0))
    # I3 = 1

    # uc12 (j)   (l) uc22
    #       | \ / |
    #       | / \ |
    # uc11 (i)---(k) uc21

    # 6 +
    i += uc11
    k += uc21
    Δij += uc12
    Δkl += uc22
    Δkj += uc12
    Δil += uc22

    # 4 *, 2 ±, 4 getindex -> 4x (1 *, 1 +)
    # 8 *, 6 ±, 4 read
    4 * Gll.val.val[k, Δkl] * G00.val.val[i, Δij] + 
    2 * (I3 - G0l.val.val[i, Δil]) * Gl0.val.val[k, Δkj]
end

@inline Base.@propagate_inbounds function temp_kernel(packed_greens, idxs, I3) 
    G00, G0l, Gl0, Gll = packed_greens
    i, i2, d1, d2, p1, p2 = idxs
    #  (j)   (l)
    #   | \ / |
    #   | / \ |
    #  (i)---(k)
    # i, k, Δij, Δkl, Δkj, Δil = idxs

    4 * Gll.val.val[i2, d2] * G00.val.val[i, d1] + 
    2 * (I3 - G0l.val.val[i, p2]) * Gl0.val.val[i2, p1]
end





################################################################################
### Old prototyping
################################################################################



# const restructure_cache = Ref(-1)
# Bravais_modcache(l::Lattice) = vcat(1:prod(l.Ls), 1:prod(l.Ls))

# """
#     restructure(output, input, cached_mod)
#     restructure(output, greens_matrix, l::Lattice)

# Takes a Matrix `input` and reorders it into diagonals, saved in `output`. After 
# the transformation `output[i, d]` is `input[i, mod1(i+d-1, size(input, 2))]`, 
# i.e. `d` picks a (periodic) diagonal  and `i` an element of the diagonal. 

# `cached_mod` should be `vcat(1:size(input, 2), 1:size(input, 2))` and is 
# automatically generated from `l::Lattice`.
# """
# function restructure!(Q::AbstractMatrix, G::AbstractMatrix, cached_mod)
#     @inbounds for j in axes(Q, 2)
#         @simd for i in axes(Q, 1)
#             Q[i, j] = G[i, cached_mod[j + i - 1]]
#         end
#     end
#     return Q
# end

# function restructure!(Q::Matrix, G::Matrix, l::Lattice)
#     n = length(Bravais(l))
#     cached_mod = get!(l, :Bravais_modcache, Bravais_modcache)::Vector{Int}

#     if size(G, 1) == n
#         return restructure!(Q, G, cached_mod)
#     else
#         # we want to work on views
#         # - flv already works as an offsets so restructuring views allows this
#         #   to continue
#         # - doing the same for bravais indices would be simple/nice
#         # - since we transform diagonals to rows, rows 2:end would grab from 
#         #   multiple basis combinations. That's not what we want?
#         for i in 1:div(size(G, 1), n), j in 1:div(size(G, 2), n)
#             xr = (i-1) * n + 1 : i * n
#             yr = (j-1) * n + 1 : j * n
#             restructure!(view(Q, xr, yr), view(G, xr, yr), cached_mod)
#         end
#     end
#     return Q
# end

# function restructure!(Q::GreensMatrix, G::GreensMatrix, l::Lattice)
#     return restructure!(Q.val, G.val, l)
# end

# function restructure!(Q::BlockDiagonal, G::BlockDiagonal, l::Lattice)
#     for i in eachidnex(G.blocks)
#         restructure!(Q.blocks[i], G.blocks[i], l)
#     end
#     return Q
# end

# function restructure!(Q::CMat64, G::CMat64, l::Lattice)
#     restructure!(Q.re, G.re, l)
#     restructure!(Q.im, G.im, l)
#     return Q
# end
