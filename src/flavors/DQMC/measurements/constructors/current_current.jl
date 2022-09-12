"""
    current_current_susceptibility(dqmc, model[; K = #NN + 1, kwargs...])

Creates a measurement recordering the current-current susceptibility 
`OBS[dr, dr1, dr2] = Λ_{dr1, dr2}(dr) = ∫ 1/N \\sum_ ⟨j_{dr2}(i+dr, τ) j_{dr1}(i, 0)⟩ dτ`.

By default this measurement synchronizes `dr1` and `dr2`, simplifying the 
formula to 
`OBS[dr, dr'] = Λ_{dr'}(dr) = ∫ 1/N \\sum_ ⟨j_{dr'}(i+dr, τ) j_{dr'}(i, 0)⟩ dτ`.
The direction/index `dr'` runs from `1:K` through the `directions(mc)` which are 
sorted. By default we include on-site (1) and nearest neighbors (2:K).

The current density `j_{dr}(i, τ)` with `j = i + dr` is given by
`j_{j-i}(i, τ) = i T_{ij} c_i^†(τ) c_j(0) + h.c.`. Note that the Hermitian
conjugate generates a term in `-dr` direction with prefactor `-i`. If `+dr` and
`-dr` are included in the calculation of the superfluid density (etc) you may
see larger values than expected. 

"""
function current_current_susceptibility(
        dqmc::DQMC, model::Model; 
        greens_iterator = TimeIntegral(dqmc), wrapper = nothing,
        lattice_iterator = EachBondPairByBravaisDistance(dqmc), 
        flavor_iterator = FlavorIterator(dqmc, 2), 
        kernel = cc_kernel,
        kwargs...
    )
    @assert is_approximately_hermitian(hopping_matrix(model)) "CCS assumes Hermitian matrix"
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    Measurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end


################################################################################
### New kernels
################################################################################

@inline Base.@propagate_inbounds function cc_kernel(
        mc, m::Model, sites, G::_GM, flv
    )
    return cc_kernel(mc, m, sites, (G, G, G, G), flv)
end


# Basic full Matrix
@inline Base.@propagate_inbounds function cc_kernel(
        mc, ::Model, sites::NTuple{4, Int}, 
        packed_greens::_GM4{<: Matrix}, 
        flv::NTuple{2, Int}
    )

    src1, trg1, src2, trg2 = sites
    G00, G0l, Gl0, Gll = packed_greens
    T = mc.stack.hopping_matrix
    id = I[G0l.k, G0l.l]

    N = length(lattice(mc))
    f1, f2 = flv
    s1 = src1 + N * (f1 - 1); s2 = src2 + N * (f2 - 1)
    t1 = trg1 + N * (f1 - 1); t2 = trg2 + N * (f2 - 1)

    # If G is BlockDiagonal T must be too.
    # T should always be hermitian
    # conj should beat second matrix access
    T1_st = T[s1, t1]
    T1_ts = conj(T1_st)
    T2_st = T[s2, t2]
    T2_ts = conj(T2_st)

    output = (
        (T2_ts * Gll.val[s2, t2] - T2_st * Gll.val[t2, s2]) * 
        (T1_st * G00.val[t1, s1] - T1_ts * G00.val[s1, t1])
    ) + (
        - T2_ts * T1_ts * (id * I[s1, t2] - G0l.val[s1, t2]) * Gl0.val[s2, t1] +
        + T2_ts * T1_st * (id * I[t1, t2] - G0l.val[t1, t2]) * Gl0.val[s2, s1] +
        + T2_st * T1_ts * (id * I[s1, s2] - G0l.val[s1, s2]) * Gl0.val[t2, t1] +
        - T2_st * T1_st * (id * I[s2, t1] - G0l.val[t1, s2]) * Gl0.val[t2, s1] 
    )
    
    return output
end


# Repeating Matrix
@inline Base.@propagate_inbounds function cc_kernel(
        mc, ::Model, sites::NTuple{4, Int}, 
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, 
        flvs::NTuple{2, Int}
    )
    f1, f2 = flvs
    @assert (f1 == 1) && (f2 == 1) "This method should only be called once"

    s1, t1, s2, t2 = sites
    G00, G0l, Gl0, Gll = packed_greens
    T = mc.stack.hopping_matrix
    id = I[G0l.k, G0l.l]
    flv = total_flavors(mc.model)

    # If G is BlockDiagonal T must be too.
    # T should always be hermitian
    # conj should beat second matrix access
    T1_st = T[s1, t1]
    T1_ts = conj(T1_st)
    T2_st = T[s2, t2]
    T2_ts = conj(T2_st)

    # The given G represents [G 0; 0 G]
    # the uncorrelated part always triggers -> flv^2
    # the correlated part only triggers on diagonal -> flv
    output = flv * flv * (
        (T2_ts * Gll.val.val[s2, t2] - T2_st * Gll.val.val[t2, s2]) * 
        (T1_st * G00.val.val[t1, s1] - T1_ts * G00.val.val[s1, t1])
    ) + flv * (
        - T2_ts * T1_ts * (id * I[s1, t2] - G0l.val.val[s1, t2]) * Gl0.val.val[s2, t1] +
        + T2_ts * T1_st * (id * I[t1, t2] - G0l.val.val[t1, t2]) * Gl0.val.val[s2, s1] +
        + T2_st * T1_ts * (id * I[s1, s2] - G0l.val.val[s1, s2]) * Gl0.val.val[t2, t1] +
        - T2_st * T1_st * (id * I[s2, t1] - G0l.val.val[t1, s2]) * Gl0.val.val[t2, s1] 
    )
    
    return output
end


# BlockDiagonal Optimization
@inline Base.@propagate_inbounds function cc_kernel(
        mc, ::Model, sites::NTuple{4, Int}, 
        packed_greens::_GM4{<: BlockDiagonal}, 
        flv::NTuple{2, Int}
    )

    f1, f2 = flv
    s1, t1, s2, t2 = sites
    G00, G0l, Gl0, Gll = packed_greens
    T = mc.stack.hopping_matrix
    id = I[G0l.k, G0l.l]

    # If G is BlockDiagonal T must be too.
    # T should always be hermitian
    # conj should beat second matrix access
    T1_st = T.blocks[f1][s1, t1]
    T1_ts = conj(T1_st)
    T2_st = T.blocks[f2][s2, t2]
    T2_ts = conj(T2_st)

    gll = Gll.val.blocks[f2] # called only with f2
    g00 = G00.val.blocks[f1] # called onyl with f1
    g0l = G0l.val.blocks[f1] # called with f1, f2
    gl0 = Gl0.val.blocks[f1] # called with f2, f1

    # correlated part always triggers
    output = (
        (T2_ts * gll[s2, t2] - T2_st * gll[t2, s2]) * 
        (T1_st * g00[t1, s1] - T1_ts * g00[s1, t1])
    )

    # uncorrelated part only triggers with f1 == f2 
    # since blockdiagonal is zero else
    # Branch prediction handles this well I think?
    if f1 == f2
        output += (
            - T2_ts * T1_ts * (id * I[s1, t2] - g0l[s1, t2]) * gl0[s2, t1] +
            + T2_ts * T1_st * (id * I[t1, t2] - g0l[t1, t2]) * gl0[s2, s1] +
            + T2_st * T1_ts * (id * I[s1, s2] - g0l[s1, s2]) * gl0[t2, t1] +
            - T2_st * T1_st * (id * I[s2, t1] - g0l[t1, s2]) * gl0[t2, s1] 
        )
    end
    
    return output
end


# StructArray fallback
@inline Base.@propagate_inbounds function cc_kernel(
        mc, ::Model, sites::NTuple{4, Int}, 
        packed_greens::_GM4{<: StructArray}, 
        flv::NTuple{2, Int}
    )

    @warn "StructArrays should be converted earlier" maxlog = 1

    f1, f2 = flv
    s1, t1, s2, t2 = sites
    G00, G0l, Gl0, Gll = packed_greens
    T = mc.stack.hopping_matrix
    id = I[G0l.k, G0l.l]

    # If G is BlockDiagonal T must be too.
    # T should always be hermitian
    # conj should beat second matrix access
    T1_st = T.blocks[f1][s1, t1]
    T1_ts = conj(T1_st)
    T2_st = T.blocks[f2][s2, t2]
    T2_ts = conj(T2_st)

    output = (
        (T2_ts * Gll.val[s2, t2] - T2_st * Gll.val[t2, s2]) * 
        (T1_st * G00.val[t1, s1] - T1_ts * G00.val[s1, t1])
    ) + (
        - T2_ts * T1_ts * (id * I[s1, t2] - G0l.val[s1, t2]) * Gl0.val[s2, t1] +
        + T2_ts * T1_st * (id * I[t1, t2] - G0l.val[t1, t2]) * Gl0.val[s2, s1] +
        + T2_st * T1_ts * (id * I[s1, s2] - G0l.val[s1, s2]) * Gl0.val[t2, t1] +
        - T2_st * T1_st * (id * I[s2, t1] - G0l.val[t1, s2]) * Gl0.val[t2, s1] 
    )
    
    return output
end





################################################################################
### Old kernels
################################################################################


# These kernels have been checked vs 
# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.68.2830 
# (more direct, simple square lattice, repulsive model)
# https://arxiv.org/pdf/1912.08848.pdf
# (less direct, square w/ basis, more bonds, repulsive model)

#=
@bm function cc_kernel(mc, ::Model, sites::NTuple{4}, packed_greens::NTuple{4}, flv)
    @warn "Unoptimized cc_kernel" maxlog=1
    # Computes
    # ⟨j_{t2-s2}(s2, l) j_{t1-s1}(s1, 0)⟩
    # where t2-s2 (t1-s1) is usually a NN vector/jump, and
    # j_{t2-s2}(s2, l) = i \sum_flv [T_{ts} c_t^†(l) c_s(τ) - T_{st} c_s^†(τ) c_t(τ)]
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    T = mc.stack.hopping_matrix
    output = zero(eltype(G00))
    id = I[G0l.k, G0l.l]

    # Iterate through (spin up, spin down)
    for f1 in (0, N), f2 in (0, N)
        s1 = src1 + f1; t1 = trg1 + f1
        s2 = src2 + f2; t2 = trg2 + f2

        # Optimizations
        # - combine first Wicks terms
        # - index into T only once, use conj instead of transpose
        # - use src != trg (i.e. I[s1, t1] = 0)
        T1_st = T[s1, t1]
        T1_ts = conj(T1_st)
        T2_st = T[s2, t2]
        T2_ts = conj(T2_st)

        # output += (
        #     (T[s2, t2] * (I[t2, s2] - Gll.val[t2, s2]) - T[t2, s2] * (I[t2, s2] - Gll.val[s2, t2])) * 
        #     (T[t1, s1] * (I[s1, t1] - G00.val[s1, t1]) - T[s1, t1] * (I[s1, t1] - G00.val[t1, s1])) +
        #     - T[t2, s2] * T[t1, s1] * (id * I[s1, t2] - G0l.val[s1, t2]) * Gl0.val[s2, t1] +
        #     + T[t2, s2] * T[s1, t1] * (id * I[t1, t2] - G0l.val[t1, t2]) * Gl0.val[s2, s1] +
        #     + T[s2, t2] * T[t1, s1] * (id * I[s1, s2] - G0l.val[s1, s2]) * Gl0.val[t2, t1] +
        #     - T[s2, t2] * T[s1, t1] * (id * I[s2, t1] - G0l.val[t1, s2]) * Gl0.val[t2, s1] 
        # )
        output += (
            (T2_ts * Gll.val[s2, t2] - T2_st * Gll.val[t2, s2]) * 
            (T1_st * G00.val[t1, s1] - T1_ts * G00.val[s1, t1]) +
            - T2_ts * T1_ts * (id * I[s1, t2] - G0l.val[s1, t2]) * Gl0.val[s2, t1] +
            + T2_ts * T1_st * (id * I[t1, t2] - G0l.val[t1, t2]) * Gl0.val[s2, s1] +
            + T2_st * T1_ts * (id * I[s1, s2] - G0l.val[s1, s2]) * Gl0.val[t2, t1] +
            - T2_st * T1_st * (id * I[s2, t1] - G0l.val[t1, s2]) * Gl0.val[t2, s1] 
        )
    end
    
    output
end


@bm function cc_kernel(mc, ::Model, sites::NTuple{4}, pg::NTuple{4}, ::Val{1})
    @warn "Unoptimized cc_kernel" maxlog=1
    src1, trg1, src2, trg2 = sites
    G00, G0l, Gl0, Gll = pg
    T = mc.stack.hopping_matrix
    id = I[G0l.k, G0l.l]

    # up-up counts, down-down counts, mixed only on 11s or 22s
    s1 = src1; t1 = trg1
    s2 = src2; t2 = trg2

    # See other method
    T1_st = T[s1, t1]
    T1_ts = conj(T1_st)
    T2_st = T[s2, t2]
    T2_ts = conj(T2_st)

    output = (
        ( T1_st * Gll[s1, t1] - T1_ts * Gll[t1, s1] ) * 
        ( T2_ts * G00[t2, s2] - T2_st * G00[s2, t2] ) +
        - T1_st * T2_st * (id * I[t1, s2] - G0l[s2, t1]) * Gl0[s1, t2] +
        + T1_ts * T2_st * (id * I[s1, s2] - G0l[s2, s1]) * Gl0[t1, t2] +
        + T1_st * T2_ts * (id * I[t1, t2] - G0l[t2, t1]) * Gl0[s1, s2] +
        - T1_ts * T2_ts * (id * I[s1, t2] - G0l[t2, s1]) * Gl0[t1, s2]
    )
    
    output
end
=#