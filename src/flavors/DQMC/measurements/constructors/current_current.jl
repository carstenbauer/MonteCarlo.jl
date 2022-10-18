"""
    current_current_susceptibility(mc, model; kwargs...)

Creates a measurement recordering the current-current susceptibility 
`OBS[dx, dy, bond1, bond2] = Λ_{bond1, bond2}(dx, dy) = 1/N ∑ᵢ ∫ ⟨j_{dr2}(i+(dx, dy), τ) j_{dr1}(i, 0)⟩ dτ`.

## Optional Keyword Arguments

- `kernel = cc_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement.
- `lattice_iterator = EachBondPairByBravaisDistance(mc)` controls which sites  
are passed to the kernel and how they are summed. By default this will consider
all bonds in the lattice without reverse bonds for `bond1, bond2` as well as all 
Bravais source sites i and all Bravais target sites i + (dx, dy). (The basis is 
included through bonds.)
- `flavor_iterator = FlavorIterator(mc, 2)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
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
    DQMCMeasurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end


################################################################################
### New kernels
################################################################################

# These kernels have been checked vs 
# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.68.2830 
# (more direct, simple square lattice, repulsive model)
# https://arxiv.org/pdf/1912.08848.pdf
# (less direct, square w/ basis, more bonds, repulsive model)


# I didn't think about the index order in this docstring carefully...
"""
    cc_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨j(τ, src1, trg1) j(0, src2, trg2)⟩ where j(τ, src, trg) = 
i T[src, trg] c_trg^†(τ) c_src(τ) - i T[trg, src] c_src^†(τ) c_trg(τ)
with T the hopping matrix.

With the default lattice iterator this is equivalent to 
⟨j_{b1}(τ, r + Δr) j_{b2}(0, r)⟩ where b1, b2 are bond indices.
"""
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
### Distance based kernels
################################################################################

# For reference:
# uc12 (j)   (l) uc22
#       | \ / |
#       | / \ |
# uc11 (i)---(k) uc21
# uc jumps in units of length(lattice), starting at 0.
# uc already included in positions and distances
# format: (x, y, ..., uc[, flv]) and (dx, dy, ..., uc[, flv])


# Basic full Matrix
@inline Base.@propagate_inbounds function cc_kernel(
        mc::DQMC, ::Model, 
        sources::NTuple{2, Int},
        directions::NTuple{4, Int},
        uc_shifts::NTuple{4, Int},
        packed_greens::_GM4{<: Matrix},
        flavors::NTuple{2, Int}
    )
    N = length(lattice(mc))
    G00, G0l, Gl0, Gll = packed_greens
    i, k = sources
    Δij, Δkl, Δkj, Δil = directions
    uc11, uc12, uc21, uc22 = uc_shifts
    f1, f2 = N .* (flavors .- 1)

    I3 = Int((Δil == 1+uc22) && (uc11 == uc22) && (G0l.l == 0) && (f1 == f2))

    return Gll.val[k+f2, Δkl+f2] * G00.val[i+f1, Δij+f1] + 
           (I3 - G0l.val[i+f1, Δil+f2]) * Gl0.val[k+f2, Δkj+f1]
end


# Repeating Matrix
@inline Base.@propagate_inbounds function cc_kernel(
        ::DQMC, m::Model, 
        sources::NTuple{2, Int},
        directions::NTuple{4, Int},
        uc_shifts::NTuple{4, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix},
        flavors,
    )
    G00, G0l, Gl0, Gll = packed_greens
    i, k = sources
    Δij, Δkl, Δkj, Δil = directions
    uc11, uc12, uc21, uc22 = uc_shifts
    flv = total_flavors(m)

    I3 = Int((Δil == 1+uc22) && (uc11 == uc22) && (G0l.k == G0l.l))

    return flv*flv * Gll.val.val[k, Δkl] * G00.val.val[i, Δij] + 
           flv * (I3 - G0l.val.val[i, Δil]) * Gl0.val.val[k, Δkj]
end


# BlockDiagonal Optimization
@inline Base.@propagate_inbounds function cc_kernel(
        mc::DQMC, ::Model, 
        sources::NTuple{2, Int},
        directions::NTuple{4, Int},
        uc_shifts::NTuple{4, Int},
        packed_greens::_GM4{<: BlockDiagonal},
        flavors::NTuple{2, Int}
    )
    G00, G0l, Gl0, Gll = packed_greens
    i, k = sources
    Δij, Δkl, Δkj, Δil = directions
    uc11, uc12, uc21, uc22 = uc_shifts
    f1, f2 = flavors

    output = Gll.val.blocks[f2][k, Δkl] * G00.val.blocks[f1][i, Δij]

    # Maybe (f1 == f2) * (...) is better?
    if f1 == f2
        I3 = Int((Δil == 1+uc22) && (uc11 == uc22) && (G0l.l == 0))
        output += (I3 - G0l.val.blocks[f1][i, Δil]) * Gl0.val.blocks[f1][k, Δkj]
    end

    return output
end