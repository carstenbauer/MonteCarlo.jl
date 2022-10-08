function pairing(
        dqmc::DQMC, model::Model, greens_iterator; 
        K = 1 + nearest_neighbor_count(lattice(dqmc)),
        lattice_iterator = EachLocalQuadByDistance(1:K), wrapper = nothing,  
        flavor_iterator = FlavorIterator(dqmc, 0), 
        kernel = pc_combined_kernel, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    DQMCMeasurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end

"""
    pairing_correlation(mc, model; kwargs...)

Generates an equal-time pairing correlation measurement. Note that the 
measurement needs to be added to the simulation via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = pc_combined_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `pc_combined_kernel`, `pc_alt_kernel`
 and `pc_alt_kernel`.
- `lattice_iterator = EachLocalQuadByDistance(1:K)` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `K = 1 + nearest_neighbor_count(lattice(mc))` sets the number of directions to
include in the lattice iteration. This includes onsite as the [0, 0] direction.
- `flavor_iterator = FlavorIterator(mc, 0)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
pairing_correlation(mc, m; kwargs...) = pairing(mc, m, Greens(); kwargs...)

"""
    pairing_correlation(mc, model; kwargs...)

Generates an time-integrated pairing susceptibility measurement. Note that the 
measurement needs to be added to the simulation via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = pc_combined_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `pc_combined_kernel`, `pc_alt_kernel`
 and `pc_alt_kernel`.
- `lattice_iterator = EachLocalQuadByDistance(1:K)` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `K = 1 + nearest_neighbor_count(lattice(mc))` sets the number of directions to
include in the lattice iteration. This includes onsite as the [0, 0] direction.
- `flavor_iterator = FlavorIterator(mc, 0)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
pairing_susceptibility(mc, m; kwargs...) = pairing(mc, m, TimeIntegral(mc); kwargs...)



################################################################################
### Methods
################################################################################


"""
    pc_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨Δ(src1, trg1)(τ) Δ^†(src2, trg2)(0)⟩ for a given set of indices,
where Δ(src, trg)(τ) = c_{src, ↑}(τ) c_{trg, ↓}(τ). 
"""
@inline Base.@propagate_inbounds function pc_kernel(
        mc, model, sites::NTuple{4}, packed_greens::_GM4, flv
    )
    # We only need Gl0
    return pc_kernel(mc, model, sites, packed_greens[3], flv)
end


# implementations
@inline Base.@propagate_inbounds function pc_kernel(
        mc, ::Model, sites::NTuple{4}, Gl0::_GM{<: Matrix}, flv
    )
    src1, trg1, src2, trg2 = sites
    N = length(lattice(mc))
    # Δ_v(src1, trg1)(τ) Δ_v^†(src2, trg2)(0)
    # G_{i, j}^{↑, ↑}(τ, 0) G_{i+d, j+d'}^{↓, ↓}(τ, 0) - 
    # G_{i, j+d'}^{↑, ↓}(τ, 0) G_{i+d, j}^{↓, ↑}(τ, 0)
    
    Gl0.val[src1, src2] * Gl0.val[trg1+N, trg2+N] - 
    Gl0.val[src1, trg2+N] * Gl0.val[trg1+N, src2]
end

@inline Base.@propagate_inbounds function pc_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, Gl0::_GM{<: DiagonallyRepeatingMatrix}, flv
    )
    src1, trg1, src2, trg2 = sites
    Gl0.val.val[src1, src2] * Gl0.val.val[trg1, trg2]
end

@inline Base.@propagate_inbounds function pc_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, Gl0::_GM{<: BlockDiagonal}, flv
    )
    src1, trg1, src2, trg2 = sites
    Gl0.val.blocks[1][src1, src2] * Gl0.val.blocks[2][trg1, trg2]
end


"""
    pc_alt_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨Δ^†(src1, trg1)(τ) Δ(src2, trg2)(0)⟩ for a given set of indices,
where Δ(src, trg)(τ) = c_{src, ↑}(τ) c_{trg, ↓}(τ). 
"""
@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, model, sites::NTuple{4}, packed_greens::_GM4, flv
    )
    # here it's always G0l
    return pc_alt_kernel(mc, model, sites, packed_greens[2], flv)
end


@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, ::Model, sites::NTuple{4}, Gl0::_GM{<: Matrix}, flv
    )
    
    src1, trg1, src2, trg2 = sites
    N = length(lattice(mc))
    # Δ_v^†(src1, trg1)(τ) Δ_v(src2, trg2)(0)
    # (I-G)_{j, i}^{↑, ↑}(0, τ) (I-G)_{j+d', i+d}^{↓, ↓}(0, τ) - 
    # (I-G)_{j, i+d}^{↑, ↓}(0, τ) G_{j+d', i}^{↓, ↑}(0, τ)
    (I[trg1, trg2] * I[Gl0.k, Gl0.l] - Gl0.val[trg2+N, trg1+N]) * 
    (I[src1, src2] * I[Gl0.k, Gl0.l] - Gl0.val[src2, src1]) -
    Gl0.val[src2, trg1+N] * Gl0.val[trg2+N, src1]
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, Gl0::_GM{<: DiagonallyRepeatingMatrix}, flv
    )

    src1, trg1, src2, trg2 = sites
    (I[trg1, trg2] * I[Gl0.k, Gl0.l] - Gl0.val.val[trg2, trg1]) * 
    (I[src1, src2] * I[Gl0.k, Gl0.l] - Gl0.val.val[src2, src1])
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, Gl0::_GM{<: BlockDiagonal}, flv
    )

    src1, trg1, src2, trg2 = sites
    (I[trg1, trg2] * I[Gl0.k, Gl0.l] - Gl0.val.blocks[2][trg2, trg1]) * 
    (I[src1, src2] * I[Gl0.k, Gl0.l] - Gl0.val.blocks[1][src2, src1])
end

"""
    pc_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Calculates ⟨Δ(src1, trg1)(τ) Δ^†(src2, trg2)(0) + Δ^†(src1, trg1)(τ) Δ(src2, trg2)(0)⟩ 
for a given set of indices, where Δ(src, trg)(τ) = c_{src, ↑}(τ) c_{trg, ↓}(τ). 
This is the combination of `pc_kernel` and `pc_alt_kernel`
"""
@inline Base.@propagate_inbounds function pc_combined_kernel(mc, model, sites::NTuple{4}, G, flv)
    # Δ^† Δ + Δ Δ^†
    # same as in https://arxiv.org/pdf/1912.08848.pdf
    pc_kernel(mc, model, sites, G, flv) + pc_alt_kernel(mc, model, sites, G, flv)
end


################################################################################
### Distance based Methods
################################################################################


@inline Base.@propagate_inbounds function pc_kernel(
        mc, model, sources, dirs, ucs, packed_greens::_GM4, flv
    )
    return pc_kernel(mc, model, sources, dirs, ucs, packed_greens[3], flv)
end


@inline Base.@propagate_inbounds function pc_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: Matrix}, flv
    )
    N = length(lattice(mc))
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions
    
    Gl0.val[i, Δik] * Gl0.val[j+N, Δjl+N] - 
    Gl0.val[i, Δil+N] * Gl0.val[j+N, Δjk]
end

@inline Base.@propagate_inbounds function pc_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions

    Gl0.val.val[i, Δik] * Gl0.val.val[j, Δjl]
end

@inline Base.@propagate_inbounds function pc_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: BlockDiagonal}, flv
    )
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions

    Gl0.val.blocks[1][i, Δik] * Gl0.val.blocks[2][j, Δjl]
end


@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, model, sources, dirs, ucs, packed_greens::_GM4, flv
    )
    return pc_alt_kernel(mc, model, sources, dirs, ucs, packed_greens[2], flv)
end


@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: Matrix}, flv
    )
    N = length(lattice(mc))
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions
    uc11, uc12, uc21, uc22 = uc_shifts

    I1 = Int((Δjl == 1+uc22) && (uc12 == uc22) && (Gl0.k == 0))
    I2 = Int((Δik == 1+uc21) && (uc11 == uc21) && (Gl0.k == 0))

    (I1 - Gl0.val[l+N, Δlj+N]) * (I2 - Gl0.val[k, Δki]) -
    Gl0.val[k, Δkj+N] * Gl0.val[l+N, Δli]
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: DiagonallyRepeatingMatrix}, flv
    )
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions
    uc11, uc12, uc21, uc22 = uc_shifts

    I1 = Int((Δlj == 1+uc12) && (uc12 == uc22) && (Gl0.k == Gl0.l))
    I2 = Int((Δki == 1+uc11) && (uc11 == uc21) && (Gl0.k == Gl0.l))

    (I1 - Gl0.val.val[l, Δlj]) * (I2 - Gl0.val.val[k, Δki])
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, ::Model, sources::NTuple{4}, directions::NTuple{12, Int},
        uc_shifts::NTuple{4, Int}, Gl0::_GM{<: BlockDiagonal}, flv
    )
    i, j, k, l = sources
    Δij, Δik, Δil, Δji, Δjk, Δjl, Δki, Δkj, Δkl, Δli, Δlj, Δlk = directions
    uc11, uc12, uc21, uc22 = uc_shifts

    I1 = Int((Δlj == 1+uc12) && (uc12 == uc22) && (Gl0.k == 0))
    I2 = Int((Δki == 1+uc11) && (uc11 == uc21) && (Gl0.k == 0))

    (I1 - Gl0.val.blocks[2][l, Δlj]) * (I2 - Gl0.val.val.blocks[1][k, Δki])
end


# TODO discourage this
@inline Base.@propagate_inbounds function pc_combined_kernel(
        mc, model, sources, dirs, ucs, G, flv
    )
    pc_kernel(mc, model, sources, dirs, ucs, G, flv) + 
    pc_alt_kernel(mc, model, sources, dirs, ucs, G, flv)
end


################################################################################
### Old Methods
################################################################################


#=


"""
    pc_kernel(mc, model, ij::NTuple{4, Integer}, G::GreensMatrix)
    pc_kernel(mc, model, ij::NTuple{4, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair pairing `⟨Δᵢⱼ(τ) Δₖₗ^†(0)⟩` where `τ = 0` for the 
first signature and `Δᵢⱼ(τ) = 0.5 cᵢ↑(τ) cⱼ↓(τ)`. The Delta operator is a 
deferred version of the  pair field operator `Δᵢ = 0.5 ∑ₐ f(a) cᵢ↑ cᵢ₊ₐ↓` which 
leaves the execution of the sum for after the simulation. 

* Lattice Iterators: `EachLocalQuadByDistance` or `EachLocalQuadBySyncedDistance`
* Greens Iterators: `Greens`, `GreensAt`, `CombinedGreensIterator` or `TimeIntegral`
"""
function pc_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix, flv)
    pc_kernel(mc, model, sites, (G, G, G, G), flv)
end
function pc_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4}, flv)
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    # Δ_v(src1, trg1)(τ) Δ_v^†(src2, trg2)(0)
    # G_{i, j}^{↑, ↑}(τ, 0) G_{i+d, j+d'}^{↓, ↓}(τ, 0) - 
    # G_{i, j+d'}^{↑, ↓}(τ, 0) G_{i+d, j}^{↓, ↑}(τ, 0)
    Gl0.val[src1, src2] * Gl0.val[trg1+N, trg2+N] - 
    Gl0.val[src1, trg2+N] * Gl0.val[trg1+N, src2]
end


"""
    pc_alt_kernel(mc, model, ij::NTuple{4, Integer}, G::GreensMatrix)
    pc_alt_kernel(mc, model, ij::NTuple{4, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair pairing `⟨Δᵢⱼ^†(τ) Δₖₗ(0)⟩` where `τ = 0` for the 
first signature and `Δᵢⱼ(τ) = 0.5 cᵢ↑(τ) cⱼ↓(τ)`. The Delta operator is a 
deferred version of the  pair field operator `Δᵢ = 0.5 ∑ₐ f(a) cᵢ↑ cᵢ₊ₐ↓` which 
leaves the execution of the sum for after the simulation. 

* Lattice Iterators: `EachLocalQuadByDistance` or `EachLocalQuadBySyncedDistance`
* Greens Iterators: `Greens`, `GreensAt`, `CombinedGreensIterator` or `TimeIntegral`
"""
function pc_alt_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix, flv)
    pc_alt_kernel(mc, model, sites, (G, G, G, G), flv)
end
function pc_alt_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4}, flv)
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    # Δ_v^†(src1, trg1)(τ) Δ_v(src2, trg2)(0)
    # (I-G)_{j, i}^{↑, ↑}(0, τ) (I-G)_{j+d', i+d}^{↓, ↓}(0, τ) - 
    # (I-G)_{j, i+d}^{↑, ↓}(0, τ) G_{j+d', i}^{↓, ↑}(0, τ)
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val[trg2+N, trg1+N]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val[src2, src1]) -
    G0l[src2, trg1+N] * G0l[trg2+N, src1]
end

function pc_combined_kernel(mc, model, sites::NTuple{4}, G, flv)
    # Δ^† Δ + Δ Δ^†
    # same as in https://arxiv.org/pdf/1912.08848.pdf
    pc_kernel(mc, model, sites, G, flv) + pc_alt_kernel(mc, model, sites, G, flv)
end


function pc_ref_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix, flv)
    # Δ^† Δ + Δ Δ^† but ↑ and ↓ are swapped
    pc_ref_kernel(mc, model, sites, (G, G, G, G), flv)
end
function pc_ref_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4}, flv)
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    Gl0.val[src1+N, src2+N] * Gl0.val[trg1, trg2] - 
    Gl0.val[src1+N, trg2]   * Gl0.val[trg1, src2+N] +
    (I[trg2, trg1] * I[G0l.k, G0l.l] - G0l.val[trg2, trg1]) * 
    (I[src2, src1] * I[G0l.k, G0l.l] - G0l.val[src2+N, src1+N]) -
    (I[src2, trg1] * I[G0l.k, G0l.l] - G0l.val[src2+N, trg1]) * 
    (I[trg2, src1] * I[G0l.k, G0l.l] - G0l.val[trg2, src1+N])
end

function pc_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix, ::Val{1})
    src1, trg1, src2, trg2 = sites
    G.val[src1, src2] * G.val[trg1, trg2]
end
function pc_kernel(mc, model, sites::NTuple{4}, pg::NTuple{4}, ::Val{1})
    src1, trg1, src2, trg2 = sites
    pg[3].val[src1, src2] * pg[3].val[trg1, trg2]
end
function pc_alt_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4}, ::Val{1})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val[trg2, trg1]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val[src2, src1])
end
function pc_ref_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4}, ::Val{1})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    Gl0.val[src1, src2] * Gl0.val[trg1, trg2] +
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val[trg2, trg1]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val[src2, src1])
end
=#