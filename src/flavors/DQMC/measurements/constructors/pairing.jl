function pairing(
        dqmc::DQMC, model::Model, greens_iterator; 
        K = 1 + nearest_neighbor_count(lattice(dqmc)),
        lattice_iterator = EachLocalQuadByDistance(1:K), wrapper = nothing,  
        flavor_iterator = FlavorIterator(dqmc, 0), 
        kernel = pc_combined_kernel, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    Measurement(dqmc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end
pairing_correlation(mc, m; kwargs...) = pairing(mc, m, Greens(); kwargs...)
pairing_susceptibility(mc, m; kwargs...) = pairing(mc, m, TimeIntegral(mc); kwargs...)



################################################################################
### Methods
################################################################################



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



@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, model, sites::NTuple{4}, packed_greens::_GM4, flv
    )
    # here it's always G0l
    return pc_alt_kernel(mc, model, sites, packed_greens[2], flv)
end


@inline Base.@propagate_inbounds function pc_alt_kernel(
        mc, ::Model, sites::NTuple{4}, G0l::_GM{<: Matrix}, flv
    )
    
    src1, trg1, src2, trg2 = sites
    N = length(lattice(mc))
    # Δ_v^†(src1, trg1)(τ) Δ_v(src2, trg2)(0)
    # (I-G)_{j, i}^{↑, ↑}(0, τ) (I-G)_{j+d', i+d}^{↓, ↓}(0, τ) - 
    # (I-G)_{j, i+d}^{↑, ↓}(0, τ) G_{j+d', i}^{↓, ↑}(0, τ)
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val[trg2+N, trg1+N]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val[src2, src1]) -
    G0l[src2, trg1+N] * G0l[trg2+N, src1]
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, G0l::_GM{<: DiagonallyRepeatingMatrix}, flv
    )

    src1, trg1, src2, trg2 = sites
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val.val[trg2, trg1]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val.val[src2, src1])
end

@inline Base.@propagate_inbounds function pc_alt_kernel(
        ::DQMC, ::Model, sites::NTuple{4}, G0l::_GM{<: BlockDiagonal}, flv
    )

    src1, trg1, src2, trg2 = sites
    (I[trg1, trg2] * I[G0l.k, G0l.l] - G0l.val.blocks[2][trg2, trg1]) * 
    (I[src1, src2] * I[G0l.k, G0l.l] - G0l.val.blocks[1][src2, src1])
end


@inline Base.@propagate_inbounds function pc_combined_kernel(mc, model, sites::NTuple{4}, G, flv)
    # Δ^† Δ + Δ Δ^†
    # same as in https://arxiv.org/pdf/1912.08848.pdf
    pc_kernel(mc, model, sites, G, flv) + pc_alt_kernel(mc, model, sites, G, flv)
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