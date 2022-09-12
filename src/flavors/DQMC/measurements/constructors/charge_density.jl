function charge_density(
        mc::DQMC, model::Model, greens_iterator; 
        lattice_iterator = EachSitePairByDistance(), wrapper = nothing, 
        flavor_iterator = FlavorIterator(mc, 2),
        kernel = full_cdc_kernel, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    Measurement(mc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end

charge_density_correlation(mc, m; kwargs...) = charge_density(mc, m, Greens(); kwargs...)
charge_density_susceptibility(mc, m; kwargs...) = charge_density(mc, m, TimeIntegral(mc); kwargs...)



################################################################################
### Full CDC kernel
################################################################################


"""
    full_cdc_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Computes `⟨nᵢ(τ) nⱼ(0)⟩` for the given indices.
"""
@inline Base.@propagate_inbounds function full_cdc_kernel(mc, model, ij, G::GreensMatrix, flv)
    return full_cdc_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
    f1, f2 = flv
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    
    id = I[i, j] * I[G0l.k, G0l.l] * I[f1, f2]
    s1 = N * (f1 - 1)
    s2 = N * (f2 - 1)

    # ∑_{σ₁, σ₂} ⟨n_{σ₁}(l) n_{σ₂}(0)⟩ =
    return (1 - Gll.val[i+s1, i+s1]) * (1 - G00.val[j+s2, j+s2]) +
            (id - G0l.val[j+s1, i+s2]) * Gl0.val[i+s1, j+s2]
end

@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, ij::NTuple{2}, 
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flvs
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    flv = total_flavors(mc.model)

    return flv * flv * (1 - Gll.val.val[i, i]) * (1 - G00.val.val[j, j]) +
            flv * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function full_cdc_kernel(
        ::DQMC, ::Model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
    f1, f2 = flv
	G00, G0l, Gl0, Gll = packed_greens    
    id = I[i, j] * I[G0l.k, G0l.l]

    output = (1 - Gll.val.blocks[f1][i, i]) * (1 - G00.val.blocks[f2][j, j])
    if f1 == f2
        output += (id - G0l.val.blocks[f1][j, i]) * Gl0.val.blocks[f1][i, j]
    end
    return output
end


################################################################################
### Reduced CDC kernel
################################################################################


"""
    reduced_cdc_kernel(mc, model, site_indices, greens_matrices, flavor_indices)

Computes `⟨nᵢ(τ) nⱼ(0)⟩ - ⟨nᵢ(τ)⟩⟨nⱼ(0)⟩` for the given indices.
"""
@inline Base.@propagate_inbounds function reduced_cdc_kernel(mc, model, ij, G::GreensMatrix, flv)
    return reduced_cdc_kernel(mc, model, ij, (G, G, G, G), flv)
end

@inline Base.@propagate_inbounds function reduced_cdc_kernel(
        mc, ::Model, ij::NTuple{2}, packed_greens::_GM4{<: Matrix}, flv
    )
    i, j = ij
    f1, f2 = flv
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    
    id = I[i, j] * I[G0l.k, G0l.l] * I[f1, f2]
    s1 = N * (f1 - 1)
    s2 = N * (f2 - 1)

    return (id - G0l.val[j+s1, i+s2]) * Gl0.val[i+s1, j+s2]
end

@inline Base.@propagate_inbounds function reduced_cdc_kernel(
        mc, ::Model, ij::NTuple{2}, 
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flvs
    )
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    id = I[i, j] * I[G0l.k, G0l.l]
    flv = total_flavors(mc.model)

    return flv * (id - G0l.val.val[j, i]) * Gl0.val.val[i, j]
end

@inline Base.@propagate_inbounds function reduced_cdc_kernel(
        ::DQMC, ::Model, ij::NTuple{2}, packed_greens::_GM4{<: BlockDiagonal}, flv
    )
    i, j = ij
    f1, f2 = flv
	G00, G0l, Gl0, Gll = packed_greens    
    id = I[i, j] * I[G0l.k, G0l.l]

    output = zero(eltype(G0l.val.blocks[1]))
    if f1 == f2
        output += (id - G0l.val.blocks[f1][j, i]) * Gl0.val.blocks[f1][i, j]
    end
    return output
end