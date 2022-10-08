function charge_density(
        mc::DQMC, model::Model, greens_iterator; 
        lattice_iterator = EachSitePairByDistance(), wrapper = nothing, 
        flavor_iterator = FlavorIterator(mc, 2),
        kernel = full_cdc_kernel, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper(lattice_iterator)
    DQMCMeasurement(mc, model, greens_iterator, li, flavor_iterator, kernel; kwargs...)
end

"""
    charge_density_correlation(mc, model; kwargs...)

Generates an equal-time charge density measurement. Note that the result needs
to be added to the simulation via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = full_cdc_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `full_cdc_kernel` and 
`reduced_cdc_kernel`.
- `lattice_iterator = EachSitePairByDistance()` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `flavor_iterator = FlavorIterator(mc, 2)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
charge_density_correlation(mc, m; kwargs...) = charge_density(mc, m, Greens(); kwargs...)

"""
    charge_density_susceptibility(mc, model; kwargs...)

Generates a time-integrated charge density measurement. Note that the result needs
to be added to the simulation via `mc[:name] = result`.

## Optional Keyword Arguments

- `kernel = full_cdc_kernel` sets the function representing the Wicks expanded 
expectation value of the measurement. See `full_cdc_kernel` and 
`reduced_cdc_kernel`.
- `lattice_iterator = EachSitePairByDistance()` controls which sites are passed 
to the kernel and how they are summed. See lattice iterators
- `flavor_iterator = FlavorIterator(mc, 2)` controls which flavor indices 
(spins) are passed to the kernel. This should generally not be changed.
- kwargs from `DQMCMeasurement`
"""
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

    # ⟨n_{σ₁}(l) n_{σ₂}(0)⟩ =
    #   ⟨n_{σ₁}(l)⟩ ⟨n_{σ₂}(0)⟩ + 
    #   ⟨c_{σ₁}^†(l) c_{σ₂}(0)⟩ ⟨c_{σ₁}(l) c_{σ₂}^†(0)⟩ =
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


################################################################################
### Distance based full CDC kernel
################################################################################



@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, sources, directions, uc_shifts,
        G::GreensMatrix, flavors
    )
    return full_cdc_kernel(mc, model, sources, directions, uc_shifts, (G, G, G, G), flavors)
end

@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: Matrix}, flavors::NTuple{2, Int}
    )
    N = length(lattice(mc))
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    f1, f2 = N .* (flavors .- 1)
    id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0) && (f1 == f2))

    return flv * flv * (1 - Gll.val[i+f1, 1+uc1+f1]) * (1 - G00.val[j+f2, 1+uc2+f2]) +
            flv * (id - G0l.val[j+f2, Δji+f1]) * Gl0.val[i+f1, Δij+f2]
end


@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: DiagonallyRepeatingMatrix}, flvs
    )
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    id = Int((Δji == 1+uc1) && (uc1 == uc2) && (G0l.l == 0))
    flv = total_flavors(mc.model)

    # println("$sources, $directions $uc_shifts $id")

    return flv * flv * (1 - Gll.val.val[i, 1+uc1]) * (1 - G00.val.val[j, 1+uc2]) +
            flv * (id - G0l.val.val[j, Δji]) * Gl0.val.val[i, Δij]
end


@inline Base.@propagate_inbounds function full_cdc_kernel(
        mc, ::Model, sources::NTuple{2}, 
        directions::NTuple{2, Int}, uc_shifts::NTuple{2, Int},
        packed_greens::_GM4{<: BlockDiagonal}, flvs
    )
    
    i, j = sources
    Δij, Δji = directions
    uc1, uc2 = uc_shifts
	G00, G0l, Gl0, Gll = packed_greens
    
    output = (1 - Gll.val.blocks[f1][i, 1+uc1]) * (1 - G00.val.blocks[f2][j, 1+uc2])
    if f1 == f2
        id = Int((Δij == 1+uc2) && (uc1 == uc2) && (G0l.l == 0))
        output += (id - G0l.val.blocks[f1][j, Δji]) * Gl0.val.blocks[f1][i, Δij]
    end
    return output
end