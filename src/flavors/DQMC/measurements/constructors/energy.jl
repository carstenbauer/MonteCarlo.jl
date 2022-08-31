function boson_energy_measurement(dqmc, model; kwargs...)
    Measurement(dqmc, model, Nothing, nothing, energy_boson; kwargs...)
end

@deprecate noninteracting_energy kinetic_energy
function kinetic_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = kinetic_energy_kernel,
        kwargs...
    )
    Measurement(
        dqmc, model, greens_iterator, lattice_iterator, flavor_iterator, 
        kernel; kwargs...
    )
end

# These require the model to implement intE_kernel
function interacting_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = interaction_energy_kernel,
        kwargs...
    )
    Measurement(
        dqmc, model, greens_iterator, lattice_iterator, flavor_iterator, 
        kernel; kwargs...
    )
end

function total_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = total_energy_kernel,
        kwargs...
    )
    Measurement(
        dqmc, model, greens_iterator, lattice_iterator, flavor_iterator, 
        kernel; kwargs...
    )
end


################################################################################
### kernels
################################################################################

@deprecate nonintE_kernel kinetic_energy_kernel false
@deprecate intE_kernel interaction_energy_kernel false
@deprecate totalE_kernel total_energy_kernel false

@inline Base.@propagate_inbounds function kinetic_energy_kernel(mc, model, ::Nothing, G::_GM{<: Matrix}, flv)
    # <T> = \sum Tji * (Iij - Gij) = - \sum Tji * (Gij - Iij)
    T = mc.stack.hopping_matrix
    output = zero(eltype(G.val))
    @inbounds @fastmath for i in axes(G.val, 1), j in axes(G.val, 2)
        # using T Hermitian for better cache friendliness (conj(Tᵢⱼ) = Tⱼᵢ)
        output += conj(T[i, j]) * (I[i, j] - G.val[i, j])
    end
    output
end

@inline Base.@propagate_inbounds function kinetic_energy_kernel(mc, model, ::Nothing, G::_GM{<: DiagonallyRepeatingMatrix}, flv)
    T = mc.stack.hopping_matrix
    output = zero(eltype(G.val.val))
    @inbounds @fastmath for i in axes(G.val.val, 1), j in axes(G.val.val, 2)
        output += conj(T[i, j]) * (I[i, j] - G.val.val[i, j])
    end
    
    return 2.0 * output
end

@inline Base.@propagate_inbounds function kinetic_energy_kernel(mc, model, ::Nothing, G::_GM{<: BlockDiagonal}, flv)
    T = mc.stack.hopping_matrix
    output = zero(eltype(G.val.blocks[1]))

    @inbounds @fastmath for b in eachindex(G.val.blocks)
        g = G.val.blocks[b]
        for i in axes(g, 1), j in axes(g, 2)
            output += conj(T.blocks[b][i, j]) * (I[i, j] - g[i, j])
        end
    end
        
    return output
end


@inline Base.@propagate_inbounds function total_energy_kernel(mc, model, sites, G, flv)
    kinetic_energy_kernel(mc, model, sites, G, flv) + 
    interaction_energy_kernel(mc, model, sites, G, flv)
end
