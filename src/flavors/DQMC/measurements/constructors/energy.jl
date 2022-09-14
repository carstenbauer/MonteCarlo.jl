function boson_energy_measurement(dqmc, model; kwargs...)
    DQMCMeasurement(dqmc, model, nothing, nothing, nothing, energy_boson; kwargs...)
end

@deprecate noninteracting_energy kinetic_energy

"""
    kinetic_energy(mc, model; kwargs...)

Constructs a measurement of the kinetic energy terms of a model. Note that this 
includes all two-operator terms like hopping, chemical potential,
magnetic fields, etc.

## Optional Keyword Arguments

- `kernel = kinetic_energy_kernel` sets the function representing the Wicks 
expanded expectation value of the measurement. See `kinetic_energy_kernel`
- `lattice_iterator = nothing` controls which sites are passed 
to the kernel and how they are summed. With `nothing` this is left to the kernel.
- `flavor_iterator = nothing` controls which flavor indices 
(spins) are passed to the kernel. With `lattice_iterator = nothing` this is is 
also left to the kernel.
- kwargs from `DQMCMeasurement`
"""
function kinetic_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = kinetic_energy_kernel,
        kwargs...
    )
    DQMCMeasurement(
        dqmc, model, greens_iterator, lattice_iterator, flavor_iterator, 
        kernel; kwargs...
    )
end

# These require the model to implement interaction_energy_kernel
"""
    interaction_energy(mc, model; kwargs...)

Constructs a measurement of the interaction energy of the model. Note that this 
measurement requires `interaction_energy_kernel` to be implemented for the model
in question.

## Optional Keyword Arguments

- `kernel = interaction_energy_kernel` sets the function representing the Wicks 
expanded expectation value of the measurement. See `interaction_energy_kernel`
- `lattice_iterator = nothing` controls which sites are passed 
to the kernel and how they are summed. With `nothing` this is left to the kernel.
- `flavor_iterator = nothing` controls which flavor indices 
(spins) are passed to the kernel. With `lattice_iterator = nothing` this is is 
also left to the kernel.
- kwargs from `DQMCMeasurement`
"""
function interaction_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = interaction_energy_kernel,
        kwargs...
    )
    DQMCMeasurement(
        dqmc, model, greens_iterator, lattice_iterator, flavor_iterator, 
        kernel; kwargs...
    )
end

"""
    total_energy(mc, model; kwargs...)

Constructs a measurement of the full energy of the model. Note that this 
measurement requires `interaction_energy_kernel` to be implemented for the model
in question.

## Optional Keyword Arguments

- `kernel = total_energy_kernel` sets the function representing the Wicks 
expanded expectation value of the measurement. See `total_energy_kernel`
- `lattice_iterator = nothing` controls which sites are passed 
to the kernel and how they are summed. With `nothing` this is left to the kernel.
- `flavor_iterator = nothing` controls which flavor indices 
(spins) are passed to the kernel. With `lattice_iterator = nothing` this is is 
also left to the kernel.
- kwargs from `DQMCMeasurement`
"""
function total_energy(
        dqmc, model; 
        greens_iterator = Greens(),
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        kernel = total_energy_kernel,
        kwargs...
    )
    DQMCMeasurement(
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

# TODO Tij or Tji? I think Tji...?
"""
    kinetic_energy_kernel(mc, model, ::Nothing, greens_matrices, ::Nothing)

Computes the kinetic energy ⟨T[j, i] (I[i, j] - G[i, j])⟩ = ⟨tⱼᵢ cⱼ^† cᵢ⟩
"""
@inline Base.@propagate_inbounds function kinetic_energy_kernel(
        mc, model, ::Nothing, G::_GM{<: Matrix}, flv
    )
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

"""
    total_energy_kernel(mc, model, ::Nothing, greens_matrices, ::Nothing)

Computes the total energy by adding the results from `kinetic_energy_kernel` and 
`interaction_energy_kernel`.
"""
@inline Base.@propagate_inbounds function total_energy_kernel(mc, model, sites, G, flv)
    kinetic_energy_kernel(mc, model, sites, G, flv) + 
    interaction_energy_kernel(mc, model, sites, G, flv)
end
