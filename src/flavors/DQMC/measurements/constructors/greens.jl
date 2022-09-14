"""
    greens_measurement(mc, model; kwargs...)

Constructs a measurement of the greens function.

## Optional Keyword Arguments

- `kernel = greens_kernel` sets the function representing the Wicks 
expanded expectation value of the measurement. In this case the kernel just 
returns the given greens function.
- `lattice_iterator = nothing` controls which sites are passed to the kernel 
and how they are summed. With `nothing` this is left to the kernel.
- `flavor_iterator = nothing` controls which flavor indices 
(spins) are passed to the kernel. With `lattice_iterator = nothing` this is is 
also left to the kernel.
- kwargs from `DQMCMeasurement`
"""
function greens_measurement(
        mc::DQMC, model::Model, greens_iterator = Greens(); 
        lattice_iterator = nothing,
        flavor_iterator = nothing,
        capacity = _default_capacity(mc), eltype = geltype(mc),
        obs = let
            N = length(lattice(model)) * unique_flavors(mc)
            LogBinner(zeros(eltype, (N, N)), capacity=capacity)
        end, kwargs...
    )
    DQMCMeasurement(
        mc, model, greens_iterator, lattice_iterator, flavor_iterator, greens_kernel, 
        obs = obs; kwargs...
    )
end

"""
    greens_kernel(mc, model, sites_indices, greens_matrix, flavor_indices)

Returns the unprocessed Greens function `greens(mc) = {⟨cᵢcⱼ^†⟩}`.
"""
greens_kernel(mc, model, ::Nothing, G::GreensMatrix, flv) = G.val
greens_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv) = G.val[ij[1], ij[2]]