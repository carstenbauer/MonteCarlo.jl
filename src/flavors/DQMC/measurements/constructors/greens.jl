# This has lattice_iteratorator = Nothing, because it straight up copies G
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
    Measurement(
        mc, model, greens_iterator, lattice_iterator, flavor_iterator, greens_kernel, 
        obs = obs; kwargs...
    )
end

"""
    greens_kernel(mc, model, G::GreensMatrix)

Returns the unprocessed Greens function `greens(mc) = {⟨cᵢcⱼ^†⟩}`.

* Lattice Iterators: `nothing` (zero index)
* Greens Iterators: `Greens` or `GreensAt`
"""
greens_kernel(mc, model, ::Nothing, G::GreensMatrix, flv) = G.val
greens_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix, flv) = G.val[ij[1], ij[2]]