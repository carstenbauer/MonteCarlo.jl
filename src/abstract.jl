# abstract monte carlo flavor definition
"""
Abstract definition of a Monte Carlo flavor.

A concrete monte carlo flavor must implement the following methods:

    - `init!(mc)`: initialize the simulation without overriding parameters (will also automatically be available as `reset!`)
    - `run!(mc)`: run the simulation
"""
abstract type MonteCarloFlavor end

"""
    reset!(mc::MonteCarloFlavor)

Resets the Monte Carlo simulation `mc`.
Previously set parameters will be retained.
"""
reset!(mc::MonteCarloFlavor) = init!(mc) # convenience mapping


# abstract lattice definition
"""
Abstract definition of a lattice.
Necessary fields depend on Monte Carlo flavor.
However, any concrete Lattice type should have at least the following fields:

    - `sites`: number of lattice sites
    - `neighs::Matrix{Int}`: neighbor matrix (row = neighbors, col = siteidx)
"""
abstract type Lattice end

"""
Abstract cubic lattice.

- 1D -> Chain
- 2D -> SquareLattice
- ND -> NCubeLattice
"""
abstract type AbstractCubicLattice <: Lattice end


# abstract model definition
"""
Abstract model.
"""
abstract type Model end


# general functions
"""
	observables(mc::MonteCarloFlavor)

Get a list of all observables defined for a given Monte Carlo simulation.

Returns a `Dict{String, String}` where values are the observables names and
keys are short versions of those names. The keys can be used to
collect correponding observable objects from the Monte Carlo simulation, e.g. like `mc.obs[key]`.

Note, there is no need to implement this function for a custom `MonteCarloFlavor`.
"""
function observables(mc::MonteCarloFlavor)
	obs = Dict{String, String}()
	obsobjects = prepare_observables(mc, mc.model)
	for (s, o) in obsobjects
		obs[s] = name(o)
	end
	return obs
end