# abstract monte carlo flavor definition
"""
Abstract definition of a Monte Carlo flavor.
"""
abstract type MonteCarloFlavor end

# MonteCarloFlavor interface: mandatory
"""
    init!(mc)

Initialize the Monte Carlo simulation. Has an alias function `reset!`.
"""
init!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `init!`!")

"""
    run!(mc)

Run the Monte Carlo Simulation.
"""
run!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `run!`!")






# abstract lattice definition
"""
Abstract definition of a lattice.
"""
abstract type Lattice end


# Lattice interface: mandatory
# TODO: This needs to be updated. There must be a general way to access sites and bonds.
"""
    nsites(l::Lattice)

Number of lattice sites.
"""
nsites(l::Lattice) = error("Lattice $(typeof(l)) doesn't implement `nsites`.")

# Typically, you also want to implement

#     - `neighbors_lookup_table(lattice)`: return a neighbors matrix where
#                                         row = neighbors and col = siteidx.


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


# Model interface: mandatory
"""
    nsites(m::Model)

Number of lattice sites of the given model.
"""
nsites(m::Model) = error("Model $(typeof(m)) doesn't implement `nsites`!")






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
		obs[s] = MonteCarloObservable.name(o)
	end
	return obs
end


"""
    reset!(mc::MonteCarloFlavor)

Resets the Monte Carlo simulation `mc`.
Previously set parameters will be retained.
"""
reset!(mc::MonteCarloFlavor) = init!(mc) # convenience mapping