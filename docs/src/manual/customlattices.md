# Custom lattices

As described in [Custom models](@ref) a lattice is considered to be part of a model. Hence, most of the requirements for fields of a `Lattice` subtype come from potential models (see [Lattice requirements](@ref)). Below you'll find information on which fields are mandatory from a Monte Carlo flavor point of view.

## Mandatory fields and methods

Any concrete lattice type, let's call it `MyLattice` in the following, must be a subtype of the abstract type `MonteCarlo.Lattice`. To work with a Monte Carlo flavor, it **must** internally have at least have the following field,

 * `sites`: number of lattice sites.

However, as already mentioned above depending on the physical model of interest it will typically also have (at least) something like

 * `neighs`: next nearest neighbors,

 as most Hamiltonian will need next nearest neighbor information.

The only reason why such a field isn't generally mandatory is that the Monte Carlo routine doesn't care about the lattice much. Neighbor information is usually only used in the energy (difference) calculation of a particular configuration like done in [`energy`](@ref MonteCarlo.energy) or [`propose_local`](@ref MonteCarlo.propose_local) which both belong to a `Model`.