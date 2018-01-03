# Custom models

Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, a key idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to implement your own physical model. Sometimes examples say more than words, so feel encouraged to have a look at the implementations of the above mentioned models.

## Mandatory fields and methods

Any concrete model type, let's call it `MyModel` in the following, must be a subtype of the abstract type `MonteCarlo.Model`. Internally it must have at least the following fields:

 * `β::Float64`: inverse temperature
 * `l::Lattice`: any [`Lattice`](@ref MonteCarlo.Lattice)

Furthermore it **must** implement the following methods:

 * [`conftype`](@ref MonteCarlo.conftype): type of a configuration
 * [`energy`](@ref MonteCarlo.energy): energy of configuration
 * [`rand`](@ref MonteCarlo.rand): random configuration
 * [`propose_local`](@ref MonteCarlo.propose_local): propose local move
 * [`accept_local`](@ref MonteCarlo.accept_local): accept a local move

 A full list of methods that should be implemented for `MyModel` can be found here: [Methods: Models](@ref).