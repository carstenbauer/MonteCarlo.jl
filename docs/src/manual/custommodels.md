# Custom models

Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, a key idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to implement your own physical model. Sometimes examples say more than words, so feel encouraged to have a look at the implementations of the above mentioned models.

## Mandatory fields and methods

Any concrete model type, let's call it `MyModel` in the following, must be a subtype of the abstract type `MonteCarlo.Model`. Internally it must have at least the following fields:

 * `β::Float64`: temperature (depends on MC flavor if this will actually be used)
 * `l::Lattice`: any [`MonteCarlo.Lattice`](@ref)

Furthermore it must implement the following methods:

 * `conftype(m::Model)`: type of a configuration
 * `energy(m::Model, conf)`: energy of configuration
 * `rand(m::Model)`: random configuration
 * `propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi`: propose local move
 * `accept_local(m::Model, i::Int, conf, E::Float64)`: accept a local move