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

# abstract model definition
"""
Abstract definition of a model.
A concrete model type must have two fields:

    - `β::Float64`: temperature (depends on MC flavor if this will actually be used)
    - `l::Lattice`: any [Lattice](@ref)

A concrete model must implement the following methods:

    - `conftype(m::Model)`: type of a configuration
    - `energy(m::Model, conf)`: energy of configuration
    - `rand(m::Model)`: random configuration
    - `propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi`: propose local move
    - `accept_local(m::Model, i::Int, conf, E::Float64)`: accept a local move
"""
abstract type Model end

"""
    conftype(m::Model)

Returns the type of a configuration.
"""
conftype(m::Model) = error("Model has no implementation of `conftype(m::Model)`!")

"""
    energy(m::Model, conf)

Calculate energy of configuration `conf` for Model `m`.
"""
energy(m::Model, conf) = error("Model has no implementation of `energy(m::Model, conf)`!")

import Base.rand
"""
    rand(m::Model)

Draw random configuration.
"""
rand(m::Model) = error("Model has no implementation of `rand(m::Model)`!")

"""
    propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi

Propose a local move for element `i` of current configuration `conf`
with energy `E`. Returns the local move `Δi = new[i] - conf[i]` and energy difference `ΔE = E_new - E_old`.
"""
propose_local(m::Model, i::Int, conf, E::Float64) = error("Model has no implementation of `propose_local(m::Model, i::Int, conf, E::Float64)`!")

"""
    accept_local(m::Model, i::Int, conf, E::Float64)

Accept a local move for site `i` of current configuration `conf`
with energy `E`. Arguments `Δi` and `ΔE` correspond to output of `propose_local()`
for that local move.
"""
accept_local!(m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64) = error("Model has no implementation of `accept_local!(m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64)`!")


# abstract model parameters definition
# """
# Abstract definition of model parameters.
# Necessary fields depend on Monte Carlo flavor.
# However, any concrete model parameters type should (most likely) have the following fields:
#     - β: temperature of the system
# """


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
abstract type CubicLattice <: Lattice end
