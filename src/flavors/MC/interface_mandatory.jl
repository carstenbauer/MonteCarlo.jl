# fields
#= A model that wants to use MC must have two fields:

    - `β::Float64`: temperature
    - `l::Lattice`: any [`Lattice`](@ref)
=#

# methods
"""
    conftype(m::Model)

Returns the type of a configuration.
"""
conftype(m::Model) = error("Model has no implementation of `conftype(m::Model)`!")

"""
    energy(mc::MC, m::Model, conf)

Calculate energy of configuration `conf` for Model `m`.
"""
energy(mc::MC, m::Model, conf) = error("Model has no implementation of `energy(mc::MC, m::Model, conf)`!")

import Base.rand
"""
    rand(mc::MC, m::Model)

Draw random configuration.
"""
rand(mc::MC, m::Model) = error("Model has no implementation of `rand(mc::MC, m::Model)`!")

"""
    propose_local(mc::MC, m::Model, i::Int, conf, E::Float64) -> ΔE, Δi

Propose a local move for lattice site `i` of current configuration `conf`
with energy `E`. Returns local move information `Δi`
(e.g. `new[i] - conf[i]`, will be forwarded to `accept_local!`) and energy
difference `ΔE = E_new - E_old`.

See also [`accept_local!`](@ref).
"""
propose_local(mc::MC, m::Model, i::Int, conf, E::Float64) = error("Model has no implementation of `propose_local(mc::MC, m::Model, i::Int, conf, E::Float64)`!")

"""
    accept_local(mc::MC, m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64)

Accept a local move for site `i` of current configuration `conf`
with energy `E`. Arguments `Δi` and `ΔE` correspond to output of `propose_local()`
for that local move.

See also [`propose_local`](@ref).
"""
accept_local!(mc::MC, m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64) = error("Model has no implementation of `accept_local!(mc::MC, m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64)`!")
