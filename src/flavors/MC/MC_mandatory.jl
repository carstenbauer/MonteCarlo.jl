# fields
#= A model that wants to use MC must have fields:

    - `l::Lattice`: any [`Lattice`](@ref)
=#

# methods
"""
    conftype(::Type{MC}, m::Model)

Returns the type of a configuration.
"""
conftype(::Type{MC}, m::Model) = error("Model has no implementation of `conftype(::Type{MC}, m::Model)`!")

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
    propose_local(mc::MC, m::Model, i::Int, conf, E::Float64) -> delta_E, delta_i

Propose a local move for lattice site `i` of current configuration `conf`
with energy `E`. Returns local move information `delta_i`
(e.g. `new[i] - conf[i]`, will be forwarded to `accept_local!`) and energy
difference `delta_E = E_new - E_old`.

See also [`accept_local!`](@ref).
"""
propose_local(mc::MC, m::Model, i::Int, conf, E::Float64) = error("Model has no implementation of `propose_local(mc::MC, m::Model, i::Int, conf, E::Float64)`!")

"""
    accept_local(mc::MC, m::Model, i::Int, conf, E::Float64, delta_i, delta_E::Float64)

Accept a local move for site `i` of current configuration `conf`
with energy `E`. Arguments `delta_i` and `delta_E` correspond to output of `propose_local()`
for that local move.

See also [`propose_local`](@ref).
"""
accept_local!(mc::MC, m::Model, i::Int, conf, E::Float64, delta_i, delta_E::Float64) = error("Model has no implementation of `accept_local!(m::Model, i::Int, conf, E::Float64, delta_i, delta_E::Float64)`!")
