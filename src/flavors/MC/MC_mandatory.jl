"""
    nsites(m::Model)

Number of lattice sites of the given model.
"""
nsites(m::Model) = error("Model has no implementation of `nsites(m::Model)`!")


"""
    rand(m::Model)

Draw random configuration.
"""
Base.rand(m::Model) = error("Model has no implementation of `rand(m::Model)`!")


"""
    propose_local(mc::MC, m::Model, i::Int, conf) -> delta_E, delta_i

Propose a local move for lattice site `i` of current configuration `conf`.
Returns local move information `delta_i` (e.g. `new[i] - conf[i]`, will be
forwarded to `accept_local!`) and energy difference `delta_E = E_new - E_old`.

See also [`accept_local!`](@ref).
"""
propose_local(mc::MC, m::Model, i::Int, conf) = error("Model has no implementation of `propose_local(mc::MC, m::Model, i::Int, conf)`!")


"""
    accept_local(mc::MC, m::Model, i::Int, conf, delta_i, delta_E::Float64)

Accept a local move for site `i` of current configuration `conf`.
Arguments `delta_i` and `delta_E` correspond to output of `propose_local()`
for that local move.

See also [`propose_local`](@ref).
"""
accept_local!(mc::MC, m::Model, i::Int, conf, delta_i, delta_E::Float64) = error("Model has no implementation of `accept_local!(m::Model, i::Int, conf, delta_i, delta_E::Float64)`!")
