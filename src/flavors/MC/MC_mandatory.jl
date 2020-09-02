"""
    rand(::Type{MC}, m::Model)

Draw random configuration.
"""
Base.rand(::Type{MC}, m::Model) = MethodError(rand, (MC, m))

"""
    propose_local(mc::MC, m::Model, i::Int, conf) -> delta_E, passthrough

Propose a local move for lattice site `i` of current configuration `conf`.
Returns local energy difference `delta_E = E_new - E_old` alongside some 
additional information `passthrough` which is passed to `accept_local!`.

See also [`accept_local!`](@ref).
"""
propose_local(mc::MC, m::Model, i::Int, conf) = MethodError(propose_local, (mc, m, i, conf))


"""
    accept_local(mc::MC, m::Model, i::Int, conf, delta_E::Float64, passthrough)

Accept a local move for site `i` of current configuration `conf`.
Arguments `delta_E` and `passthrough` correspond to output of `propose_local()`
for that local move.

See also [`propose_local`](@ref).
"""
accept_local!(mc::MC, m::Model, i::Int, conf, delta_E::Float64, passthrough) = 
    MethodError(accept_local!, (mc, m, i, conf, delta_E, passthrough))
