# fields
# none

# methods

"""
    energy(mc::Integrator, m::Model, value)

Calculate energy (i.e. function value) at location `value` for Model `m`.
"""
energy(mc::Integrator, m::Model, value::Vector{Float64}) = error("Model has no implementation of `energy(m::Model, conf)`!")
