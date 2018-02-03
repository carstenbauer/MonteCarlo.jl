# fields
# none

# methods

"""
    energy(mc::Integrator, m::Model, x::Vector{Float64})

Calculate energy (i.e. function value) at location `x` for Model `m`.
"""
energy(mc::Integrator, m::Model, x::Vector{Float64}) = error("Model has no implementation of `energy(mc::Integrator, m::Model, x::Vector{Float64})`!")
