# fields
#= A model that wants to use `Integrator` has to implement

    - `min_x::Float64[]`: minimum value(s) for input
    - `max_x::Float64[]`: maximum value(s) for input
=#

# methods

"""
    energy(mc::Integrator, m::Model, value)

Calculate energy (i.e. function value) at location `value` for Model `m`.
"""
energy(mc::Integrator, m::Model, value::Array{Float64, 1}) = error("Model has no implementation of `energy(m::Model, conf)`!")
