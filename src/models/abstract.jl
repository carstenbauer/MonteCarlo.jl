# abstract model definition
"""
Abstract model.
"""
abstract type Model end


# Model interface: mandatory

"""
    lattice(model)

Returns the lattice of a given model.
"""
lattice(::T) where {T <: Model} = error("lattice() not implemented for $T")


# Optional

"""
    parameters(model)

Collects relevant parametrs of a model into a named tuple.
"""
@inline parameters(::Model) = NamedTuple{}()

# See configurations.jl - compression of configurations
# compress, decompress

# copy constructor
function Model(model::T; kwargs...) where {T <: Model}
    args = (get(kwargs, field, getfield(model, field)) for field in fieldnames(T))
    return T(args...)
end