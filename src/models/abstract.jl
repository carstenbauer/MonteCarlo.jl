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