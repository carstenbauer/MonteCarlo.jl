# abstract model definition
"""
Abstract model.
"""
abstract type Model end


# Model interface: mandatory
"""
    nsites(m::Model)

Number of lattice sites of the given model.
"""
nsites(m::Model) = error("Model $(typeof(m)) doesn't implement `nsites`!")