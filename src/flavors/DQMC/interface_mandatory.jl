"""
    conftype(m::Model)

Returns the type of a configuration.
"""
conftype(m::Model) = error("Model has no implementation of `conftype(m::Model)`!")

import Base.rand
"""
    rand(mc::DQMC, m::Model)

Draw random configuration.
"""
rand(mc::DQMC, m::Model) = error("Model has no implementation of `rand(mc::DQMC, m::Model)`!")

"""
	hopping_matrix(mc::DQMC, m::Model)

Calculates the hopping matrix \$ T_{i\\sigma, j\\sigma '} \$ where \$ i, j \$ are site indices and \$ \\sigma , \\sigma ' \$
are flavor indices (e.g. spin indices).

A matrix element is the hopping amplitude for a hopping process: \$ j,\\sigma ' \\rightarrow i,\\sigma \$.

Regarding the order of indices, if `T[i, σ, j, σ']` is your desired 4D hopping array, then `reshape(T, (n_sites * n_flavors, :))`
is the hopping matrix.
"""
hopping_matrix(mc::DQMC, m::Model) = error("Model has no implementation of `hopping_matrix`.")

"""
	interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, slice::Int, power::Float64) -> nothing

Calculates the matrix exponential \$ exp(- power \\Delta \\tau V_{slice}) \$ and stores it into `result`.

Efficient in-place (in `result`) construction of the interaction matrix might speed up the simulation.
"""
interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, slice::Int, power::Float64=1.) = error("Model has no implementation of `interaction_matrix_exp!`.")
