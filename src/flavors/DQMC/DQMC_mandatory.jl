"""
    conftype(::Type{DQMC}, m::Model)

Returns the type of a configuration.
"""
conftype(::Type{DQMC}, m::Model) = error("Model has no implementation of `conftype(::Type{DQMC}, m::Model)`!")

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


"""
    propose_local(mc::DQMC, m::Model, i::Int, conf, E_boson::Float64) -> detratio, delta_E_boson, delta

Propose a local move for lattice site `i` of current configuration `conf`
with boson energy `E_boson`. Returns the Green's function determinant ratio,
the boson energy difference `delta_E_boson = delta_E_boson_new - delta_E_boson`,
and additional local move information `delta` (will be forwarded to
`accept_local!`).

See also [`accept_local!`](@ref).
"""
propose_local(mc::DQMC, m::Model, i::Int, conf, E_boson::Float64) = error("Model has no implementation of `propose_local(mc::DQMC, m::Model, i::Int, conf, E_boson::Float64)`!")

"""
    accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)

Accept a local move for site `i` at imaginary time slice `slice` of current
configuration `conf` with energy `E`. Arguments `delta`, `detratio` and
`delta_E_boson` correspond to output of `propose_local()` for that local move.

See also [`propose_local`](@ref).
"""
accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, delta, detratio, delta_E_boson) = error("Model has no implementation of `accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)`!")
