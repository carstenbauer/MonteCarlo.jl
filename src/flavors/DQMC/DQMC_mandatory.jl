"""
    rand(::Type{DQMC}, m::Model, nslices::Int)

Draw random configuration.
"""
Base.rand(::Type{DQMC}, m::Model, nslices::Int) = error("Model $(typeof(m)) doesn't implement `rand(::Type{DQMC}, m::Model, nslices::Int)`!")


"""
	hopping_matrix(mc::DQMC, m::Model)

Calculates the hopping matrix \$T_{i\\sigma, j\\sigma '}\$ where \$i, j\$ are site indices and \$\\sigma , \\sigma '\$
are flavor indices (e.g. spin indices). The hopping matrix should also contain
potential chemical potential terms on the diagonal.

A matrix element is the hopping amplitude for a hopping process: \$j,\\sigma ' \\rightarrow i,\\sigma\$.

Regarding the order of indices, if `T[i, σ, j, σ']` is your desired 4D hopping array, then `reshape(T, (n_sites * n_flavors, :))`
is the hopping matrix.
"""
hopping_matrix(mc::DQMC, m::Model) = error("Model has no implementation of `hopping_matrix(mc::DQMC, m::Model)`.")


"""
    interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.) -> nothing

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`. Potential chemical potential terms should be
part of the `hopping_matrix` and not the interaction.

This is a performance critical method and one might consider efficient in-place (in `result`) construction.
"""
interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.) = error("Model has no implementation of `interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.)`.")


"""
    propose_local(mc::DQMC, m::Model, i::Int, conf) -> detratio, ΔE_boson, Δ

Propose a local move for lattice site `i` of current configuration `conf` . Returns the
Green's function determinant ratio, the boson energy difference `ΔE_boson = E_boson_new - E_boson`,
and potentially additional local move information `Δ` (will be forwarded to `accept_local!`).

See also [`accept_local!`](@ref).
"""
propose_local(mc::DQMC, m::Model, i::Int, conf) =
    error("Model has no implementation of `propose_local(mc::DQMC, m::Model, i::Int, conf)`!")

"""
    accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, Δ, detratio, ΔE_boson)

Accept a local move for site `i` at imaginary time slice `slice` of current configuration `conf`.
Arguments `Δ`, `detratio` and `ΔE_boson` correspond to output of `propose_local` for that local move.

See also [`propose_local`](@ref).
"""
accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, Δ, detratio, ΔE_boson) =
    error("Model has no implementation of `accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, Δ, detratio, ΔE_boson)`!")
