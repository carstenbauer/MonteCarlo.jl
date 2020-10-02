"""
    rand(::Type{DQMC}, m::Model, nslices::Int)

Draw random configuration.
"""
Base.rand(::Type{DQMC}, m::Model, nslices::Int) = MethodError(rand, (DQMC, m, nslices))


"""
    nflavors(model)

Returns the number of activer fermion flavors of a given Quantum Monte Carlo
model.

The size of the hopping matrix, the interaction matrix, the greens matrix and
any derived quantity should have the 
`(length(lattice(model)) * nflavors(model), length(lattice(model)) * nflavors(model)`.
"""
nflavors(::T) where {T <: Model} = error("nflavors() not implemented for $T")


"""
    hopping_matrix(mc::DQMC, m::Model)

Calculates the hopping matrix \$T_{i\\sigma, j\\sigma '}\$ where \$i, j\$ are site indices and \$\\sigma , \\sigma '\$
are flavor indices (e.g. spin indices). The hopping matrix should also contain
potential chemical potential terms on the diagonal.

A matrix element is the hopping amplitude for a hopping process: \$j,\\sigma ' \\rightarrow i,\\sigma\$.

Regarding the order of indices, if `T[i, σ, j, σ']` is your desired 4D hopping array, then `reshape(T, (n_sites * n_flavors, :))`
is the hopping matrix.
"""
hopping_matrix(mc::DQMC, m::Model) = MethodError(hopping_matrix, (mc, m))


"""
    interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.) -> nothing

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`. Potential chemical potential terms should be
part of the `hopping_matrix` and not the interaction.

This is a performance critical method and one might consider efficient in-place (in `result`) construction.
"""
interaction_matrix_exp!(mc::DQMC, m::Model, result::Matrix, conf, slice::Int, power::Float64=1.) = 
    MethodError(interaction_matrix_exp!, (mc, m, result, conf, slice, power))


"""
    propose_local(mc::DQMC, m::Model, i::Int, conf) -> detratio, ΔE_boson, passthrough

Propose a local move for lattice site `i` of current configuration `conf` . Returns the
Green's function determinant ratio, the boson energy difference `ΔE_boson = E_boson_new - E_boson`,
and potentially additional local move information `passthrough` (will be forwarded to `accept_local!`).

See also [`accept_local!`](@ref).
"""
propose_local(mc::DQMC, m::Model, i::Int, conf) =
    MethodError(propose_local, (mc, m, i, conf))

"""
    accept_local(mc::DQMC, m::Model, i::Int, slice::Int, conf, detratio, ΔE_boson, passthrough)

Accept a local move for site `i` at imaginary time slice `slice` of current configuration `conf`.
Arguments `detratio`, `ΔE_boson` and `passthrough` correspond to output of `propose_local` for that 
local move.

See also [`propose_local`](@ref).
"""
accept_local!(mc::DQMC, m::Model, i::Int, slice::Int, conf, detratio, ΔE_boson, passthrough) =
    MethodError(accept_local!, (mc, m, i, slice, conf, detratio, ΔE_boson, passthrough))
