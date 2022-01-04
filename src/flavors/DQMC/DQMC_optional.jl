"""
    greenseltype(::Type{DQMC}, m::Model)

Returns the type of the elements of the Green's function matrix. Defaults to 
    `ComplexF64`.
"""
greenseltype(::Type{DQMC}, m::Model) = ComplexF64

"""
    hoppingeltype(::Type{DQMC}, m::Model)

Returns the type of the elements of the hopping matrix. Defaults to `Float64`.
"""
hoppingeltype(::Type{DQMC}, m::Model) = Float64



"""
    interaction_matrix_type(T::Type{DQMC}, m::Model)

Returns the (matrix) type of the interaction matrix. Defaults to 
`Matrix{greenseltype(T, m)}`.
"""
interaction_matrix_type(T::Type{DQMC}, m::Model) = Matrix{greenseltype(T, m)}

"""
    hopping_matrix_type(T::Type{DQMC}, m::Model)

Returns the (matrix) type of the hopping matrix. Defaults to 
`Matrix{hoppingeltype(T, m)}`.
"""
hopping_matrix_type(T::Type{DQMC}, m::Model) = Matrix{hoppingeltype(T, m)}

"""
    greens_matrix_type(T::Type{DQMC}, m::Model)

Returns the (matrix) type of the greens and most work matrices. Defaults to 
`Matrix{greenseltype(T, m)}`.
"""
greens_matrix_type(T::Type{DQMC}, m::Model) = Matrix{greenseltype(T, m)}

"""
    interaction_matrix_exp!(
        mc::DQMC, m::Model, field::AbstractField, 
        result::AbstractArray, slice::Int, power::Float64 = 1.0
    )

Calculate the, exponentiated interaction matrix 
`exp(- power * delta_tau * V(slice))` and store it in `result::AbstractArray`. 

This only includes terms with 4 operators, i.e. not the chemical potential or 
any hopping. By default the calculation will be performed by the appropriate 
field type (i.e. by `interaction_matrix_exp!(field, result, slice, power)`)

This is a performance critical method and one might consider efficient in-place 
(in `result`) construction.
"""
@inline function interaction_matrix_exp!(
        mc::DQMC, model::Model, field::AbstractField,
        result::AbstractArray, slice::Int, power::Float64 = +1.0
    )
    interaction_matrix_exp!(field, result, slice, power)
end


"""
    propose_local(mc::DQMC, m::Model, field::AbstractField, i::Int, slice::Int)

Propose a local move for lattice site `i` at time slice `slice` for a `field` 
holding the current configuration. Returns the Green's function determinant 
ratio, the boson energy difference `ΔE_boson = E_boson_new - E_boson`,
and any extra information `passthrough` that might be useful in `accept_local`.

By default this function will call 
`propose_local(mc, field, mc.stack.greens, i, slice)`.

See also [`accept_local!`](@ref).
"""
@inline function propose_local(
        mc::DQMC, model::Model, field::AbstractField, i::Int, slice::Int
    )
    propose_local(mc, field, mc.stack.greens, i, slice)
end

"""
    accept_local!(
        mc::DQMC, m::Model, field::AbstractField, i::Int, slice::Int, 
        detratio, ΔE_boson, passthrough
    )

Accept a local move for site `i` at imaginary time slice `slice` of current 
configuration in `field`. Arguments `detratio`, `ΔE_boson` and `passthrough` 
correspond to output of `propose_local` for that local move.

By default this function will call
`accept_local!(mc, field, mc.stack.greens, i, slice, detration, ΔE_boson, passthrough)`

See also [`propose_local`](@ref).
"""
@inline function accept_local!(
        mc::DQMC, model::Model, field::AbstractField, i::Int, slice::Int, 
        detratio, ΔE_boson, passthrough
    )
    accept_local!(mc, field, mc.stack.greens, i, slice, detratio, ΔE_boson, passthrough)
end

"""
    energy_boson(mc::DQMC, m::Model, conf)

Calculate bosonic part (non-Green's function determinant part) of energy for 
configuration `conf` for Model `m`.

This is required for global and parallel updates as well as boson energy 
measurements, but not for local updates.
"""
energy_boson(mc::DQMC, m::Model, conf) = energy_boson(field(mc), conf)