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
    init_interaction_matrix(m::Model)

Initializes the interaction matrix.
"""
function init_interaction_matrix(m::Model)
    N = length(lattice(m))
    flv = nflavors(m)
    zeros(greenseltype(DQMC, m), N*flv, N*flv)
end


"""
    energy_boson(mc::DQMC, m::Model, conf)

Calculate bosonic part (non-Green's function determinant part) of energy for 
configuration `conf` for Model `m`.

This is required for global and parallel updates as well as boson energy 
measurements.
"""
energy_boson(mc::DQMC, m::Model, conf) = throw(MethodError(energy_boson, (mc, m, conf)))

# Should collect the parameters of the model and lattice in a named tuple
parameters(m::Model) = NamedTuple()