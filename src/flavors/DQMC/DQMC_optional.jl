"""
    greenseltype(::Type{DQMC}, m::Model)

Returns the type of the elements of the Green's function matrix. Defaults to `ComplexF64`.
"""
greenseltype(::Type{DQMC}, m::Model) = ComplexF64

"""
    hoppingeltype(::Type{DQMC}, m::Model)

Returns the type of the elements of the hopping matrix. Defaults to `Float64`.
"""
hoppingeltype(::Type{DQMC}, m::Model) = Float64

"""
    energy(mc::DQMC, m::Model, conf)

Calculate bosonic part (non-Green's function determinant part) of energy for configuration `conf` for Model `m`.
"""
energy_boson(mc::DQMC, m::Model, conf) = 0.
