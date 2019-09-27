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

"""
    prepare_observables(mc::DQMC, m::Model) -> Dict{String, Observable}

Initializes observables and returns a `Dict{String, Observable}`. In the latter,
keys are abbreviations for the observables names and values are the observables themselves.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
prepare_observables(mc::DQMC, m::Model) = Dict{String, Observable}()

"""
    measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf)

Measures observables and updates corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf) = nothing

"""
    finish_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable})

Finish measurements of observables.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
finish_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}) = nothing
