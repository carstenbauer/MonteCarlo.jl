"""
    greenstype(m::Model)

Returns the type of the Green's function matrix. Defaults to `Matrix{Complex128}`.
"""
greenstype(m::Model) = Matrix{Complex128}

"""
    energy(mc::DQMC, m::Model, conf)

Calculate non-Green's function determinant (bosonic) part of energy for configuration `conf` for Model `m`.
"""
energy(mc::DQMC, m::Model, conf) = 0.

"""
    prepare_observables(mc::DQMC, m::Model) -> Dict{String, Observable}

Initializes observables and returns a `Dict{String, Observable}`. In the latter,
keys are abbreviations for the observables names and values are the observables themselves.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
prepare_observables(mc::DQMC, m::Model) = Dict{String, Observable}()

"""
    measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Measures observables and updates corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf, E::Float64) = nothing

"""
    measure_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Measure observables and update corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
finish_observables!(mc::DQMC, m::Model, obs::Dict{String,Observable}) = nothing
