# abstract model definition
"""
Abstract definition of a model.
A concrete model type must have two fields:

    - `β::Float64`: temperature (depends on MC flavor if this will actually be used)
    - `l::Lattice`: any [`Lattice`](@ref)

A concrete model must implement the following methods:

    - `conftype(m::Model)`: type of a configuration
    - `energy(m::Model, conf)`: energy of configuration
    - `rand(m::Model)`: random configuration
    - `propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi`: propose local move
    - `accept_local(m::Model, i::Int, conf, E::Float64)`: accept a local move
"""
abstract type Model end

"""
    conftype(m::Model)

Returns the type of a configuration.
"""
conftype(m::Model) = error("Model has no implementation of `conftype(m::Model)`!")

"""
    energy(m::Model, conf)

Calculate energy of configuration `conf` for Model `m`.
"""
energy(m::Model, conf) = error("Model has no implementation of `energy(m::Model, conf)`!")

import Base.rand
"""
    rand(m::Model)

Draw random configuration.
"""
rand(m::Model) = error("Model has no implementation of `rand(m::Model)`!")

"""
    propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi

Propose a local move for lattice site `i` of current configuration `conf`
with energy `E`. Returns local move information `Δi` 
(e.g. `new[i] - conf[i]`, will be forwarded to `accept_local!`) and energy
difference `ΔE = E_new - E_old`.

See also [`accept_local!`](@ref).
"""
propose_local(m::Model, i::Int, conf, E::Float64) = error("Model has no implementation of `propose_local(m::Model, i::Int, conf, E::Float64)`!")

"""
    accept_local(m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64)

Accept a local move for site `i` of current configuration `conf`
with energy `E`. Arguments `Δi` and `ΔE` correspond to output of `propose_local()`
for that local move.

See also [`propose_local`](@ref).
"""
accept_local!(m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64) = error("Model has no implementation of `accept_local!(m::Model, i::Int, conf, E::Float64, Δi, ΔE::Float64)`!")

"""
    global_move(m::Model, conf, E::Float64) -> accepted::Bool

Propose a global move for configuration `conf` with energy `E`.
Returns wether the global move has been accepted or not.
"""
global_move(m::Model, conf, E::Float64) = false

"""
    prepare_observables(m::Model) -> Dict{String, Observable}

Initializes observables and returns a `Dict{String, Observable}`. In the latter,
keys are abbreviations for the observables names and values are the observables themselves.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
prepare_observables(m::Model) = Dict{String, Observable}()

"""
    measure_observables!(m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Measures observables and updates corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
measure_observables!(m::Model, obs::Dict{String,Observable}, conf, E::Float64) = nothing

"""
    measure_observables!(m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Measure observables and update corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
finish_observables!(m::Model, obs::Dict{String,Observable}) = nothing