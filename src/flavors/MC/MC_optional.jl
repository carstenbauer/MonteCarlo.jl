"""
    global_move(mc::MC, m::Model, conf) -> accepted::Bool

Propose a global move for configuration `conf`.
Returns wether the global move has been accepted or not.
"""
global_move(mc::MC, m::Model, conf) = false

"""
    prepare_observables(mc::MC, m::Model) -> Dict{String, Observable}

Initializes observables and returns a `Dict{String, Observable}`.
In the latter, keys are abbreviations for the observables names and values
are the observables themselves.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
prepare_observables(mc::MC, m::Model) = Dict{String, Observable}()

"""
    measure_observables!(mc::MC, m::Model, obs::Dict{String,Observable}, conf)

Measures observables and updates corresponding `MonteCarloObservable.Observable`
objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
measure_observables!(mc::MC, m::Model, obs::Dict{String,Observable}, conf) = nothing

"""
    finish_observables!(mc::MC, m::Model, obs::Dict{String,Observable})

Finish measurements of observables.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
finish_observables!(mc::MC, m::Model, obs::Dict{String,Observable}) = nothing
