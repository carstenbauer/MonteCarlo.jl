"""
    prepare_observables(m::HubbardModel)

Initializes observables for the Hubbard model and returns a `Dict{String, Observable}`.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
@inline function prepare_observables(m::HubbardModel)
    obs = Dict{String,Observable}()
    obs["confs"] = Observable(HubbardConf, "Configurations")

    return obs
end

"""
    measure_observables!(m::HubbardModel, obs::Dict{String,Observable}, conf::HubbardConf, E::Float64)

Measures observables and updates corresponding `Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
@inline function measure_observables!(m::HubbardModel, obs::Dict{String,Observable}, conf::HubbardConf, E::Float64)
    add!(obs["confs"], conf)
    nothing
end

"""
    measure_observables!(m::HubbardModel, obs::Dict{String,Observable}, conf::HubbardConf, E::Float64)

Calculates magnetic susceptibility and specific heat and updates corresponding `Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""
@inline function finish_observables!(m::HubbardModel, obs::Dict{String,Observable})
    nothing
end