import Base.rand
"""
    rand(mc::Integrator, m::Model)

Draw random value in the domain.
"""
function rand(mc::Integrator, m::Model)
    value = copy(mc.p.min_x)

    for i in 1:size(mc.p.min_x, 1)
        value[i] = rand() * (mc.p.max_x[i] - mc.p.min_x[i]) + mc.p.min_x[i]
    end

    return value
end

"""
    propose(mc::Integrator, m::Model, value::Vector{Float64}, energy::Float64) -> proposed_value, r

Propose a local move from point `value` to `proposed_value`.
Returns the `proposed_value` and the ratio of weights `r`.
"""
@inline function propose(mc::Integrator, m::Model, value::Vector{Float64}, E::Float64)
    proposed_shift = [(mc.p.max_x[i] - mc.p.min_x[i]) * 0.2 * (rand() - 0.5) for i in size(mc.p.min_x, 1)]
    proposed_value = max.(min.(value + proposed_shift, mc.p.max_x), mc.p.min_x)
    r = energy(mc, m, proposed_value) / E

    return proposed_value, r
end


"""
    prepare_observables(m::Model) -> Dict{String, Observable}

Initializes observables and returns a `Dict{String, Observable}`. In the latter,
keys are abbreviations for the observables names and values are the observables themselves.

See also [`measure_observables!`](@ref) and [`finish_observables!`](@ref).
"""
@inline function prepare_observables(mc::Integrator, m::Model)
    obs = Dict{String,Observable}()
    obs["energy"] = Observable(Float64, "Energies")
    obs["integral"] = Observable(Float64, "Integral")
    return obs
end

"""
    measure_observables!(mc::Integrator, m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Measures observables and updates corresponding `MonteCarloObservable.Observable` objects in `obs`.

See also [`prepare_observables`](@ref) and [`finish_observables!`](@ref).
"""
@inline function measure_observables!(mc::Integrator, m::Model, obs::Dict{String,Observable}, E::Float64)
    add!(obs["energy"], E)
    nothing
end

"""
    finish_observables!(mc::Integrator, m::Model, obs::Dict{String,Observable}, conf, E::Float64)

Final processing of observable objects in `obs`.

See also [`prepare_observables`](@ref) and [`measure_observables!`](@ref).
"""

@inline function finish_observables!(mc::Integrator, m::Model, obs::Dict{String,Observable})
    volume = prod(abs.(mc.p.max_x - mc.p.min_x))
    add!(obs["integral"], volume * mean(obs["energy"]))
    println("Integral is $(volume * mean(obs["energy"]))")
    nothing
end
