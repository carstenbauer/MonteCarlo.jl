################################################################################
# Interface
# A Measurement must inherit from AbstractMeasurement and implement the
# following functions.

abstract type AbstractMeasurement end


"""
    prepare!(measurement, mc, model)

This method is called between the thermalization and the measurement phase.
"""
function prepare!(m::AbstractMeasurement, mc, model)
    throw(MethodError(prepare!, (m, mc, model)))
end
"""
    measure!(measurement, mc, model, sweep_index)

Performs a measurement during the measurement phase.
"""
function measure!(m::AbstractMeasurement, mc, model, i)
    throw(MethodError(measure!, (m, mc, model)))
end
"""
    finish!(measurement, mc, model)

Finish a measurement. This method is called after the measurement phase.
"""
function finish!(m::AbstractMeasurement, mc, model)
    throw(MethodError(finish!, (m, mc, model)))
end

################################################################################
# A new model may implement the following for convenience

"""
    default_measurements(mc, model)

Return a dictionary of default measurements for a given Monte Carlo flavour and
model. If there is no implementation given for the specific Monte Carlo flavour
an empty dictionary will be returned.
"""
default_measurements(mc, model) = Dict{Symbol, AbstractMeasurement}()

################################################################################
# mc based, default measurements

"""
    ConfigurationMeasurement(mc, model, rate=1)

Measures configurations of the given Monte Carlo flavour and model. The rate of
measurements can be reduced with `rate`. (e.g. `rate=10` means 1 measurement per
10 sweeps)
"""
struct ConfigurationMeasurement <: AbstractMeasurement
    obs::Observable
    rate::Int64
    ConfigurationMeasurement(mc, model, rate=1) = new(
        Observable(typeof(mc.conf), "Configurations"), rate
    )
end
prepare!(::ConfigurationMeasurement, mc, model) = nothing
function measure!(m::ConfigurationMeasurement, mc, model, i::Int64)
    (i % m.rate == 0) && push!(m.obs, conf(mc))
    nothing
end
finish!(::ConfigurationMeasurement, mc, model) = nothing

################################################################################
# called by simulation

for function_name in (:prepare!, :finish!)
    @eval begin
        function $(function_name)(
                measurements::Dict{Symbol, AbstractMeasurement},
                mc, model
            )
            for (k, m) in measurements
                $(function_name)(m, mc, model)
            end
            nothing
        end
    end
end

function measure!(
        measurements::Dict{Symbol, AbstractMeasurement},
        mc, model, sweep_index::Int64
    )
    for (k, m) in measurements
        measure!(m, mc, model, sweep_index)
    end
    nothing
end

################################################################################
# other convenience functions

# printing
function Base.show(io::IO, m::AbstractMeasurement)
    #  no parametrization -v    v- no MonteCarlo.
    typename = typeof(m).name.name
    fnames = fieldnames(typeof(m))
    observables = [s for s in fnames if getfield(m, s) isa AbstractObservable]
    other = [s for s in fnames if !(getfield(m, s) isa AbstractObservable)]
    println(io, typename)
    for obs_fieldname in observables
        o = getfield(m, obs_fieldname)
        oname = MonteCarloObservable.name(o)
        otypename = typeof(o).name.name
        println(io, "\t", obs_fieldname, "::", otypename, "\t → \"", oname, "\"")
    end
    for fieldname in other
        println(io, "\t", fieldname, "::", typeof(getfield(m, fieldname)))
    end
    nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::AbstractMeasurement)
    #small
    #  no parametrization -v    v- no MonteCarlo.
    typename = typeof(m).name.name
    fnames = fieldnames(typeof(m))
    temp = [s for s in fnames if getfield(m, s) isa AbstractObservable]
    observable_names = map(temp) do obs_fieldname
        MonteCarloObservable.name(getfield(m, obs_fieldname))
    end
    print(io, typename, " (\"", join(observable_names, "\", \""), "\")")
    nothing
end


"""
    measurements(mc)

Returns a nested dictionary of all measurements used in a given Monte Carlo
simulation `mc`. The thermalization stage is accessed by `:TH`, the measurement
stage by `:ME`.
"""
function measurements(mc::MonteCarloFlavor)
    return Dict(
        :TH => mc.thermalization_measurements,
        :ME => mc.measurements
    )
end


"""
    observables(mc)

Returns a nested dictionary of all observables used in a given Monte Carlo
simulation `mc`. The result `obs` is indexed as `obs[stage][measurement][name]`,
where `stage` is `:TH` (thermalization stage) or `:ME` (measurement stage),
`measurement::Symbol` is the name of the measurement and `name::String` is the
name of the observable.
"""
function observables(mc::MonteCarloFlavor)
    th_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
        k => let
            fns = fieldnames(typeof(mc))
            os = [getfield(mc, fn) for fn in fns if getfield(mc, fn) isa AbstractObservable]
            Dict{String, AbstractObservable}(MonteCarloObservable.name(o) => o for o in os)
        end for (k, m) in mc.thermalization_measurements
    )
    me_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
        k => let
            fns = fieldnames(typeof(m))
            os = [getfield(m, fn) for fn in fns if getfield(m, fn) isa AbstractObservable]
            Dict{String, AbstractObservable}(MonteCarloObservable.name(o) => o for o in os)
        end for (k, m) in mc.measurements
    )

    return Dict(:TH => th_obs, :ME => me_obs)
end



"""
    push!(mc, tag::Symbol, MT::Type{<:AbstractMeasurement}[, stage=:ME])

Adds a new pair `tag => MT(mc, model)`, where `MT` is a type
`<: AbstractMeasurement`, to either the thermalization or measurement `stage`
(`:TH` or `:ME`) of the simulation `mc`.

See also: [`unsafe_push!`](@ref)
"""
function Base.push!(mc::MonteCarloFlavor, tag::Symbol, MT::Type{<:AbstractMeasurement}, stage=:ME)
    unsafe_push!(mc, tag, MT(mc, mc.model), stage)
end

"""
    unsafe_push!(mc, tag::Symbol, m::AbstractMeasurement[, stage=:ME])

Adds a pair `tag => m` to either the thermalization or measurement stage (`:TH`
or `:ME`) of the given simulation `mc`.

Note that this function is unsafe as it does not test whether `m` is a valid
measurement for the given simulation.

See also: [`MonteCarlo.push!`](@ref)
"""
function unsafe_push!(mc::MonteCarloFlavor, tag::Symbol, m::AbstractMeasurement, stage=:ME)
    if stage in (:ME, :me, :Measurement, :measurement)
        push!(mc.measurements, tag => m)
    elseif stage in (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
        push!(mc.thermalization_measurements, tag => m)
    else
        throw(ErrorException("`stage = $stage` is not valid."))
    end
end


"""
    delete!(mc, key[, stage=:MC])
    delete!(mc, MT::Type{<:AbstractMeasurement}[, stage=:MC])

Deletes a measurement from the given Monte Carlo simulation by key or by type.
When deleting by type, multiple measurements can be targeted using inheritance.
For example, `delete!(mc, IsingMeasurement)` wil delete all
`IsingEnergyMeasurement` and `IsingMagnmetizationMeasurement` objects.
"""
function Base.delete!(mc::MonteCarloFlavor, key::Symbol, stage=:ME)
    if stage in (:ME, :me, :Measurement, :measurement)
        delete!(mc.measurements, key)
    elseif stage in (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
        delete!(mc.thermalization_measurements, key)
    else
        throw(ErrorException("`stage = $stage` is not valid."))
    end
end

function Base.delete!(mc::MonteCarloFlavor, MT::Type{<: AbstractMeasurement}, stage=:ME)
    if stage in (:ME, :me, :Measurement, :measurement)
        ks = collect(keys(mc.measurements))
        for k in ks
            if typeof(mc.measurements[k]) <: MT
                delete!(mc.measurements, k)
            end
        end
        mc.measurements
    elseif stage in (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
        ks = collect(keys(mc.thermalization_measurements))
        for k in ks
            if typeof(mc.thermalization_measurements[k]) <: MT
                delete!(mc.thermalization_measurements, k)
            end
        end
        mc.thermalization_measurements
    else
        throw(ErrorException("`stage = $stage` is not valid."))
    end
end
