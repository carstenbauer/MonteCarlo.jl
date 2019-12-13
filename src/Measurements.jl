################################################################################
# Interface
# A Measurement must inherit from AbstractMeasurement and implement the
# following functions.

abstract type AbstractMeasurement end


###################################
# You must implement
###################################


"""
    measure!(measurement, mc, model, sweep_index)

Performs a measurement during the measurement phase.
"""
function measure!(m::AbstractMeasurement, mc, model, i)
    throw(MethodError(measure!, (m, mc, model)))
end


###################################
# You may implement
###################################


"""
    prepare!(measurement, mc, model)

This method is called between the thermalization and the measurement phase.
"""
prepare!(m::AbstractMeasurement, mc, model) = nothing


"""
    finish!(measurement, mc, model)

Finish a measurement. This method is called after the measurement phase.
"""
finish!(m::AbstractMeasurement, mc, model) = nothing


"""
    observables(measurement)

Returns a dictionary of observables.

The default implementation searches for all fields `<: AbstractObservable` and
builds a dictionary pairs `name(obs) => obs`.
"""
function observables(m::AbstractMeasurement)
    os = [getfield(m, n) for n in obs_fieldnames_from_obj(m)]
    Dict{String, AbstractObservable}(MonteCarloObservable.name(o) => o for o in os)
end


"""
    save!(measurement, filename, entryname)

Saves a measurement to a jld-file `filename` in group `entryname`.

The default implementation uses `observables(measurement)` to find observables
and saves them using
`MonteCarloObservable.saveobs(obs, filename, entryname * "/" * obs.group)`.
The `entryname` passed to this function is equivalent to the structure given by
`measurements()`, e.g. `ME/config/` if `measurements(mc)[:ME][:config]` carries
a measurement.

See also [`save_measurements!`](@ref), [`observables`](@ref)
"""
function save!(m::AbstractMeasurement, filename::String, entryname::String)
    for (k, obs) in observables(m)
        saveobs(obs, filename, entryname * "/" * obs.group)
    end
    nothing
end


# Statistics forwarded from MonteCarloObservable/BinningAnalysis
# Generates functions
#   mean(measurement)       - returns the mean/expectation value of a measurement
#   var(measurement)        - returns the variance of a measurement
#   std_error(measurement)  - returns the standard error of the mean of a measurement
#   tau(measurement)        - return the autocorrelation time of a measurement
for (func, name) in zip(
        (:mean, :var, :std_error, :tau),
        ("mean", "variance", "standard error", "autocorrelation time")
    )
    docstring = """
        $func(measurement)

    Returns the $name of a given `measurement`.

    The default implementation searches for all fields `<: AbstractObservable`
    and returns `$func(x)` for each field `x`. If there are multiple fields
    with type `<: AbstractObservable` a dictionary will be return.
    """
    @eval begin
        @doc $docstring $func
        function MonteCarloObservable.$(func)(m::AbstractMeasurement)
            fn = obs_fieldnames_from_obj(m)
            os = [getfield(m, n) for n in fn]
            if isempty(os)
                throw(error("Did not find any observables in $m."))
            elseif length(os) == 1
                return $(func)(os[1])
            else
                return Dict(MonteCarloObservable.name(o) => $(func)(o) for o in os)
            end
        end
    end
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
    ConfigurationMeasurement(mc, model[, rate=1])

Measures configurations of the given Monte Carlo flavour and model. The rate of
measurements can be reduced with `rate`. (e.g. `rate=10` means 1 measurement per
10 sweeps)
"""
struct ConfigurationMeasurement{OT <: Observable} <: AbstractMeasurement
    obs::OT
    rate::Int64
    function ConfigurationMeasurement(mc, model, rate=1)
        o = Observable(typeof(mc.conf), "Configurations")
        new{typeof(o)}(o, rate)
    end
end
function measure!(m::ConfigurationMeasurement, mc, model, i::Int64)
    (i % m.rate == 0) && push!(m.obs, conf(mc))
    nothing
end


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


# Searches `obj` for fields which <: AbstractObservables
# if get_all_names = false, return array of fieldnames <: AbstractObservable
# if get_all_names = true, also return other fieldnames
function obs_fieldnames_from_obj(obj, get_all_names=false)
    fnames = fieldnames(typeof(obj))
    obs_names = [
        s for s in fnames if getfield(obj, s) isa AbstractObservable
    ]
    if get_all_names
        other_names = [
            s for s in fnames if !(getfield(obj, s) isa AbstractObservable)
        ]
        return obs_names, other_names
    else
        return obs_names
    end
end


# printing
function Base.show(io::IO, m::AbstractMeasurement)
    #  no parametrization -v    v- no MonteCarlo.
    typename = typeof(m).name.name
    observables, other = obs_fieldnames_from_obj(m, true)
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
    temp = obs_fieldnames_from_obj(m)
    observable_names = map(temp) do obs_fieldname
        MonteCarloObservable.name(getfield(m, obs_fieldname))
    end
    print(io, typename, "(\"", join(observable_names, "\", \""), "\")")
    nothing
end


"""
    measurements(mc, stage = :ME)

Returns a nested dictionary of measurements used in a given Monte Carlo
simulation `mc`.
By default, only measurements from the measurement stage (`:ME`) are returned.
To get measurements from the thermalization stage, use `stage = :TH`. To get all
measurements, use `stage = :all`.

```julia
julia> julia> measurements(dqmc)
Dict{Symbol,MonteCarlo.AbstractMeasurement} with 3 entries:
  :conf              => ConfigurationMeasurement…
  :Greens            => GreensMeasurement…
  :BosonEnergy       => BosonEnergyMeasurement…
```
"""
function measurements(mc::MonteCarloFlavor, stage = :ME)
    measurement_stage = (:ME, :me, :Measurement, :measurement)
    thermalization_stage = (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
    all_stages = (:all, :ALL)

    if stage in measurement_stage
        return mc.measurements
    elseif stage in thermalization_stage
        return mc.thermalization_measurements
    elseif stage in all_stages
        return Dict(
            :TH => mc.thermalization_measurements,
            :ME => mc.measurements
        )
    else
        throw(error(
            "The given stage `$stage` was not recognized. Try one of (" *
            join(string.([
                measurement_stage...,
                thermalization_stage...,
                all_stages...
            ]), ", ") * ")"
        ))
    end
end


"""
    observables(mc::MonteCarloFlavor, stage = :ME)

Returns a nested dictionary of all observables used during a given `stage` of
the given Monte Carlo simulation `mc`. By default the resulting dictionary will
contain measurements from the measurement (`:ME`) stage.

The thermalization stage can be accessed with `stage = :TH` and both stages can
be retrieved together using `stage = :all`.

The dictionary is generated by calling `observables(measurement)` for each
`measurement` in the given `stage` of the Monte Carlo simulation `mc`. By
default this will generate a nested dictionary indexed as
`observables(mc)[measurement_name][observable_name]`.

```julia
julia> observables(dqmc)
Dict{Symbol,Dict{String,MonteCarloObservable.AbstractObservable}} with 3 entries:
  :BosonEnergy       => Dict{String,MonteCarloObservable.AbstractObservable}("Bosonic Energy"=>LightObservable{Float64,20}())
  :conf              => Dict{String,MonteCarloObservable.AbstractObservable}("Configurations"=>Array{Int8,2} Observable…
  :Greens            => Dict{String,MonteCarloObservable.AbstractObservable}("Equal-times Green's function"=>LightObservable{Array{Complex{Float64},2},20}())
```
"""
function observables(mc::MonteCarloFlavor, stage = :ME)
    measurement_stage = (:ME, :me, :Measurement, :measurement)
    thermalization_stage = (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
    all_stages = (:all, :ALL)

    if stage in measurement_stage
        me_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
            k => observables(m) for (k, m) in mc.measurements
        )
        return me_obs

    elseif stage in thermalization_stage
        th_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
            k => observables(m) for (k, m) in mc.thermalization_measurements
        )
        return th_obs

    elseif stage in all_stages
        th_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
            k => observables(m) for (k, m) in mc.thermalization_measurements
        )
        me_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
            k => observables(m) for (k, m) in mc.measurements
        )
        return Dict(:TH => th_obs, :ME => me_obs)

    else
        throw(error(
            "The given stage `$stage` was not recognized. Try one of (" *
            join(string.([
                measurement_stage...,
                thermalization_stage...,
                all_stages...
            ]), ", ") * ")"
        ))
    end
end


"""
    push!(mc, tag::Symbol => MT::Type{<:AbstractMeasurement}[, stage=:ME])

Adds a new pair `tag => MT(mc, model)`, where `MT` is a type
`<: AbstractMeasurement`, to either the thermalization or measurement `stage`
(`:TH` or `:ME`) of the simulation `mc`.

Examples:
```
push!(mc, :conf => ConfigurationMeasurement)
push!(mc, :conf => ConfigurationMeasurement, stage=:ME)
```

See also: [`unsafe_push!`](@ref)
"""
function Base.push!(mc::MonteCarloFlavor, p::Pair{Symbol, T}, stage=:ME) where T
    tag, MT = p
    p[2] <: AbstractMeasurement || throw(ErrorException(
        "The given `tag => MT` pair should be of type " *
        "`Pair{Symbol, Type(<: AbstractMeasurement)}`, but is " *
        "`Pair{Symbol, Type{$(p[2])}}`."
    ))
    unsafe_push!(mc, tag => MT(mc, mc.model), stage)
end

"""
    unsafe_push!(mc, tag::Symbol => m::AbstractMeasurement[, stage=:ME])

Adds a pair `tag => m` to either the thermalization or measurement stage (`:TH`
or `:ME`) of the given simulation `mc`.

Note that this function is unsafe as it does not test whether `m` is a valid
measurement for the given simulation.

Examples:
```
push!(mc, :conf => ConfigurationMeasurement(mc, model))
push!(mc, :conf => ConfigurationMeasurement(mc, model), stage=:ME)
```

See also: [`MonteCarlo.push!`](@ref)
"""
function unsafe_push!(mc::MonteCarloFlavor, p::Pair{Symbol, <:AbstractMeasurement}, stage=:ME)
    measurement_stage = (:ME, :me, :Measurement, :measurement)
    thermalization_stage = (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
    all_stages = (:all, :ALL)

    if stage in measurement_stage
        push!(mc.measurements, p)
    elseif stage in thermalization_stage
        push!(mc.thermalization_measurements, p)
    elseif stage in all_stages
        push!(mc.thermalization_measurements, p)
        push!(mc.measurements, p)
    else
        throw(error(
            "The given stage `$stage` was not recognized. Try one of (" *
            join(string.([
                measurement_stage...,
                thermalization_stage...,
                all_stages...
            ]), ", ") * ")"
        ))
    end
end


"""
    delete!(mc, key[, stage=:MC])
    delete!(mc, MT::Type{<:AbstractMeasurement}[, stage=:MC])

Deletes a measurement from the given Monte Carlo simulation by key or by type.
When deleting by type, multiple measurements can be targeted using inheritance.
For example, `delete!(mc, IsingMeasurement)` wil delete all
`IsingEnergyMeasurement` and `IsingMagnmetizationMeasurement` objects.

Examples:
```
delete!(mc, :conf)
delete!(mc, ConfigurationMeasurement)
delete!(mc, ConfigurationMeasurement, stage=:ME)
```
"""
function Base.delete!(mc::MonteCarloFlavor, key::Symbol, stage=:ME)
    measurement_stage = (:ME, :me, :Measurement, :measurement)
    thermalization_stage = (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)

    if stage in measurement_stage
        delete!(mc.measurements, key)
    elseif stage in thermalization_stage
        delete!(mc.thermalization_measurements, key)
    else
        throw(error(
            "The given stage `$stage` was not recognized. Try one of (" *
            join(string.([
                measurement_stage...,
                thermalization_stage...
            ]), ", ") * ")"
        ))
    end
end

function Base.delete!(mc::MonteCarloFlavor, MT::Type{<: AbstractMeasurement}, stage=:ME)
    measurement_stage = (:ME, :me, :Measurement, :measurement)
    thermalization_stage = (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)

    if stage in measurement_stage
        ks = collect(keys(mc.measurements))
        for k in ks
            if typeof(mc.measurements[k]) <: MT
                delete!(mc.measurements, k)
            end
        end
        mc.measurements
    elseif stage in thermalization_stage
        ks = collect(keys(mc.thermalization_measurements))
        for k in ks
            if typeof(mc.thermalization_measurements[k]) <: MT
                delete!(mc.thermalization_measurements, k)
            end
        end
        mc.thermalization_measurements
    else
        throw(error(
            "The given stage `$stage` was not recognized. Try one of (" *
            join(string.([
                measurement_stage...,
                thermalization_stage...
            ]), ", ") * ")"
        ))
    end
end


"""
    save_measurements!(mc, filename[; force_overwrite=false, allow_rename=true])

Saves all measurements to `filename`.

If `force_overwrite = true` the file will
be overwritten if it already exists. If `allow_rename = true` random characters
will be added to the filename until it becomes unique.
"""
function save_measurements!(
        mc::MonteCarloFlavor, filename::String;
        force_overwrite = false, allow_rename = true
    )
    isfile(filename) && !force_overwrite && !allow_rename && throw(ErrorException(
        "Cannot save because \"$filename\" already exists. Consider setting " *
        "`allow_reanme = true` to adjust the filename or `force_overwrite = true`" *
        " to overwrite the file."
    ))
    if isfile(filename) && !force_overwrite && allow_rename
        while isfile(filename)
            # those map to 0-9, A-Z, a-z
            x = rand([(48:57)..., (65:90)..., (97:122)...])
            s = string(Char(x))
            filename = filename[1:end-4] * s * ".jld"
        end
    end

    measurement_dict = measurements(mc, :all)
    for (k0, v0) in measurement_dict # :TH or :ME
        for (k1, meas) in v0 # Measurement name
            entryname = "Measurements/$k0/$k1"
            save!(meas, filename, entryname)
        end
    end
end



"""
    load_measurements(filename)

Loads recorded measurements for a given `filename`. The output is formated like
`observables(mc)`

See also [`observables`](@ref)
"""
function load_measurements(filename::String)
    input = load(filename)
    output = Dict{Symbol, Dict{Symbol, Dict{String, AbstractObservable}}}()
    for (k1, v1) in input["Measurements"]
        # k1 = TH or ME
        push!(output, Symbol(k1) => Dict{Symbol, Dict{String, AbstractObservable}}())
        for (k2, v2) in v1
            # k2 is the key for Measurement
            push!(output[Symbol(k1)], Symbol(k2) => v2)
        end
    end
    output
end
