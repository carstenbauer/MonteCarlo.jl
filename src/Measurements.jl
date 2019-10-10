################################################################################
# Interface
# A Measurement must inherit from AbstractMeasurement and implement the
# following functions.

abstract type AbstractMeasurement end


"""
    measure!(measurement, mc, model, sweep_index)

Performs a measurement during the measurement phase.
"""
function measure!(m::AbstractMeasurement, mc, model, i)
    throw(MethodError(measure!, (m, mc, model)))
end


###################################
# You may also implement


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
    fns = fieldnames(typeof(m))
    os = [getfield(m, fn) for fn in fns if getfield(m, fn) isa AbstractObservable]
    Dict{String, AbstractObservable}(MonteCarloObservable.name(o) => o for o in os)
end


"""
    save_measurement(filename, measurement, entryname)

Saves a measurement to a jld-file `filename` in group `entryname`.

The default implementation saves the full measurement object. This requires the
default constructor to be available, i.e. construction by passing fields. The
group `entryname` follows the structure given by `measurements()`, e.g.
`ME/config` for a `ConfigurationMeasurement` with tag `:config` in the
measurement phase.

See also [`save_measurements!`](@ref), [`measurements`](@ref), [`load`](@ref)
"""
function save_measurement(filename::String, m::AbstractMeasurement, entryname::String)
    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, entryname * "/VERSION", 0)
        write(f, entryname * "/type", typeof(m))
        write(f, entryname * "/data", m)
    end
    nothing
end
function load_measurement(data, ::Type{T}) where {T <: AbstractMeasurement}
    @assert data["VERSION"] == 0
    data["data"]
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
end
function ConfigurationMeasurement(mc::MonteCarloFlavor, model::Model, rate=1)
    o = Observable(typeof(mc.conf), "Configurations")
    ConfigurationMeasurement{typeof(o)}(o, rate)
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
    print(io, typename, "(\"", join(observable_names, "\", \""), "\")")
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
    observables(mc::MonteCarloFlavor)

Returns a nested dictionary of all observables used in a given Monte Carlo
simulation `mc`. The result `obs` is indexed as `obs[stage][measurement][name]`,
where `stage` is `:TH` (thermalization stage) or `:ME` (measurement stage),
`measurement::Symbol` is the name of the measurement and `name::String` is the
name of the observable.
"""
function observables(mc::MonteCarloFlavor)
    th_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
        k => observables(m) for (k, m) in mc.thermalization_measurements
    )
    me_obs = Dict{Symbol, Dict{String, AbstractObservable}}(
        k => observables(m) for (k, m) in mc.measurements
    )

    return Dict(:TH => th_obs, :ME => me_obs)
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
    if stage in (:ME, :me, :Measurement, :measurement)
        push!(mc.measurements, p)
    elseif stage in (:TH, :th, :Thermalization, :thermalization, :thermalization_measurements)
        push!(mc.thermalization_measurements, p)
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

Examples:
```
delete!(mc, :conf)
delete!(mc, ConfigurationMeasurement)
delete!(mc, ConfigurationMeasurement, stage=:ME)
```
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


"""
    save_measurements!(mc, filename[; force_overwrite=false, allow_rename=true])

Saves all measurements to `filename`.

If `force_overwrite = true` the file will
be overwritten if it already exists. If `allow_rename = true` random characters
will be added to the filename until it becomes unique.
"""
function save_measurements(
        filename::String, mc::MonteCarloFlavor, entryname::String="";
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

    !isempty(entryname) && !endswith(entryname, "/") && (entryname *= "/")
    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, entryname * "VERSION", 1)
    end
    measurement_dict = measurements(mc)
    for (k0, v0) in measurement_dict # :TH or :ME
        for (k1, meas) in v0 # Measurement name
            _entryname = entryname * "$k0/$k1"
            save_measurement(filename, meas, _entryname)
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
    input = JLD.load(filename)
    if input["VERSION"] == 1
        return load_measurements(input["MC"]["Measurements"])
    else
        throw(Meta.ParseError(
            "Failed to find `MC/Measurements` in $filename."
        ))
    end
end
function load_measurements(data::Dict)
    @assert data["VERSION"] == 1
    !haskey(data, "ME") && @warn "No measurement stage found (key \"ME\" missing)"
    !haskey(data, "TH") && @warn "No thermalization stage found (key \"TH\" missing)"

    Dict{Symbol, Dict{Symbol, AbstractMeasurement}}(
        :TH => Dict{Symbol, AbstractMeasurement}(
            if haskey(data, "TH")
                [Symbol(k) => load_measurement(v, v["type"]) for (k, v) in data["TH"]]
            else
                []
            end
        ),
        :ME => Dict{Symbol, AbstractMeasurement}(
            if haskey(data, "ME")
                [Symbol(k) => load_measurement(v, v["type"]) for (k, v) in data["ME"]]
            else
                []
            end
        )
    )
end
