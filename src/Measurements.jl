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
