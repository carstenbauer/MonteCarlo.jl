abstract type AbstractMeasurement end

# Interface

"""
    prepare!(measurement, mc, model)

This method is called between the thermalization and the measurement phase.
"""
function prepare!(m::AbstractMeasurement, mc, model)
    throw(MethodError(prepare!, (m, mc, model)))
end
"""
    measure!(measurement, mc, model)

Performs a measurement during the measurement phase.
"""
function measure!(m::AbstractMeasurement, mc, model)
    throw(MethodError(measure!, (m, mc, model)))
end
"""
    finish!(measurement, mc, model)

Finish a measurement. This method is called after the measurement phase.
"""
function finish!(m::AbstractMeasurement, mc, model)
    throw(MethodError(finish!, (m, mc, model)))
end



# called by simulation:
for function_name in (:prepare!, :measure!, :finish!)
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

default_measurements(mc) = Dict{Symbol, AbstractMeasurement}()
