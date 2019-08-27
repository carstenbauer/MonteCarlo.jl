"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model)
    o = Observable(typeof(mc.s.greens), "Equal-times Green's function")
    GreensMeasurement{typeof(o)}(o)
end

prepare!(::GreensMeasurement, mc::DQMC, model) = nothing
function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end
finish!(::GreensMeasurement, mc::DQMC, model) = nothing



"""
    BosonEnergyMeasurement(mc::DQMC, model)

Measures the bosnic energy of the given DQMC simulation and model.

Note that this measurement requires `energy_boson(mc, model, conf)` to be
implemented for the specific `model`.
"""
struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function BosonEnergyMeasurement(mc::DQMC, model)
    o = Observable(Float64, "Bosonic Energy")
    BosonEnergyMeasurement{typeof(o)}(o)
end
prepare!(::BosonEnergyMeasurement, mc::DQMC, model) = nothing
function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end
finish!(::BosonEnergyMeasurement, mc::DQMC, model) = nothing


function default_measurements(mc::DQMC, model)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end
