abstract type HubbardMeasurement <: AbstractMeasurement end

prepare!(::HubbardMeasurement, mc::DQMC, model::HubbardModelAttractive) = nothing
finish!(::HubbardMeasurement, mc::DQMC, model::HubbardModelAttractive) = nothing


"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: HubbardMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model::HubbardModelAttractive)
    o = Observable(typeof(mc.s.greens), "Equal-times Green's function")
    GreensMeasurement{typeof(o)}(o)
end

function measure!(m::GreensMeasurement, mc::DQMC, model::HubbardModelAttractive, i::Int64)
    push!(m.obs, greens(mc))
end

"""
    BosonEnergyMeasurement(mc::DQMC, model)

Measures the bosnic energy of the given DQMC simulation and model.
"""
struct BosonEnergyMeasurement{OT <: AbstractObservable} <: HubbardMeasurement
    obs::OT
end
function BosonEnergyMeasurement(mc::DQMC, model::HubbardModelAttractive)
    o = Observable(Float64, "Bosonic Energy")
    BosonEnergyMeasurement{typeof(o)}(o)
end

function measure!(m::BosonEnergyMeasurement, mc::DQMC, model::HubbardModelAttractive, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end


function default_measurements(mc::DQMC, model::HubbardModelAttractive)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end
