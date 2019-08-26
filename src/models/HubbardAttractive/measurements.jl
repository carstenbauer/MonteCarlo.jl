abstract type HubbardMeasurements end

prepare!(::IsingMeasurement, mc::MC, model::IsingModel) = nothing
finish!(::IsingMeasurement, mc::MC, model::IsingModel) = nothing

struct GreensMeasurement{OT <: AbstractObservable} <: HubbardMeasurements
    obs::OT
    function GreensMeasurement()
        o = Observable(typeof(mc.s.greens), "Equal-times Green's function")
        new{typeof(o)}(o)
    end
end

function measure!(m::GreensMeasurement, mc::DQMC, model::HubbardModelAttractive, i::Int64)
    push!(m.obs, greens(mc))
end

struct BosonEnergyMeasurement{OT <: AbstractObservable} <: HubbardMeasurements
    obs::OT
    function BosonEnergyMeasurement()
        o = Observable(typeof(mc.s.greens), "Bosonic Energy")
        new{typeof(o)}(o)
    end
end

function measure!(m::BosonEnergyMeasurement, mc::DQMC, model::HubbardModelAttractive, i::Int64)
    push!(m.obs, energy_boson(mc, m, conf))
end


function default_measurements(mc::DQMC, model::HubbardModelAttractive)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :Boson_Energy => BosonMeasurement(mc, model)
    )
end
