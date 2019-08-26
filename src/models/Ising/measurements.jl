abstract type IsingMeasurement <: AbstractMeasurement end

prepare!(::IsingMeasurement, mc::MC, model::IsingModel) = nothing
finish!(::IsingMeasurement, mc::MC, model::IsingModel) = nothing


struct IsingEnergyMeasurement <: IsingMeasurement
    invN::Float64
    E::Observable
    E2::Observable
    e::Observable
    C::Observable

    IsingEnergyMeasurement(mc::MC, model::IsingModel) = new(
        1.0 / model.l.sites,
        Observable(Float64, "Total energy"),
        Observable(Float64, "Total energy squared"),
        Observable(Float64, "Energy per site"),
        Observable(Float64, "Specific heat")
    )
end

function measure!(m::IsingEnergyMeasurement, mc::MC, model::IsingModel)
    push!(m.E, model.energy[])
    push!(m.E2, model.energy[]^2)
    push!(m.e, model.energy[] * m.invN)
    nothing
end

function finish!(m::IsingEnergyMeasurement, mc::MC, model::IsingModel)
    E = mean(m.E)
    E2 = mean(m.E2)
    push!(m.C, mc.p.beta^2 * m.invN * (E2 - E^2))
    nothing
end



struct IsingMagnetizationMeasurement <: IsingMeasurement
    invN::Float64
    M::Observable
    M2::Observable
    m::Observable
    chi::Observable

    IsingMagnetizationMeasurement(mc::MC, model::IsingModel) = new(
        1.0 / model.l.sites,
        Observable(Float64, "Total magnetization"),
        Observable(Float64, "Total magnetization squared"),
        Observable(Float64, "Magnetization per site"),
        Observable(Float64, "Magnetic susceptibility")
    )
end

function measure!(m::IsingMagnetizationMeasurement, mc::MC, model::IsingModel)
    M = abs(sum(mc.conf))
    push!(m.M, M)
    push!(m.M2, M^2)
    push!(m.m, M * m.invN)
    nothing
end

function finish!(m::IsingMagnetizationMeasurement, mc::MC, model::IsingModel)
    M = mean(m.M)
    M2 = mean(m.M2)
    push!(m.chi, mc.p.beta * m.invN * (M2 - M^2))
    nothing
end



function default_measurements(mc::MC, model::IsingModel)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Magn => IsingMagnetizationMeasurement(mc, model),
        :Energy => IsingEnergyMeasurement(mc, model)
    )
end
