abstract type IsingMeasurement <: AbstractMeasurement end


"""
    IsingEnergyMeasurement(mc::MC, model::IsingModel, rate=1)

Measures observables related to the energy of the IsingModel. This includes
- `E`: the total energy
- `E2`: the squared energy
- `e`: the energy per spin
- `C`: the specific heat
"""
struct IsingEnergyMeasurement <: IsingMeasurement
    invN::Float64
    E::FullBinner
    E2::FullBinner
    e::FullBinner
    C::FullBinner
end
function IsingEnergyMeasurement(mc::MC, model::IsingModel)
    IsingEnergyMeasurement(
        1.0 / model.l.sites,
        FullBinner(Float64),
        FullBinner(Float64),
        FullBinner(Float64),
        FullBinner(Float64)
    )
end

function measure!(m::IsingEnergyMeasurement, mc::MC, model::IsingModel, i::Int64)
    E = energy(mc, model, mc.conf)
    push!(m.E, E)
    push!(m.E2, E^2)
    push!(m.e, E * m.invN)
    nothing
end

function finish!(m::IsingEnergyMeasurement, mc::MC, model::IsingModel)
    E = mean(m.E)
    E2 = mean(m.E2)
    push!(m.C, mc.p.beta^2 * m.invN * (E2 - E^2))
    nothing
end


"""
    IsingMagnetizationMeasurement(mc::MC, model::IsingModel, rate=1)

Measures observables related to the magnetization of the IsingModel. This
includes
- `M`: the total magnetization
- `M2`: the squared magnetization
- `m`: the magnetization per spin
- `chi`: the magentic susceptibility
"""
struct IsingMagnetizationMeasurement <: IsingMeasurement
    invN::Float64
    M::FullBinner
    M2::FullBinner
    m::FullBinner
    chi::FullBinner
end
function IsingMagnetizationMeasurement(mc::MC, model::IsingModel)
    IsingMagnetizationMeasurement(
        1.0 / model.l.sites,
        FullBinner(Float64),
        FullBinner(Float64),
        FullBinner(Float64),
        FullBinner(Float64)
    )
end

function measure!(m::IsingMagnetizationMeasurement, mc::MC, model::IsingModel, i::Int64)
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
        :Magn => IsingMagnetizationMeasurement(mc, model),
        :Energy => IsingEnergyMeasurement(mc, model)
    )
end
