abstract type HeisenbergMeasurement <: AbstractMeasurement end


"""
    HeisenbergEnergyMeasurement(mc::MC, model::HeisenbergModel, rate=1)

Measures observables related to the energy of the HeisenbergModel. This includes
- `E`: the total energy
- `E2`: the squared energy
- `e`: the energy per spin
- `C`: the specific heat
"""
struct HeisenbergEnergyMeasurement <: HeisenbergMeasurement
    invN::Float64
    E::Observable
    E2::Observable
    e::Observable
    C::Observable

    HeisenbergEnergyMeasurement(mc::MC, model::HeisenbergModel) = new(
        1.0 / model.l.sites,
        Observable(Float64, "Total energy"),
        Observable(Float64, "Total energy squared"),
        Observable(Float64, "Energy per site"),
        Observable(Float64, "Specific heat")
    )
end

function prepare!(m::HeisenbergEnergyMeasurement, mc::MC, model::HeisenbergModel)
    model.energy = energy(mc, model, mc.conf)
    nothing
end

function measure!(m::HeisenbergEnergyMeasurement, mc::MC, model::HeisenbergModel, i::Int64)
    push!(m.E, model.energy)
    push!(m.E2, model.energy^2)
    push!(m.e, model.energy * m.invN)
    nothing
end

function finish!(m::HeisenbergEnergyMeasurement, mc::MC, model::HeisenbergModel)
    E = mean(m.E)
    E2 = mean(m.E2)
    push!(m.C, mc.p.beta^2 * m.invN * (E2 - E^2))
    nothing
end


"""
    HeisenbergMagnetizationMeasurement(mc::MC, model::HeisenbergModel, rate=1)

Measures observables related to the magnetization of the HeisenbergModel. This
includes
- `M`: the total magnetization
- `M2`: the squared magnetization
- `m`: the magnetization per spin
# - `chi`: the magentic susceptibility (nope)
"""
struct HeisenbergMagnetizationMeasurement <: HeisenbergMeasurement
    invN::Float64
    M::LightObservable
    M2::LightObservable
    m::LightObservable
    # chi::Observable

    HeisenbergMagnetizationMeasurement(mc::MC, model::HeisenbergModel) = new(
        1.0 / model.l.sites,
        LightObservable(HeisenbergSpin, name="Total magnetization", alloc=10_000_000),
        LightObservable(HeisenbergSpin, name="Total magnetization squared", alloc=10_000_000),
        LightObservable(HeisenbergSpin, name="Magnetization per site", alloc=10_000_000),
        # Observable(Float64, "Magnetic susceptibility")
    )
end

function measure!(m::HeisenbergMagnetizationMeasurement, mc::MC, model::HeisenbergModel, i::Int64)
    # TODO Chi
    # M = abs(sum(mc.conf))
    M = sum(mc.conf)
    push!(m.M, M)
    push!(m.M2, M.^2)
    push!(m.m, M * m.invN)
    nothing
end

# function finish!(m::HeisenbergMagnetizationMeasurement, mc::MC, model::HeisenbergModel)
#     M = mean(m.M)
#     M2 = mean(m.M2)
#     # push!(m.chi, mc.p.beta * m.invN * (M2 - M^2))
#     nothing
# end



function default_measurements(mc::MC, model::HeisenbergModel)
    Dict(
        :Magn => HeisenbergMagnetizationMeasurement(mc, model),
        :Energy => HeisenbergEnergyMeasurement(mc, model)
    )
end
