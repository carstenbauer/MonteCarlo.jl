"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model)
    o = LightObservable(
        LogBinner(zeros(eltype(mc.s.greens), size(mc.s.greens))),
        "Equal-times Green's function",
        "Observables.jld",
        "Equal-times Green's function"
    )
    GreensMeasurement{typeof(o)}(o)
end
@bm function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end
function save_measurement(file::JLD.JldFile, m::GreensMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function load_measurement(data, ::Type{T}) where T <: GreensMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end


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
    o = LightObservable(Float64, name="Bosonic Energy", alloc=1_000_000)
    BosonEnergyMeasurement{typeof(o)}(o)
end
@bm function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end
function save_measurement(file::JLD.JldFile, m::BosonEnergyMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function load_measurement(data, ::Type{T}) where T <: BosonEnergyMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end


function default_measurements(mc::DQMC, model)
    Dict(
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end
