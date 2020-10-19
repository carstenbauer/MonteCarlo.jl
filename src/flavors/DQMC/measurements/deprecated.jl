struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function save_measurement(file::JLDFile, m::GreensMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: GreensMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end



struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function save_measurement(file::JLDFile, m::BosonEnergyMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: BosonEnergyMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end



struct OccupationMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function OccupationMeasurement(m::GreensMeasurement{<: LightObservable}; capacity=capacity(m.obs.B))
    N = size(m.obs)[1]
    o = LightObservable(
        LogBinner(zeros(Float64, N), capacity=capacity),
        "State resolved Occupation",
        "Observables.jld",
        "Occ"
    )

    N = min(length(o.B.count), length(m.obs.B.count))
    if m.obs.B.count[N] > 1
        throw(OverflowError("OccupationMeasurement does not have enough capacity!"))
    end
    N = findlast(>(1), m.obs.B.count)
    N === nothing && return OccupationMeasurement{typeof(o)}(o)
    
    for i in 1:N
        o.B.x_sum[i]  .= m.obs.B.count[i] .- diag(m.obs.B.x_sum[i])
        o.B.x2_sum[i] .= m.obs.B.count[i] .- 2diag(m.obs.B.x_sum[i]) .+ diag(m.obs.B.x2_sum[i])
        o.B.count[i]   = m.obs.B.count[i]
        o.B.compressors[i].value .= 1 .- diag(m.obs.B.compressors[i].value)
        o.B.compressors[i].switch = m.obs.B.compressors[i].switch
    end
    OccupationMeasurement{typeof(o)}(o)
end
function save_measurement(file::JLDFile, m::OccupationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: OccupationMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end


_get_shape(model) = (length(lattice(model)),)
_get_shape(mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(mask::DistanceMask) = length(mask)
mask(m::AbstractMeasurement) = m.mask


abstract type SpinOneHalfMeasurement <: AbstractMeasurement end



struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
end
function save_measurement(file::JLDFile, m::ChargeDensityCorrelationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: ChargeDensityCorrelationMeasurement
    @assert data["VERSION"] == 1
    OccupationMeasurement(data["obs"])
end



struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: AbstractArray,
        ATy <: AbstractArray
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    tempy::ATy
end
function save_measurement(file::JLDFile, m::MagnetizationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", MagnetizationMeasurement)
    write(file, entryname * "/x", m.x)
    write(file, entryname * "/y", m.y)
    write(file, entryname * "/z", m.z)
    nothing
end
function _load(data, ::Type{T}) where T <: MagnetizationMeasurement
    @assert data["VERSION"] == 1
    x = data["x"]
    y = data["y"]
    z = data["z"]
    temp = similar(x.B.x_sum[1])
    tempy = similar(y.B.x_sum[1])
    data["type"](x, y, z, temp, tempy)
end



struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    mask::MT
end



struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: DistanceMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
    rsm::RestrictedSourceMask
end