################################################################################
### Pre measurement rework
################################################################################


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
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
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
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
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
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
end


_get_shape(model::Model) = (length(lattice(model)),)
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
    if data["VERSION"] == 1
        OccupationMeasurement(data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
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
    if data["VERSION"] == 1
        x = data["x"]
        y = data["y"]
        z = data["z"]
        temp = similar(x.B.x_sum[1])
        tempy = similar(y.B.x_sum[1])
        data["type"](x, y, z, temp, tempy)
    else
        _load(data, AbstractMeasurement)
    end
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



to_logbinner(B::LogBinner) = B
to_logbinner(B::LightObservable) = B.B


function mask2iter(mask, iter::EachSitePairByDistance)
    # A[idxs] should output iter order
    # idxs = [mask_idx@iter1, mask_idx@iter2, ...]
    idxs = Vector{Int64}(undef, ndirections(iter))
    
    N = sqrt(length(iter))
    @assert N ≈ round(Int64, N)
    N = round(Int64, N)
    mask_idxs = Matrix{Int64}(undef, N, N)
    for (idx, src, trg) in getorder(mask)
        mask_idxs[src, trg] = idx
    end

    for i in 1:ndirections(iter)
        pairs = in_direction(iter, i)
        x = first(pairs)
        idx = mask_idxs[x[1], x[2]]
        for (src, trg) in pairs
            if mask_idxs[src, trg] != idx
                error("Mapping is not unqiue: $(mask_idxs[src, trg]) != $(idx)")
            end
        end
        idxs[i] = idx
    end

    idxs
end

function reorder(input::LogBinner, idxs::Vector)
    B = deepcopy(input)
    for lvl in eachindex(B.compressors)
        permute!(B.compressors[lvl].value, idxs)
        permute!(B.x_sum[lvl], idxs)
        permute!(B.x2_sum[lvl], idxs)
    end
    B
end

function reorder(input::LogBinner, idxs1::Vector, idxs2::Vector, idxs3::Vector)
    B = deepcopy(input)
    for lvl in eachindex(B.compressors)
        B.compressors[lvl].value = B.compressors[lvl].value[idxs1, idxs2, idxs3]
        B.x_sum[lvl] = B.x_sum[lvl][idxs1, idxs2, idxs3]
        B.x2_sum[lvl] = B.x2_sum[lvl][idxs1, idxs2, idxs3]
    end
    B
end



function DQMCMeasurement(mc::DQMC, m::GreensMeasurement)
    obs = to_logbinner(m.obs)
    temp = mean(obs)
    DQMCMeasurement{Greens, Nothing}(greens_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::BosonEnergyMeasurement)
    obs = to_logbinner(m.obs)
    DQMCMeasurement{Nothing, Nothing}(energy_boson, obs, 0.0)
end

function DQMCMeasurement(mc::DQMC, m::OccupationMeasurement)
    obs = to_logbinner(m.obs)
    temp = mean(obs)
    DQMCMeasurement{Greens, EachSiteAndFlavor}(occupation_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::MagnetizationMeasurement)
    x = to_logbinner(m.x)
    y = to_logbinner(m.y)
    z = to_logbinner(m.z)

    return (
        DQMCMeasurement{Greens, EachSite}(mx_kernel, x, mean(x)),
        DQMCMeasurement{Greens, EachSite}(my_kernel, y, mean(y)),
        DQMCMeasurement{Greens, EachSite}(mz_kernel, z, mean(z))
    )
end

function DQMCMeasurement(mc::DQMC, m::ChargeDensityCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)
    obs = reorder(to_logbinner(m.obs), idxs)
    temp = mean(obs)
    DQMCMeasurement{Greens, EachSitePairByDistance}(cdc_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::SpinDensityCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)

    x = reorder(to_logbinner(m.x), idxs)
    y = reorder(to_logbinner(m.y), idxs)
    z = reorder(to_logbinner(m.z), idxs)
    temp = mean(x)
    tempy = mean(y)
    return (
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_x_kernel, x, temp),
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_y_kernel, y, tempy),
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_z_kernel, z, temp)
    )
end

function DQMCMeasurement(mc::DQMC, m::PairingCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)
    pre = to_logbinner(m.obs)
    temp = mean(pre)
    D = size(temp, 2)
    obs = reorder(pre, idxs, idxs[1:D], idxs[1:D])
    DQMCMeasurement{Greens, EachLocalQuadByDistance{D}}(pc_kernel, obs, mean(obs))
end