"""
    Configurations(mc, model[; rate = 10])

Creates a `Configurations` object which accumulates configurations during the
simulation.
"""
struct Configurations{CT} <: AbstractConfiguartionAccumulator
    configs::Vector{CT}
    rate::Int64
end
function Configurations(
        mc::DQMC, model::Model; rate = 10, 
    )
    CT = typeof(compress(mc, model, conf(mc)))
    Configurations{CT}(Vector{CT}(), rate)
end
function Configurations(
        mc::MC, model::Model; rate = 10, 
    )
    CT = typeof(compress(mc, model, conf(mc)))
    Configurations{CT}(Vector{CT}(), rate)
end
function Base.push!(c::Configurations, mc, model, sweep)
    (sweep % c.rate == 0) && push!(c.configs, compress(mc, model, conf(mc)))
    nothing
end

# Base extensions
Base.length(c::Configurations) = length(c.configs)
Base.isempty(c::Configurations) = isempty(c.configs)
Base.lastindex(c::Configurations) = lastindex(c.configs)
Base.iterate(c::Configurations, i=1) = iterate(c.configs, i)
Base.getindex(c::Configurations, i) = c.configs[i]

# Compression
compress(mc, model, conf) = copy(conf)
decompress(mc, model, conf) = conf

compress(::DQMC, ::HubbardModelAttractive, c::Configurations) = BitArray(c .== 1)
function decompress(
        mc::DQMC{M, CB, CT, S}, ::HubbardModelAttractive, c::Configurations
    ) where {M, CB, CT, S}
    CT(2c .- 1)
end

function _save(file::JLD.JldFile, cs::Configurations, entryname::String="Configurations")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(cs))
    write(file, entryname * "/data", cs.configs)
    write(file, entryname * "/rate", cs.rate)
end
function _load(data::Dict, ::Type{T}) where T <: Configurations
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end
    data["type"](data["data"], data["rate"])
end


# For destroying configuration measurements
struct Void <: AbstractConfiguartionAccumulator end
Void(args...; kwargs...) = Void()
Base.push!(::Void, args...) = nothing
Base.length(::Void) = 0
Base.isempty(::Void) = true
Base.getindex(v::Void, i) = BoundsError(v, i)
Base.iterate(c::Void, i=1) = nothing

function _save(file::JLD.JldFile, ::Void, entryname::String="Configurations")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", Void)
end
_load(data::Dict, ::Type{Void}) = Void()