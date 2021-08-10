abstract type AbstractRecorder end



################################################################################
### ConfigRecorder
################################################################################



"""
    ConfigRecorder(mc, model[; rate = 10])

Creates a `ConfigRecorder` object which accumulates configurations during the
simulation.

ConfigRecorder pushed to a `ConfigRecorder` objects are forwarded to 
`compress(mc, model, conf)` for potential compression. When replaying a 
simulation `decompress(mc, model, conf)` is called to undo this action. You may
define both to save on memory/disk space.

See also [`Discarder`](@ref)
"""
struct ConfigRecorder{CT} <: AbstractRecorder
    configs::Vector{CT}
    rate::Int64
end
ConfigRecorder{CT}(rate::Integer = 10) where CT = ConfigRecorder{CT}(Vector{CT}(), rate)
function ConfigRecorder(MC::Type, M::Type, rate::Integer = 10)
    ConfigRecorder{compressed_conf_type(MC, M)}(rate)
end
function Base.push!(c::ConfigRecorder, mc, model, sweep)
    (sweep % c.rate == 0) && push!(c.configs, compress(mc, model, conf(mc)))
    nothing
end

# Base extensions
Base.length(c::ConfigRecorder) = length(c.configs)
Base.isempty(c::ConfigRecorder) = isempty(c.configs)
Base.lastindex(c::ConfigRecorder) = lastindex(c.configs)
Base.iterate(c::ConfigRecorder, i=1) = iterate(c.configs, i)
Base.getindex(c::ConfigRecorder, i) = c.configs[i]

# Compression
compress(mc, model, conf) = copy(conf)
decompress(mc, model, conf) = conf

function _save(file::JLDFile, cs::ConfigRecorder, entryname::String="configs")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "ConfigRecorder")
    write(file, entryname * "/type", typeof(cs))
    write(file, entryname * "/data", cs.configs)
    write(file, entryname * "/rate", cs.rate)
end
function _load(data, ::Val{:ConfigRecorder})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load $T version $(data["VERSION"])"))
    end
    data["type"](data["data"], data["rate"])
end
to_tag(::Type{<: ConfigRecorder}) = Val(:ConfigRecorder)



################################################################################
### Discarder
################################################################################



"""
    Discarder(args...; kwargs...)

Creates an object that discards all configurations pushed to it.

See also: [`ConfigRecorder`](@ref)
"""
struct Discarder <: AbstractRecorder end
Discarder(args...; kwargs...) = Discarder()
Base.push!(::Discarder, args...) = nothing
Base.length(::Discarder) = 0
Base.isempty(::Discarder) = true
Base.getindex(v::Discarder, i) = BoundsError(v, i)
Base.iterate(c::Discarder, i=1) = nothing

function _save(file::JLDFile, ::Discarder, entryname::String="configs")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "Discarder")
end
_load(data, ::Val{:Discarder}) = Discarder()
to_tag(::Type{<: Discarder}) = Val(:Discarder)