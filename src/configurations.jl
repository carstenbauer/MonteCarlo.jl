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

See also [`BufferedConfigRecorder`](@ref), [`Discarder`](@ref)
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
### BufferedConfigRecorder
################################################################################



struct FilePath
    is_relative::Bool
    relative_path::String
    absolute_path::String
end
RelativePath(filename) = FilePath(true, filename, joinpath(pwd(), filename))
AbsolutePath(filename) = FilePath(false, filename, filename)


"""
    BufferedConfigRecorder(mc, model, filename[; rate = 10, chunk_size = 1000])

Creates a `BufferedConfigRecorder` object which accumulates configurations 
during the simulation in an in-memory buffer. The buffer has a set `chunk_size`.
Every time it becomes it is commited to a JLD2 file `filename` and reset.

This uses the same configuration compression used in ConfigRecorder. I.e. every
`push!`ed configuration is passed to `compress(mc, model, conf)`. 

See also [`ConfigRecorder`](@ref), [`Discarder`](@ref)
"""
mutable struct BufferedConfigRecorder{CT} <: AbstractRecorder
    filename::FilePath
    buffer::Vector{CT}
    rate::Int64
    idx::Int64
    chunk::Int64
    total_length::Int64
    save_idx::Int64
end
function BufferedConfigRecorder(fn::String, buffer, rate, idx, chunk, N, sidx)
    BufferedConfigRecorder(AbsolutePath(fn), buffer, rate, idx, chunk, N, sidx)
end
function BufferedConfigRecorder{CT}(filename, rate = 10, chunk_size = 1000) where CT
    BufferedConfigRecorder(filename, Vector{CT}(undef, chunk_size), rate, 1, 1, 0, -1)
end
function BufferedConfigRecorder(MC::Type, M::Type, filename; rate = 10, chunk_size = 1000)
    BufferedConfigRecorder{compressed_conf_type(MC, M)}(filename, rate, chunk_size)
end
function Base.push!(cr::BufferedConfigRecorder, mc, model, sweep)
    if (sweep % cr.rate == 0)
        _push!(cr, compress(mc, model, conf(mc)))
    end
end
function _push!(cr::BufferedConfigRecorder, data)
    if cr.idx == -1
        # The recorder is not working on the right chunk
        if cr.save_idx != cr.total_length
            @error "Stored data length does not match expected data length. Reseting head to last checkpoint."
            cr.total_length = cr.save_idx
        end
        # get chunk, idx of next writeable location
        chunk, idx = idx2chunk_idx(cr, cr.total_length+1)
        # load chunk if not loaded and required
        chunk != cr.chunk && idx != 1 && load_chunk!(cr, chunk)
        cr.idx = idx
    elseif cr.idx > length(cr.buffer)
        # The buffer is full - commit to file
        save_final_chunk!(cr)
        cr.chunk += 1
        cr.idx = 1
    end
    cr.buffer[cr.idx] = data
    cr.idx += 1
    cr.total_length += 1
    nothing
end

# Base extensions
Base.length(cr::BufferedConfigRecorder) = cr.total_length
Base.isempty(cr::BufferedConfigRecorder) = cr.total_length == 0
Base.lastindex(cr::BufferedConfigRecorder) = cr.total_length
Base.iterate(cr::BufferedConfigRecorder, i=1) = iterate(cr.configs, i)

function Base.getindex(cr::BufferedConfigRecorder, i)
    @boundscheck 1 <= i <= cr.total_length
    chunk, idx = idx2chunk_idx(cr, i)
    if chunk != cr.chunk
        # Save unsaved data in buffer
        cr.save_idx < cr.total_length && save_final_chunk!(cr)
        load_chunk!(cr, chunk)
    end
    return cr.buffer[idx]
end

# maps index to (chunk_idx, buffer_idx)
function idx2chunk_idx(cr::BufferedConfigRecorder, i)
    chunk_size = length(cr.buffer)
    chunk = div(i-1, chunk_size) + 1
    idx = mod1(i, chunk_size)
    return chunk, idx
end

# chunk loading and saving
function save_final_chunk!(cr::BufferedConfigRecorder)
    if cr.total_length != (cr.chunk - 1) * length(cr.buffer) + cr.idx - 1
        error("Attempting to save non-final chunk $(cr.chunk)!")
    end
    save_chunk!(cr)
    cr.save_idx = cr.total_length
    nothing
end

function save_chunk!(cr::BufferedConfigRecorder, chunk = cr.chunk)
    @boundscheck chunk > 0 && (chunk-1) * length(cr.buffer) < cr.total_length
    k = string(chunk)
    JLD2.jldopen(cr.filename.absolute_path, "a+", compress = true) do file
        if haskey(file, k)
            @debug "Replacing chunk $k"
            delete!(file, k)
        end
        file[k] = cr.buffer
    end
    nothing
end

function load_chunk!(cr::BufferedConfigRecorder, chunk)
    @boundscheck (chunk-1) * length(cr.buffer) < cr.save_idx
    JLD2.jldopen(cr.filename.absolute_path, "r") do file
        copyto!(cr.buffer, file[string(chunk)])
    end
    cr.chunk = chunk
    cr.idx = -1
end



function _save(file, cr::BufferedConfigRecorder, entryname::String="configs")
    if cr.save_idx != cr.total_length || !isfile(cr.filename.absolute_path)
        save_final_chunk!(cr)
    end

    # adjust relative FilePath
    if cr.filename.is_relative
        path, _ = splitdir(file.path)
        filepath = joinpath(path, cr.filename.relative_path)
        if filepath != cr.filename.absolute_path
            if isfile(filepath)
                new_filepath = _generate_unique_filename(filepath)
                @warn "Config file '$filepath' already exists, renaming to '$new_filepath' to avoid collision."
                filepath = new_filepath
            end
            mv(cr.filename.absolute_path, filepath)
        end
        cr.filename = FilePath(true, string(filepath[length(path)+1:end]), filepath)
    end

    write(file, entryname * "/VERSION", 2)
    write(file, entryname * "/tag", "BufferedConfigRecorder")
    write(file, entryname * "/filename", cr.filename)
    write(file, entryname * "/buffer", cr.buffer)
    write(file, entryname * "/rate", cr.rate)
    write(file, entryname * "/total_length", cr.total_length)
    write(file, entryname * "/save_idx", cr.total_length)
    nothing
end
function _load(data, ::Val{:BufferedConfigRecorder})
    if !(data["VERSION"] <= 2)
        throw(ErrorException("Failed to load BufferedConfigRecorder version $(data["VERSION"])"))
    end
    BufferedConfigRecorder(
        data["filename"], data["buffer"], data["rate"], -1, -1, 
        data["total_length"], data["save_idx"]
    )
end



################################################################################
### Discarder
################################################################################



"""
    Discarder(args...; kwargs...)

Creates an object that discards all configurations pushed to it.

See also: [`BufferedConfigRecorder`](@ref), [`ConfigRecorder`](@ref)
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



################################################################################
### Utility
################################################################################



function Base.merge!(target::BufferedConfigRecorder, source::ConfigRecorder; rate = 1)
    for i in eachindex(source.configs)
        if i % rate == 0
            _push!(target, source.configs[i])
        end
    end
end