# Saving and loading happens in a nested fashion:
# save(filename, mc) calls
#   save_mc(filename, mc) calls
#       save_model(filename, mc.model) calls
#           save_lattice(filename, mc.lattice) (TODO, not used yet)
#       save_measurements(filename, mc) calls
#           save_measurement(filename, measurement)
#
# loading follows the same structure
# > Each level (beyond the outermost save) also has an argument `entryname::String`.
# > Each save_something() should write a "VERSION" and a "type" to the file. The
#   former simplifies updates down the line, the latter allows for dispatch.
# > Each load_something should be dispatched on a type, e.g.
#   `load_model(data, ::Type(HubbardModelAttractive))`


# Loading:
# load(filename[, groups]) 
#   - loads from filename, maybe specific group
#   - jumps into _load at fitting part
# _load(data, ::Type{...})
#   - loads from Dict or JLD2.JLDFile or JLD2.Group (data) to some type (for dispatch)

# TODOs:
# - rename save_x to _save(filename, ::X, entryname) (mirror _load)
# - make _load less volatile to changes, i.e. change Type -> Val{Symbol}


"""
    save(filename, mc; overwrite=false, rename=true)

Saves the given MonteCarlo simulation `mc` to a JLD-file `filename`.

If `rename = true` the filename will be adjusted if it already exists. If
`overwrite = true` it will be overwritten. In this case a temporary backup
will be created. If neither are true an error will be thrown.
"""
function save(
        filename::String, mc::MonteCarloFlavor; 
        overwrite = false, rename = true, compress = true, kwargs...
    )
    # handle ranming and overwriting
    isfile(filename) && !overwrite && !rename && throw(ErrorException(
        "Cannot save because \"$filename\" already exists. Consider setting " *
        "`rename = true` to adjust the filename or `overwrite = true`" *
        " to overwrite the file."
    ))
    if isfile(filename) && !overwrite && rename
        filename = _generate_unique_filename(filename)
    end

    temp_filename = ""
    if isfile(filename) && overwrite
        parts = splitpath(filename)
        parts[end] = "." * parts[end]
        temp_filename = _generate_unique_filename(joinpath(parts...))
        mv(filename, temp_filename)
    end

    mode = isfile(filename) ? "r+" : "w"
    file = JLD2.jldopen(filename, mode, compress=compress; kwargs...)

    try
        write(file, "VERSION", 2)
        write(file, "git", git)
        _save(file, "MC", mc)
        save_rng(file)
    catch e
        if overwrite && !isempty(temp_filename) && isfile(temp_filename)
            rm(filename)
            mv(temp_filename, filename)
        end
        @error exception = e
    finally
        close(file)
    end

    if overwrite && !isempty(temp_filename) && isfile(temp_filename)
        rm(temp_filename)
    end

    return filename
end

# Something like
# existing_file.jld -> existing_file_aJ3c.jld
function _generate_unique_filename(filename)
    isfile(filename) || return filename
    x = rand("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    s = "_$(Char(x))"
    parts = split(filename, '.')
    filename = join(parts[1:end-1], '.') * s
    while isfile(filename * '.' * parts[end])
        x = rand("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        s = string(Char(x))
        filename = filename * s
    end
    filename * '.' * parts[end]
end

function to_tag(data::FileLike)
    haskey(data, "tag") && return Val(Symbol(data["tag"]))
    haskey(data, "type") && return to_tag(data["type"])
    error("Failed to get tag from $data")
end
to_tag(::Type{<: AbstractLattice}) = Val(:Generic)
to_tag(::Type{<: Model}) = Val(:Generic)


"""
    load(filename[, groups...])

Loads a MonteCarlo simulation from the given file path. If `groups` are given, 
loads a specific part of the file.
"""
function load(filename::String, groups::String...)
    @assert isfile(filename) "File must exist"
    @assert endswith(filename, "jld2") "File must be a JLD2 file"

    output = try
        data = JLD2.jldopen(filename, "r")

        if data["VERSION"] == 1
            _git = (
                branch = "master", 
                commit = "a4dbd321f551e6adc079370b033555ba9a2f75e5", 
                dirty = false
            )
        else
            _git = data["git"]
        end

        output = try 
            if haskey(data, "MC") && !("MC" in groups)
                _load(data, "MC", groups...) else _load(data, groups...)
            end
        catch e
            git.branch != _git.branch && @info("Git branch missmatch $(git.branch) ≠ $(_git.branch)")
            git.commit != _git.commit && @info("Git commit missmatch $(git.commit) ≠ $(_git.commit)")
            git.dirty && @info("Repository is currently dirty.")
            _git.dirty && @info("Repository was dirty when the file was saved.")
            rethrow()
        finally
            endswith(filename, "jld2") && close(data)
        end
        output
    catch e
        println("Error loading file $filename:")
        rethrow()
    end
        
    return output
end

_load(data, g1::String, g2::String, gs::String...) = _load(data[g1], g2, gs...)
function _load(data, g::String)
    if !(data["VERSION"] in (1, 2))
        throw(ErrorException("Failed to load $(filepath(data)) version $(data["VERSION"])"))
    end

    haskey(data[g], "RNG") && load_rng!(data)

    _load(data[g], to_tag(data[g]))
end
_load(data) = _load(data, to_tag(data))


"""
    resume!(filename[; kwargs...])

Resumes a Monte Carlo simulation from a savefile generated by `run!(mc)`. Takes
the same keyword arguments as `run!`. Returns the simulation and the state
returned by `run!`.

See also: [`run!`](@ref)
"""
function resume!(filename; kwargs...)
    @assert endswith(filename, "jld2")
    data = JLD2.jldopen(filename, "r")

    if !(data["VERSION"] in (1, 2))
        throw(ErrorException("Failed to load $filename version $(data["VERSION"])"))
    end

    mc = _load(data["MC"], to_tag(data["MC"]))
    resume_init!(mc)
    load_rng!(data)
    close(data)

    state = run!(mc; kwargs...)
    mc, state
end


to_tag(::JLD2.UnknownType) = Val{:UNKNOWN}
function _load(data, ::Val{:UNKNOWN})
    @info "Failed to load (Unknowntype)"
    @info "Available fields: $(keys(data))"
    @info "You may be missing external packages."
    nothing
end


function _save(file::FileLike, entryname::String, data)
    write(file, entryname * "/VERSION", 0)
    write(file, entryname * "/tag", "Generic")
    write(file, entryname * "/data", data)
    nothing
end

_load(data, ::Val{:Generic}) = data["data"]


const _GLOBAL_RNG = if VERSION < v"1.3.0"
    Random.GLOBAL_RNG
elseif VERSION < v"1.7.0" 
    Random.default_rng()
else
    copy(Random.default_rng())
end


function save_rng(file::FileLike; rng = _GLOBAL_RNG, entryname::String="RNG")
    try
        write(file, entryname, rng)
    catch e
        error("Error while saving RNG state: ", e)
    end
end

"""
    load_rng!(data[; rng = _GLOBAL_RNG, entryname = "RNG"])

Loads an RNG from a given `data` dictinary generated by `JLD.load()` to `rng`.
"""
function load_rng!(data; rng = _GLOBAL_RNG, entryname::String="RNG")
    try
        copy!(rng, data[entryname])
    catch e
        error("Error while restoring RNG state: ", e)
    end
end


