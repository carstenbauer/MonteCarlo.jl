"""
    DummyModel()

If an unknown or undefined model is loaded a `DummyModel` will be created to 
allow the load to succeed and keep the data in check. 
"""
struct DummyModel <: Model
    data::Dict{String, Any}
end

Base.show(io::IO, model::DummyModel) = print(io, "DummyModel()")

function Base.getproperty(obj::DummyModel, field::Symbol)
    if hasfield(DummyModel, field)
        return getfield(obj, field)
    else
        return getfield(obj, :data)[string(field)]
    end
end


choose_field(::DummyModel) = DensityHirschField
unique_flavors(::DummyModel) = 1
lattice(m::DummyModel) = get(m.data, "l", get(m.data, "lattice", nothing))
hopping_matrix(m::DummyModel) = fill(1.0, length(lattice(m)), length(lattice(m)))


function _save(file::FileLike, entryname::String, ::DummyModel)
    # TODO is this ok?
    close(file)
    error("DummyModel cannot be saved.")
end

function load_model(data, ::Val)
    tag = to_tag(data)
    @warn "Failed to load $tag, creating DummyModel"
    dict = _load_to_dict(data)
    if haskey(dict, "data")
        x = pop!(dict, "data")
        push!(dict, x...)
    end
    DummyModel(dict)
end

function _load_to_dict(data::FileLike)
    output = Dict{String, Any}()
    for key in keys(data)
        try
            push!(output, key => _load(data[key]))
        catch e
            push!(output, key => _load_to_dict(data[key]))
        end
    end
    output
end

function _load_to_dict(data)
    if parentmodule(typeof(data)) == JLD2.ReconstructedTypes
        return Dict(map(fieldnames(typeof(data))) do f
            string(f) => _load_to_dict(getfield(data, f))
        end)
    else
        return data
    end
end
