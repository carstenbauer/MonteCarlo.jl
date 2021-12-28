"""
    DummyModel()

If an unknown or undefined model is loaded a `DummyModel` will be created to 
allow the load to succeed and keep the data in check. 
"""
struct DummyModel <: Model
    data::Dict{String, Any}
end

Base.show(io::IO, model::DummyModel) = print(io, "DummyModel()")

hopping_matrix_type(::Type{DQMC}, ::DummyModel) = Matrix{Float64}
greens_matrix_type( ::Type{DQMC}, ::DummyModel) = Matrix{Float64}
interaction_matrix_type(::Type{DQMC}, ::DummyModel) = Diagonal{Float64, Vector{Float64}}
greenseltype(::Type{DQMC}, m::DummyModel) = Float64
hoppingeltype(::Type{DQMC}, m::DummyModel) = Float64



function save_model(file::JLDFile, m::DummyModel, entryname::String="Model")
    close(file)
    error("DummyModel cannot be saved.")
end

function _load_model(data, ::Val)
    tag = to_tag(data)
    @warn "Failed to load $tag, creating DummyModel"
    DummyModel(_load_to_dict(data))
end

_load_to_dict(data) = data
function _load_to_dict(data::Union{FileWrapper{<: JLD2.Group}, JLD2.Group})
    output = Dict{String, Any}()
    for key in keys(data)
        push!(output, key => _load_to_dict(data[key]))
    end
    output
end