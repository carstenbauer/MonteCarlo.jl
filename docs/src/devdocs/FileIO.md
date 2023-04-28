# FileIO

MonteCarlo.jl uses JLD2 for FileIO. While it is possible to save structs directly with JLD2, it becomes difficult to load when the struct has changed. Therefore MonteCarlo.jl implements a custom `_save` and `_load` function for most structs. In general these functions are implemented as

```julia
function _save(file::FileLike, key::String, object)
    write(file, "$key/VERSION", some_int)
    write(file, "$key/tag", "some identifiying name")
    # write data ...
end

function _load(data::FileData, ::Val{:some_tag})
    # maybe check data["VERSION"]
    # load file from data
end
```

where `FileData` is a wrapper around `Dict{String, Any}`, which allows accessing the dictionary returned by `JLD2.load` as if it were a nested dictionary. I.e. rather than `data["key1/key2/key3"]` you can use `data["key1"]["key2"]["key3"]` with it, which aligns better to the way data is saved and loaded. `FileLike` is a `Union{JLD2.JLDFile, JLD2.Group, FileData}`. The "tag" saved in each `_save` function is used to identify what is saved, and it later used for dispatch in `_load`.

The entrypoint for saving is the `save` function "src/FileIO.jl". The function includes functionality to rename a file to avoid overwriting an existing and includes compression. It also writes some GIT information which is printed in `load` if the load fails.