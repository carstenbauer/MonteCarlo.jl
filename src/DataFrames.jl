import .DataFrames

function build_default_getter(whitelist, blacklist, replace)
    function default_data(mc)
        d = Dict{Symbol, Any}()
        param = parameters(mc)
        for (k, v) in pairs(param)
            if !(k in blacklist) && (isempty(whitelist) || (k in whitelist))
                push!(d, replace(k => v))
            end
        end
        # push!(d, :mc => mc)
        d
    end
end

function DataFrames.DataFrame(mcs::Vector{Union{Nothing, T}}; kwargs...) where {T <: MonteCarloFlavor}
    clean = [mc for mc in mcs if mc !== nothing]
    DataFrame(clean; kwargs...)
end
function DataFrames.DataFrame(
        mcs::Vector{<: MonteCarloFlavor};
        whitelist = [], blacklist = [], replace = identity,
        getter = build_default_getter(whitelist, blacklist, replace),
        add_columns! = (dict, mc) -> nothing
    )

    param = mcs |> first |> getter
    add_columns!(param, first(mcs))
    df = param |> DataFrames.DataFrame

    for i in 2:length(mcs)
        data = mcs[i] |> getter
        add_columns!(data, mcs[i])
        push!(df, data)
    end

    df.mc = mcs

    df
end