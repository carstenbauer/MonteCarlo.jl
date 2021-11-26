import .DataFrames

function build_default_getter(whitelist, blacklist)
    function default_data(mc)
        d = Dict{Symbol, Any}()
        param = parameters(mc)
        for (k, v) in pairs(param)
            if !(k in blacklist) && (isempty(whitelist) || (k in whitelist))
                push!(d, k => v)
            end
        end
        push!(d, :mc => mc)
        d
    end
end

function DataFrames.DataFrame(
        mcs::Vector{<: MonteCarloFlavor};
        whitelist = [], blacklist = [],
        getter = build_default_getter(whitelist, blacklist)
    )

    df = mcs |> first |> getter |> DataFrames.DataFrame

    for i in 2:length(mcs)
        data = mcs[i] |> getter
        push!(df, data)
    end

    df
end