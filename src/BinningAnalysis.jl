# FileIO for BinningAnalysis to avoid saving custom structs directly

################################################################################
### LogBinner
################################################################################

function _save(file::FileLike, key::String, obs::LogBinner{ET}) where ET
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", "LogBinner")
    write(file, "$key/cache", ET[c.value for c in obs.compressors]) 
    write(file, "$key/state", Bool[c.switch for c in obs.compressors])
    _save(file, "$key/accumulators", obs.accumulators)
    return
end

function _load(data, ::Val{:LogBinner})
    comp = Tuple(BinningAnalysis.Compressor.(data["cache"], data["state"]))
    acc =  _load(data["accumulators"])
    return LogBinner{typeof(first(comp).value), length(comp)}(comp, acc)
end

function _save(
        file::FileLike, key::String, accumulators::NTuple{N, BinningAnalysis.Variance{ET}}
    ) where {N, ET}
    write(file, "$key/tag", "VarAccum")
    write(file, "$key/delta", ET[a.δ for a in accumulators])
    write(file, "$key/m1", ET[a.m1 for a in accumulators])
    write(file, "$key/m2", ET[a.m2 for a in accumulators])
    write(file, "$key/count", Int[a.count for a in accumulators])
    return
end

function _load(data, ::Val{:VarAccum})
    Tuple(BinningAnalysis.Variance.(
        data["delta"], data["m1"], data["m2"], data["count"] 
    ))
end

function _save(
        file::FileLike, key::String, accumulators::NTuple{N, BinningAnalysis.FastVariance{ET}}
    ) where {N, ET}
    write(file, "$key/tag", "FastVarAccum")
    write(file, "$key/x_sum", ET[a.x_sum for a in accumulators])
    write(file, "$key/x2_sum", ET[a.x2_sum for a in accumulators])
    write(file, "$key/count", Int[a.count for a in accumulators])
    return
end

function _load(data, ::Val{:FastVarAccum})
    Tuple(BinningAnalysis.FastVariance.(
        data["x_sum"], data["x2_sum"],data["count"]
    ))
end


################################################################################
### FullBinner
################################################################################


function _save(file::FileLike, key::String, obs::FullBinner)
    write(file, "$key/VERSION", 1)
    write(file, "$key/tag", "FullBinner")
    write(file, "$key/data", obs.x)
    return
end

function _load(data, ::Val{:FullBinner})
    FullBinner(data["data"])
end
