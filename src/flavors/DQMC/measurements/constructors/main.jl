const _GM{MT} = GreensMatrix{<: Number, MT}
const _GM2{MT} = NTuple{2, GreensMatrix{<: Number, MT}}
const _GM4{MT} = NTuple{4, GreensMatrix{<: Number, MT}}

function _flavor_rec(maxflv, dims)
    if dims < 2
        return 1:maxflv
    else
        return [(t..., i) for t in _flavor_rec(maxflv, dims-1) for i in 1:maxflv]
    end
end

"""
    FlavorIterator(mc, dims)

Generates a set of flavor indices for the given simulation.

- `dims = 0` is interpreted as a constant flavor. In this case the iterator 
returns just `total_flavors(mc.model)`
- `dims = 1` is interpreted as `1:unique_flavors(mc.model)`
- `dims > 1` generates a Vector of Tuples of size `dims`, where each element in
the tuple runs from 1 to `unique_flavors(mc.model)`. This effectively generates
`dims` nested loops running through `1:unique_flavors(mc.model)`.
"""
FlavorIterator(mc, dims::Int) = FlavorIterator(mc, mc.model, dims)
function FlavorIterator(mc, m::Model, dims::Int)
    if dims == 0
        # if we don't iterate the unreduzed number of flavors might be useful...
        return total_flavors(m)
    elseif dims > 0
        return _flavor_rec(unique_flavors(mc), dims)
    else
        error("")
    end
end

include("charge_density.jl")
include("current_current.jl")
include("energy.jl")
include("greens.jl")
include("magnetization.jl")
include("occupation.jl")
include("pairing.jl")
include("spin_density.jl")

# iterate 2
# ccs, cdc
# iterate 1
# occs, 
# explicit mix
# pc, mx, my, mz, sdcx, sdcy, sdcz

# that's sadder than I though
# maybe I should add FlavorIterator?



# yea do it. don't need to be anything facy, just
# fi = 1
# fi = 1:2
# fi = _flavor2x2

# I think I need to pick these based on unique_flavors too... 
# which FlavorIterator now does
# maybe it should produce `Val(flv)` for dims = 0?
# That way we can still use ::Val
# though maybe having it return `maxflv::Int` and using `if`` is better since 
# branch prediction does a good job?