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


"""
    add_default_measurements!(mc; kwargs...)

Adds default measurements to a DQMC simulation. Keyword arguments can be used 
to disable (`key = false`), enable (`key = true`) or replace 
(`key = measurement`) each default measurement.

## The keys include:
- `:occ` for `occupation(mc, mc.model))` (skipped if :G not disabled)
- `:Mx` for `magnetization(mc, mc.model, :x))` (skipped if :G not disabled)
- `:My` for `magnetization(mc, mc.model, :y))` (skipped if :G not disabled)
- `:Mz` for `magnetization(mc, mc.model, :z))` (skipped if :G not disabled)
- `:G` for `greens_measurement(mc, mc.model))`
- `:K` for `kinetic_energy(mc, mc.model))`
- `:V` for `interaction_energy(mc, mc.model))`
- `:E` for `total_energy(mc, mc.model))` (skipped if neither :K or :V are disabled)

- `:CDC` for `charge_density_correlation(mc, mc.model))`
- `:PC` for `pairing_correlation(mc, mc.model))`
- `:SDCx` for `spin_density_correlation(mc, mc.model, :x))`
- `:SDCy` for `spin_density_correlation(mc, mc.model, :y))`
- `:SDCz` for `spin_density_correlation(mc, mc.model, :z))`

- `:CDS` for `charge_density_susceptibility(mc, mc.model))`
- `:PS` for `pairing_susceptibility(mc, mc.model))`
- `:SDSx` for `spin_density_susceptibility(mc, mc.model, :x))`
- `:SDSy` for `spin_density_susceptibility(mc, mc.model, :y))`
- `:SDSz` for `spin_density_susceptibility(mc, mc.model, :z))`
"""
function add_default_measurements!(mc; kwargs...)
    kwarg_dict = Dict{Symbol, Any}(kwargs)

    # auto-disable E when K and V are recorded
    if (get(kwarg_dict, :K, true) != false) && (get(kwarg_dict, :V, true) != false)
        get!(kwarg_dict, :E, false)
    end

    # auto-disable occ, Mx, My, Mz when G is recorded
    if get(kwarg_dict, :G, true) != false
        get!(kwarg_dict, :occ, false)
        get!(kwarg_dict, :Mx, false)
        get!(kwarg_dict, :My, false)
        get!(kwarg_dict, :Mz, false)
    end

    println(kwarg_dict)

    # remove true to allow autocompleting them
    filter!(kv -> kv[2] != true, kwarg_dict)

    # autocomplete measurements
    get!(kwarg_dict, :occ, occupation(mc, mc.model))
    get!(kwarg_dict, :Mx, magnetization(mc, mc.model, :x))
    get!(kwarg_dict, :My, magnetization(mc, mc.model, :y))
    get!(kwarg_dict, :Mz, magnetization(mc, mc.model, :z))
    get!(kwarg_dict, :G, greens_measurement(mc, mc.model))
    get!(kwarg_dict, :K, kinetic_energy(mc, mc.model))
    get!(kwarg_dict, :V, interaction_energy(mc, mc.model))
    get!(kwarg_dict, :E, total_energy(mc, mc.model))

    get!(kwarg_dict, :CDC, charge_density_correlation(mc, mc.model))
    get!(kwarg_dict, :PC, pairing_correlation(mc, mc.model))
    get!(kwarg_dict, :SDCx, spin_density_correlation(mc, mc.model, :x))
    get!(kwarg_dict, :SDCy, spin_density_correlation(mc, mc.model, :y))
    get!(kwarg_dict, :SDCz, spin_density_correlation(mc, mc.model, :z))

    get!(kwarg_dict, :CDS, charge_density_susceptibility(mc, mc.model))
    get!(kwarg_dict, :PS, pairing_susceptibility(mc, mc.model))
    get!(kwarg_dict, :SDSx, spin_density_susceptibility(mc, mc.model, :x))
    get!(kwarg_dict, :SDSy, spin_density_susceptibility(mc, mc.model, :y))
    get!(kwarg_dict, :SDSz, spin_density_susceptibility(mc, mc.model, :z))
    get!(kwarg_dict, :CCS, current_current_susceptibility(mc, mc.model))

    # remove all invalid entries
    filter!(kv -> kv[2] isa AbstractMeasurement, kwarg_dict)

    for (k, v) in kwarg_dict
        mc[k] = v
    end

    return mc
end