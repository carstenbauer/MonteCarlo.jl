# general functions that apply to all abstract types
"""
	observables(m::Model)

Get all observables defined for a given model.

Returns a `Dict{String, String}` where values are the observables names and
keys are short versions of those names. The keys can be used to
collect correponding observable objects from a Monte Carlo simulation, e.g. like `mc.obs[key]`.
"""
function observables(m::Model)
	obs = Dict{String, String}()
	obsobjects = prepare_observables(m)
	for (s, o) in obsobjects
		obs[s] = name(o)
	end
	return obs
end