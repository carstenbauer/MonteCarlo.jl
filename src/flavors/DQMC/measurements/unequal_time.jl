abstract type UnequalTimeMeasurement <: AbstractMeasurement end

struct UTGreensMeasurement{OT<:AbstractObservable} <: UnequalTimeMeasurement
    obs::OT
    l1::Int64
    l2::Int64
end

"""
    UTGreensMeasurement(mc, model[; capacity, τ1, τ2, slice1 = τ1/Δτ, slice2 = τ2/Δτ])

G(τ2 ← τ1)
"""
function UTGreensMeasurement(
        mc::DQMC, model; 
        capacity=_default_capacity(mc),
        τ1 = 0.5mc.p.beta, slice1 = τ1 / mc.p.delta_tau,
        τ2 = 0.0, slice2 = τ2 / mc.p.delta_tau
    )
    l1 = round(Int64, slice1)
    l2 = round(Int64, slice2)
    abs(l1 - slice1) > 1e-6 && throw(ArgumentError(
        "slice1 = τ1 / mc.p.delta_tau should be an integer, but is $slice1."
    ))
    abs(l2 - slice2) > 1e-6 && throw(ArgumentError(
        "slice2 = τ2 / mc.p.delta_tau should be an integer, but is $slice2."
    ))
    

    T = greenseltype(DQMC, model)
    N = model.flv * length(lattice(model))
    o = LightObservable(
        LogBinner(zeros(T, (N, N)), capacity=capacity),
        "Unequal-times Green's function",
        "Observables.jld",
        "G"
    )
    UTGreensMeasurement{typeof(o)}(o, l1, l2)
end
@bm function measure!(m::UTGreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens!(mc, m.l2, m.l1))
end


# TODO
# - figure out how to intialize UnequalTimeStack nicely
#   - needs to work from `resume!(filename)`, from `DQMC()` and `push!(dqmc, m)`
# - test DQMC stack
# - add unequal time Measurements
# - test those too
