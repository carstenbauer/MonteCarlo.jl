"""
    local_sweep(mc::DQMC)

Performs a local update sweep for the given simulation. This includes a sweep
through all time slices twice (up and down) and all sites for each time slice.
"""
function local_sweep(mc::DQMC, model)
    accepted = 0
    for _ in 1:2nslices(mc)
        accepted += sweep_spatial(mc, model)
        propagate(mc)
    end
    mc.last_sweep += 1
    return accepted
end


"""
    sweep_spatial(mc::DQMC)

Performs a sweep of local moves along spatial dimension at the current
imaginary time slice.
"""
@bm function sweep_spatial(mc::DQMC, m)
    N = size(conf(mc), 1)

    # @inbounds for i in rand(1:N, N)
    accepted = 0
    @inbounds for i in 1:N
        detratio, ΔE_boson, passthrough = propose_local(mc, m, i, current_slice(mc), conf(mc))

        if mc.parameters.check_sign_problem
            if abs(imag(detratio)) > 1e-6
                push!(mc.analysis.imaginary_probability, abs(imag(detratio)))
                mc.parameters.silent || @printf(
                    "Did you expect a sign problem? imag. detratio:  %.9e\n", 
                    abs(imag(detratio))
                )
            end
            if real(detratio) < 0.0
                push!(mc.analysis.negative_probability, real(detratio))
                mc.parameters.silent || @printf(
                    "Did you expect a sign problem? negative detratio %.9e\n",
                    real(detratio)
                )
            end
        end
        p = real(exp(- ΔE_boson) * detratio)

        # Gibbs/Heat bath
        # p = p / (1.0 + p)
        # Metropolis
        if p > 1 || rand() < p
            accept_local!(mc, m, i, current_slice(mc), conf(mc), detratio, ΔE_boson, passthrough)
            accepted += 1
        end
    end

    return accepted
end



################################################################################
### Local Update (sweeps)
################################################################################



struct LocalSweep <: AbstractLocalUpdate end

"""
    LocalSweep([mc, model], [N = 1])

Performs `N` sweeps of local updates and increments the sweep counter.

For `DQMC` a sweep is defined as `N ⋅ 2M` spin flips, where `N` is the number of
sites and `M` is the number of time slices.
"""
LocalSweep(mc, model, N=1) = N == 1 ? LocalSweep() : LocalSweep(N)
LocalSweep(N) = [LocalSweep() for _ in 1:N]
@bm update(::LocalSweep, mc::DQMC, model) = local_sweep(mc, model) / 2length(conf(mc))
name(::LocalSweep) = "LocalSweep"