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
    return accepted
end


"""
    sweep_spatial(mc::DQMC)

Performs a sweep of local moves along spatial dimension at the current
imaginary time slice.
"""
@bm function sweep_spatial(mc::DQMC, m)
    N = size(conf(field(mc)), 1)

    accepted = 0
    @inbounds for i in 1:N
        #i = rand(1:N)
        detratio, ΔE_boson, passthrough = propose_local(mc, m, field(mc), i, current_slice(mc))

        p = exp(- ΔE_boson) * detratio

        if mc.parameters.check_sign_problem
            if abs(imag(p)) > 1e-6
                push!(mc.analysis.imaginary_probability, abs(imag(p)))
                mc.parameters.silent || @printf(
                    "Did you expect a sign problem? imag. probability:  %.9e\n", 
                    abs(imag(p))
                )
            end
            if real(p) < 0.0
                push!(mc.analysis.negative_probability, real(p))
                mc.parameters.silent || @printf(
                    "Did you expect a sign problem? negative probability %.9e\n",
                    real(p)
                )
            end
        end

        # Gibbs/Heat bath
        # p = p / (1.0 + p)
        # Metropolis
        if real(p) > 1 || rand() < real(p)
            accept_local!(mc, m, field(mc), i, current_slice(mc), detratio, ΔE_boson, passthrough)
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
@bm update(::LocalSweep, mc::DQMC, model, field) = local_sweep(mc, model) / 2length(field)
name(::LocalSweep) = "LocalSweep"
_load(::FileLike, ::Val{:LocalSweep}) = LocalSweep()