"""
    uniform_fourier(M, dqmc)
    uniform_fourier(M, N)

Computes the uniform Fourier transform of matrix `M` in a system with `N` sites.
"""
uniform_fourier(M::AbstractArray, mc::DQMC) = sum(M) / nsites(mc.model)
uniform_fourier(M::AbstractArray, N::Integer) = sum(M) / N


struct UniformFourierWrapped{T <: AbstractObservable}
    obs::T
end
"""
    uniform_fourier(m::AbstractMeasurement[, field::Symbol])
    uniform_fourier(obs::AbstractObservable)

Wraps an observable with a `UniformFourierWrapped`.
Calling `mean` (`var`, etc) on a wrapped observable returns the `mean` (`var`,
etc) of the uniform Fourier transform of that observable.

`mean(uniform_fourier(m))` is equivalent to
`uniform_fourier(mean(m.obs), nsites(model))` where `obs` may differ between
measurements.
"""
uniform_fourier(m::PairingCorrelationMeasurement) = UniformFourierWrapped(m.obs)
uniform_fourier(m::ChargeDensityCorrelationMeasurement) = UniformFourierWrapped(m.obs)
function uniform_fourier(m::AbstractMeasurement, field::Symbol)
    UniformFourierWrapped(getfield(m, field))
end
uniform_fourier(obs::AbstractObservable) = UniformFourierWrapped(obs)

# Wrappers for Statistics functions
MonteCarloObservable.mean(x::UniformFourierWrapped) = _uniform_fourier(mean(x.obs))
MonteCarloObservable.var(x::UniformFourierWrapped) = _uniform_fourier(var(x.obs))
MonteCarloObservable.varN(x::UniformFourierWrapped) = _uniform_fourier(varN(x.obs))
MonteCarloObservable.std(x::UniformFourierWrapped) = _uniform_fourier(std(x.obs))
MonteCarloObservable.std_error(x::UniformFourierWrapped) = _uniform_fourier(std_error(x.obs))
MonteCarloObservable.all_vars(x::UniformFourierWrapped) = _uniform_fourier.(all_vars(x.obs))
MonteCarloObservable.all_varNs(x::UniformFourierWrapped) = _uniform_fourier.(all_varNs(x.obs))
# Autocorrelation time should not be averaged...
MonteCarloObservable.tau(x::UniformFourierWrapped) = maximum(tau(x.obs))
MonteCarloObservable.all_taus(x::UniformFourierWrapped) = maximum.(all_varNs(x.obs))
_uniform_fourier(M::AbstractArray) = sum(M) / length(M)





# mean(StructureFactor(meas, model))(q)





struct StructureFactorWrapped{T <: AbstractObservable, VT <: Vector}
    obs::T
    dirs::Vector{VT}
end

structure_factor(m, dqmc::DQMC, args...) = structure_factor(m, dqmc.model, args...)
structure_factor(m, model::Model, args...) = structure_factor(m, lattice(model), args...)

function structure_factor(m, lattice::AbstractLattice, args...)
    pos = positions(lattice)
    dirs = [pos[1] .- p for p in pos[m.mask[1, :]]]
    structure_factor(m, dirs, args...)
end
function structure_factor(m::AbstractMeasurement, directions::Vector{<: Vector}, field=:obs)
    structure_factor(getfield(m, field), directions)
end
function structure_factor(obs::AbstractObservable, directions::Vector{<: Vector})
    StructureFactorWrapped(obs, directions)
end


# Wrappers for Statistics functions
MonteCarloObservable.mean(x::StructureFactorWrapped) = _structure_factor(x.dirs, mean(x.obs))
MonteCarloObservable.var(x::StructureFactorWrapped) = _structure_factor(x.dirs, var(x.obs))
MonteCarloObservable.varN(x::StructureFactorWrapped) = _structure_factor(x.dirs, varN(x.obs))
MonteCarloObservable.std(x::StructureFactorWrapped) = _structure_factor(x.dirs, std(x.obs))
MonteCarloObservable.std_error(x::StructureFactorWrapped) = _structure_factor(x.dirs, std_error(x.obs))
MonteCarloObservable.all_vars(x::StructureFactorWrapped) = [_structure_factor(x.dirs, data) for data in all_vars(x.obs)]
MonteCarloObservable.all_varNs(x::StructureFactorWrapped) = [_structure_factor(x.dirs, data) for data in all_varNs(x.obs)]

function _structure_factor(dirs, data)
    q -> mapreduce(+, eachindex(data)) do i
        exp(1im * dot(q, dirs[i])) * data[i]
    end / length(data)
end
