# Implement struct T <: WrappedObservable with field obs and method
# (::T)(::Array) to apply its transformation and you get mean, etc
abstract type WrappedObservable end

# Wrappers for Statistics functions
MonteCarloObservable.mean(x::WrappedObservable) = x(mean(x.obs))
MonteCarloObservable.var(x::WrappedObservable)  = x(var(x.obs))
MonteCarloObservable.varN(x::WrappedObservable) = x(varN(x.obs))
MonteCarloObservable.std(x::WrappedObservable)  = x(std(x.obs))
MonteCarloObservable.std_error(x::WrappedObservable) = x(std_error(x.obs))
MonteCarloObservable.all_vars(x::WrappedObservable)  = x.(all_vars(x.obs))
MonteCarloObservable.all_varNs(x::WrappedObservable) = x.(all_varNs(x.obs))
# Autocorrelation time should not be averaged
MonteCarloObservable.tau(x::WrappedObservable) = maximum(tau(x.obs))
MonteCarloObservable.all_taus(x::WrappedObservable) = maximum.(all_varNs(x.obs))


################################################################################
### Occupation
################################################################################


struct Greens2Occupation{T <: AbstractObservable} <: WrappedObservable
    obs::T
end

occupations(m::GreensMeasurement) = Greens2Occupation(m.obs)
occupations(obs::AbstractObservable) = Greens2Occupation(obs)
(::Greens2Occupation)(M::Matrix) = diag(M)



################################################################################
### Uniform Fourier
################################################################################


"""
    uniform_fourier(M, dqmc)
    uniform_fourier(M, N)

Computes the uniform Fourier transform of matrix `M` in a system with `N` sites.
"""
uniform_fourier(M::AbstractArray, mc::DQMC) = sum(M) / nsites(mc.model)
uniform_fourier(M::AbstractArray, N::Integer) = sum(M) / N


struct UniformFourierWrapped{T <: AbstractObservable} <: WrappedObservable
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
uniform_fourier(m::AbstractMeasurement, field::Symbol) = UniformFourierWrapped(getfield(m, field))
uniform_fourier(obs::AbstractObservable) = UniformFourierWrapped(obs)
(::UniformFourierWrapped)(M::AbstractArray) = sum(M) / length(M)


################################################################################
### Structure Factor
################################################################################


struct StructureFactorWrapped{T <: AbstractObservable, VT <: Vector} <: WrappedObservable
    obs::T
    dirs::Vector{VT}
end

structure_factor(m, dqmc::DQMC, args...) = structure_factor(m, dqmc.model, args...)
structure_factor(m, model::Model, args...) = structure_factor(m, lattice(model), args...)

function structure_factor(m::AbstractMeasurement, lattice::AbstractLattice, args...)
    dirs = directions(mask(m), lattice)
    structure_factor(m, dirs, args...)
end
function structure_factor(m::AbstractMeasurement, directions::Vector{<: Vector}, field=:obs)
    structure_factor(getfield(m, field), directions)
end
function structure_factor(obs::AbstractObservable, directions::Vector{<: Vector})
    StructureFactorWrapped(obs, directions)
end
function (x::StructureFactorWrapped)(data)
    q -> mapreduce(+, eachindex(data)) do i
        exp(1im * dot(q, x.dirs[i])) * data[i]
    end / length(data)
end
