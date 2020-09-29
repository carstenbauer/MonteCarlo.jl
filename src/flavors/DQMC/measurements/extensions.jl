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
(::Greens2Occupation)(M::Matrix) = 1 .- diag(M)


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

See also: [`structure_factor`](@ref)
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

"""
    structure_factor(m, ::DQMC[, field])
    structure_factor(m, ::Model[, field])
    structure_factor(::AbstractMeasurement, ::Lattice[, field])
    structure_factor(m, directions[, field])

Returns a `StructureFactorWrapped` observable, given a Measurement or observable
`m`. Calling `mean` (etc) on the output will produce a function `O(q)` which
returns the `mean` (etc) of the given observable at a given momentum `q`.

E.g. `mean(structure_factor(obs, dqmc))([0, 0])` will return the expectation 
value of `obs` at `q=0`.

This is equivalent to a discrete Fourier transform.

See also: [`SymmetryWrapped`](@ref), [`swave`](@ref), [`eswave`](@ref)
"""
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
        cis(dot(q, x.dirs[i])) * data[i]
    end / length(data)
end

# Gaussian error propagation? ¯\_(ツ)_/¯
MonteCarloObservable.var(x::StructureFactorWrapped)  = sqrt(sum(var(x.obs) .^ 2))
MonteCarloObservable.varN(x::StructureFactorWrapped) = sqrt(sum(varN(x.obs) .^ 2))
MonteCarloObservable.std(x::StructureFactorWrapped)  = sqrt(sum(std(x.obs) .^ 2))
MonteCarloObservable.std_error(x::StructureFactorWrapped) = sqrt(sum(std_error(x.obs) .^ 2))


################################################################################
### Symmetry Wrapped
################################################################################


"""
    SymmetryWrapped(m, formfactor[, field=:obs])

A SymmetryWrapped observable will calculate
`sum(formfactor[i] * obs[i] for i in eachindex(formfactor))` where `obs[i]` is 
the value of the observable in the i-th direction. It may be constructed from
either an observable or measurement `m`, where the relevant field can be 
specified via `field` for the latter.
    
Quick constructors: [`swave`](@ref), [`eswave`](@ref)
See also: [`directions`](@ref), [`mask`](@ref)
"""
struct SymmetryWrapped{OT<:AbstractObservable, T} <: WrappedObservable
    obs::OT
    formfactor::Vector{T}
end

function SymmetryWrapped(m::AbstractMeasurement, formfactor, field=:obs)
    SymmetryWrapped(getfield(m, field), formfactor)
end

# TODO constructors?
# higher order is depended on lattice symmetry and therefore rather complicated
"""
    swave(m[, field=:obs])

Creates a SymmetryWrapped observable that computes the s-wave version of a 
measurement or observable `m`.

See also: [`eswave`](@ref), [`SymmetryWrapped`](@ref)
"""
swave(obs, args...) = SymmetryWrapped(obs, [1.0], args...)

"""
    eswave(measurement, lattice[, field=:obs])

Creates a SymmetryWrapped observable that computes the extended s-wave version 
of a measurement or observable `m`.

See also: [`swave`](@ref), [`SymmetryWrapped`](@ref)
"""
function eswave(m::AbstractMeasurement, lattice::AbstractLattice, field=:obs)
    _mask = mask(m)
    dirs = directions(_mask, lattice)
    j = 2; l = dot(dirs[j], dirs[j])
    while (dot(dirs[j+1], dirs[j+1]) < l + 1e-3) && (j < length(dirs))
        j += 1
    end
    SymmetryWrapped(m, vcat(0, ones(j-1)), field)
end

function (x::SymmetryWrapped)(data)
    sum(x.formfactor[i] * data[i] for i in eachindex(x.formfactor)) / length(x.formfactor)
end

# Gaussian error propagation? ¯\_(ツ)_/¯
MonteCarloObservable.var(x::SymmetryWrapped)  = sqrt(sum(var(x.obs) .^ 2))
MonteCarloObservable.varN(x::SymmetryWrapped) = sqrt(sum(varN(x.obs) .^ 2))
MonteCarloObservable.std(x::SymmetryWrapped)  = sqrt(sum(std(x.obs) .^ 2))
MonteCarloObservable.std_error(x::SymmetryWrapped) = sqrt(sum(std_error(x.obs) .^ 2))

