function default_measurements(mc::DQMC, model)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end



################################################################################
### General DQMC Measurements
################################################################################



"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.

The mean of this measurement corresponds to the expectation value of the Greens
function for the full partition function, i.e. including fermionic and bosonic
(auxiliary field) degrees of freedom.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model)
    o = LightObservable(
        LogBinner(zeros(eltype(mc.s.greens), size(mc.s.greens))),
        "Equal-times Green's function",
        "Observables.jld",
        "G"
    )
    GreensMeasurement{typeof(o)}(o)
end
function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end



"""
    BosonEnergyMeasurement(mc::DQMC, model)

Measures the bosnic energy of the given DQMC simulation and model.

Note that this measurement requires `energy_boson(mc, model, conf)` to be
implemented for the specific `model`.
"""
struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function BosonEnergyMeasurement(mc::DQMC, model)
    o = LightObservable(Float64, name="Bosonic Energy", alloc=1_000_000)
    BosonEnergyMeasurement{typeof(o)}(o)
end
function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end



################################################################################
### Spin 1/2 Measurements
################################################################################



abstract type SpinOneHalfMeasurement <: AbstractMeasurement end

# Abuse prepare! to verify requirements
function prepare!(m::SpinOneHalfMeasurement, mc::DQMC, model)
    model.flv != 2 && throw(AssertionError(
        "A spin 1/2 measurement ($(typeof(m))) requires two (spin) flavors of fermions, but " *
        "the given model has $(model.flv)."
    ))
end


#
function get_lattice_shape(model::Model)
    try
        return size(model.l)
    catch e
        @warn "Failed to get lattice shape. Using flat array of sites instead."
        @warn e
        return nsites(model)
    end
end



@doc raw"""
    ChargeDensityCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the charge density correlation
matrix `⟨nᵢnⱼ⟩`.

The mean of this measurement corresponds to the expectation value of the charge
density correlation matrix for the full partition function, i.e. including
fermionic and bosonic (auxiliary field) degrees of freedom.


The fermionic expectation value is computed via Wick's theorem.
```math
      \langle n_i n_j \rangle
    = \langle (n_{i, \uparrow} + n_{i, \downarrow}) (n_{j, \uparrow} +
    n_{j, \downarrow})\rangle
    = \langle n_{i, \uparrow} n_{j, \uparrow} \rangle +
      \langle n_{i, \uparrow} n_{j, \downarrow} \rangle +
      \langle n_{i, \downarrow} n_{j, \uparrow} \rangle +
      \langle n_{i, \downarrow} n_{j, \downarrow} \rangle
    = ... + \langle c_{i, \uparrow}^\dagger c_{i, \uparrow}
        c_{j, \downarrow}^\dagger c_{j, \downarrow} \rangle + ...
    = ... + \langle c_{i, \uparrow}^\dagger c_{i, \uparrow} \rangle
      \langle c_{j, \downarrow}^\dagger c_{j, \downarrow} \rangle +
      \langle c_{i, \uparrow}^\dagger c_{j, \downarrow} \rangle
      \langle c_{i, \uparrow} c_{j, \downarrow}^\dagger \rangle + ...
    = ... + (I - G)_{ii}^{\uparrow\uparrow} (I - G)_{jj}^{\downarrow\downarrow} +
      (I-G)_{ji}^{\downarrow\uparrow} G_{ij}^{\uparrow\downarrow} + ...
```
"""
struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
end
function ChargeDensityCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    ChargeDensityCorrelationMeasurement(obs, reshape([zero(T) for _ in 1:N], shape))
end
function measure!(m::ChargeDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G
    m.temp .= zero(eltype(m.temp))

    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += begin
                # ⟨n↑n↑⟩
                IG[i, i] * IG[j, j] +
                IG[j, i] *  G[i, j] +
                # ⟨n↑n↓⟩
                IG[i, i] * IG[j + N, j + N] +
                IG[j + N, i] *  G[i, j + N] +
                # ⟨n↓n↑⟩
                IG[i + N, i + N] * IG[j, j] +
                IG[j, i + N] *  G[i + N, j] +
                # ⟨n↓n↓⟩
                IG[i + N, i + N] * IG[j + N, j + N] +
                IG[j + N, i + N] *  G[i + N, j + N]
            end
        end
    end

    push!(m.obs, m.temp / N)
end



"""
    MagnetizationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the magnetization
`M_x = ⟨c_{i, ↑}^† c_{i, ↓} + h.c.⟩` in x-,
`M_y = -i⟨c_{i, ↑}^† c_{i, ↓} - h.c.⟩` in y- and
`M_z = ⟨n_{i, ↑} - n_{i, ↓}⟩` in z-direction.

The mean of this measurement corresponds to the expectation value of the x/y/z
magnetization for the full partition function, i.e. including fermionic and
bosonic (auxiliary field) degrees of freedom.

Note:

The Magnetization in x/y/z direction can be accessed via fields `x`, `y` and `z`.
"""
struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: AbstractArray
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
end
function MagnetizationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Magnetizations
    m1x = LightObservable(
        LogBinner([zero(T) for _ in 1:N]),
        "Magnetization x", "Observables.jld", "Mx"
    )
    m1y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N]),
        "Magnetization y", "Observables.jld", "My"
    )
    m1z = LightObservable(
        LogBinner([zero(T) for _ in 1:N]),
        "Magnetization z", "Observables.jld", "Mz"
    )

    MagnetizationMeasurement(m1x, m1y, m1z, [zero(T) for _ in 1:N])
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # Magnetization
    # c_{i, up}^† c_{i, down} + c_{i, down}^† c_{i, up}
    # mx = [- G[i+N, i] - G[i, i+N]           for i in 1:N]
    map!(i -> -G[i+N, i] - G[i, i+N], m.temp, 1:N)
    push!(m.x, m.temp)

    # -i [c_{i, up}^† c_{i, down} - c_{i, down}^† c_{i, up}]
    # my = [-1im * (G[i, i+N] - G[i+N, i])    for i in 1:N]
    map!(i -> -1im *(G[i+N, i] - G[i, i+N]), m.temp, 1:N)
    push!(m.y, m.temp)
    # c_{i, up}^† c_{i, up} - c_{i, down}^† c_{i, down}
    # mz = [G[i+N, i+N] - G[i, i]             for i in 1:N]
    map!(i -> G[i+N, i+N] - G[i, i], m.temp, 1:N)
    push!(m.z, m.temp)
end



"""
    SpinDensityCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the spin density correlation matrix
`SDC_x = ⟨(c_{i, ↑}^† c_{i, ↓} + h.c.) (c_{j, ↑}^† c_{j, ↓} + h.c.)⟩` in x-,
`SDC_y = -⟨(c_{i, ↑}^† c_{i, ↓} - h.c.) (c_{j, ↑}^† c_{j, ↓} - h.c.)⟩` in y- and
`SDC_z = ⟨(n_{i, ↑} - n_{i, ↓}) (n_{j, ↑} - n_{j, ↓})⟩` in z-direction.

The mean of this measurement corresponds to the expectation value of the x/y/z
spin density correlation matrix for the full partition function, i.e. including
fermionic and bosonic (auxiliary field) degrees of freedom.

Note:

The spin density correlation matrix in x/y/z direction can be accessed via fields `x`,
`y` and `z`.
"""
struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
end
function SpinDensityCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    sdc2x = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Spin Density Correlation x", "Observables.jld", "sdc-x"
    )
    sdc2y = LightObservable(
        LogBinner(reshape([zero(Ty) for _ in 1:N], shape)),
        "Spin Density Correlation y", "Observables.jld", "sdc-y"
    )
    sdc2z = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Spin Density Correlation z", "Observables.jld", "sdc-z"
    )

    SpinDensityCorrelationMeasurement(sdc2x, sdc2y, sdc2z, reshape([zero(T) for _ in 1:N], shape))
end
function measure!(m::SpinDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†


    # Spin Density Correlation
    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                IG[i+N, i] * IG[j+N, j] + IG[j+N, i] * G[i+N, j] +
                IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
                IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] +
                IG[i, i+N] * IG[j, j+N] + IG[j, i+N] * G[i, j+N]
            )
        end
    end
    push!(m.x, m.temp / N)

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                - IG[i+N, i] * IG[j+N, j] - IG[j+N, i] * G[i+N, j] +
                  IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
                  IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] -
                  IG[i, i+N] * IG[j, j+N] - IG[j, i+N] * G[i, j+N]
            )
        end
    end
    push!(m.y, m.temp / N)

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                IG[i, i] * IG[j, j] + IG[j, i] * G[i, j] -
                IG[i, i] * IG[j+N, j+N] - IG[j+N, i] * G[i, j+N] -
                IG[i+N, i+N] * IG[j, j] - IG[j, i+N] * G[i+N, j] +
                IG[i+N, i+N] * IG[j+N, j+N] + IG[j+N, i+N] * G[i+N, j+N]
            )
        end
    end
    push!(m.z, m.temp / N)
end



"""
    PairingCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the s-wave pairing correlation.

We define `Δᵢ = c_{i, ↑} c_{i, ↓}` s the pair-field operator and `Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩`
as the s-wave pairing correlation matrix. `Pᵢⱼ` can be accesed via the field
`mat` and its site-average via the field `uniform_fourier`.
"""
struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
end
function PairingCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    T = eltype(mc.s.greens)
    N = nsites(model)

    obs1 = LightObservable(
        LogBinner(reshape(zeros(T, N), shape)),
        "Equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s"
    )

    PairingCorrelationMeasurement(obs1, reshape(zeros(T, N), shape))
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc, model)
    N = nsites(model)
    # Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩
    #     = ⟨c_{i, ↑} c_{i, ↓} c_{j, ↓}^† c_{j, ↑}^†⟩
    #     = ⟨c_{i, ↑} c_{j, ↑}^†⟩ ⟨c_{i, ↓} c_{j, ↓}^†⟩ -
    #       ⟨c_{i, ↑} c_{j, ↓}^†⟩ ⟨c_{i, ↓} c_{j, ↑}^†⟩
    # m.temp .= G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N]

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += G[i, j] * G[i+N, j+N] - G[i, j+N] * G[i+N, j]
        end
    end

    push!(m.obs, m.temp / N)
end

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
