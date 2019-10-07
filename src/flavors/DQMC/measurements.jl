function default_measurements(mc::DQMC, model)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end



################################################################################
### Utilities
################################################################################


# NOTE: Likely broken
# """
#     greens(mc::DQMC, slice1, slice2)
#
# Computes the unequal time Green's function G(τ₁, τ₂) where τ₁ ≥ τ₂.
# """
# function greens(mc::DQMC, slice1::Int64, slice2::Int64)
#     slice1 == slice2 && return greens(mc, slice1)
#     N = nslices(mc)
#
#     @assert slice1 > slice2
#
#     G = greens(mc, slice2+1)
#     copyto!(mc.s.Ul, I)
#     mc.s.Dl .= 1
#     copyto!(mc.s.Tl, I)
#
#     for i in slice2+1:slice1
#         multiply_slice_matrix_left!(mc, mc.model, i, mc.s.Ul)
#         if i % mc.p.safe_mult == 0
#             rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
#             mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
#             mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
#             copyto!(mc.s.Ul, mc.s.U)
#             copyto!(mc.s.Tl, mc.s.tmp)
#         end
#     end
#     # Finalize product Ul Dl Tl G
#     rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
#     rmul!(mc.s.Ul, mc.s.Tl)
#     lmul!(mc.s.Ul, G)
#     return G
# end


"""
    greens(mc::DQMC, slice)

Computes the equal-time Greens function G(τ) = G(τ, τ). `slice = 1` represents
τ = Δτ, `slice = nslices(mc)` τ = β = 0.
"""
function greens(mc::DQMC, slice::Int64)
    cur_slice = current_slice(mc)
    N = nslices(mc)
    cur_slice == slice && return greens(mc)

    # use slice matrix multiplications if it's within safe_mult boundaries
    d = let
        x = slice - cur_slice
        y = N - abs(x)
        abs(x) < y ? x : sign(-x) * y
    end
    if abs(d) < div(mc.p.safe_mult, 2)
        mc.s.greens_temp .= mc.s.greens
        if d > 0 # forward in time
            for s in cur_slice:cur_slice+d-1
                multiply_slice_matrix_left!(
                    mc, mc.model, mod1(s, N), mc.s.greens_temp
                )
                multiply_slice_matrix_inv_right!(
                    mc, mc.model, mod1(s, N), mc.s.greens_temp
                )
            end
        else # backward in time
            for s in cur_slice-1:-1:cur_slice+d
                multiply_slice_matrix_inv_left!(
                    mc, mc.model, mod1(s, N), mc.s.greens_temp
                )
                multiply_slice_matrix_right!(
                    mc, mc.model, mod1(s, N), mc.s.greens_temp
                )
            end
        end
        return _greens!(mc, mc.s.greens_temp)
    end

    # Otherwise we need to explicitly recalculate stuff
    # We use these as udt "stack"
    copyto!(mc.s.Ul, I)
    mc.s.Dl .= 1
    copyto!(mc.s.Tl, I)
    copyto!(mc.s.Ur, I)
    mc.s.Dr .= 1
    copyto!(mc.s.Tr, I)

    for i in 1:slice-1
        multiply_slice_matrix_left!(mc, mc.model, i, mc.s.Ul)
        if i % mc.p.safe_mult == 0
            rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
            mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
            mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
            copyto!(mc.s.Ul, mc.s.U)
            copyto!(mc.s.Tl, mc.s.tmp)
        end
    end
    # Finalize product and UDT decomposition
    rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
    mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
    mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
    copyto!(mc.s.Ul, mc.s.U)
    copyto!(mc.s.Tl, mc.s.tmp)

    for i in N:-1:slice
        multiply_daggered_slice_matrix_left!(mc, mc.model, i, mc.s.Ur)
        if i % mc.p.safe_mult == 0
            rmul!(mc.s.Ur, Diagonal(mc.s.Dr))
            mc.s.U, mc.s.Dr, mc.s.T = udt(mc.s.Ur)
            mul!(mc.s.tmp, mc.s.T, mc.s.Tr)
            copyto!(mc.s.Ur, mc.s.U)
            copyto!(mc.s.Tr, mc.s.tmp)
        end
    end
    rmul!(mc.s.Ur, Diagonal(mc.s.Dr))
    mc.s.U, mc.s.Dr, mc.s.T = udt(mc.s.Ur)
    mul!(mc.s.tmp, mc.s.T, mc.s.Tr)
    copyto!(mc.s.Ur, mc.s.U)
    copyto!(mc.s.Tr, mc.s.tmp)

    # slightly modified calculate_greens()
    mc.s.U, mc.s.D, mc.s.T = udt_inv_one_plus(
        UDT(mc.s.Ul, mc.s.Dl, mc.s.Tl),
        UDT(mc.s.Ur, mc.s.Dr, mc.s.Tr),
        tmp = mc.s.U, tmp2 = mc.s.T, tmp3 = mc.s.tmp,
        internaluse = true
    )
    mul!(mc.s.tmp, mc.s.U, Diagonal(mc.s.D))
    mul!(mc.s.greens_temp, mc.s.tmp, mc.s.T)
    return _greens!(mc, mc.s.greens_temp)
end



################################################################################
### General DQMC Measurements
################################################################################



"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.
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



"""
    ChargeDensityCorrelationMeasurement(mc::DQMC, model)

Measures the charge density correlation matrix `⟨nᵢnⱼ⟩`.
"""
struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable
    } <: AbstractMeasurement
    obs::OT
    temp::Matrix
end
function ChargeDensityCorrelationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    ChargeDensityCorrelationMeasurement(obs, [zero(T) for _ in 1:N, __ in 1:N])
end
function measure!(m::ChargeDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    # TODO
    # implement spinflavors(model)
    # then get N from size(model.l) / spinflavors(model) ?
    N = nsites(model)
    flv = model.flv
    G = greens(mc)
    IG = I - G
    m.temp .= zero(eltype(m.temp))
    for f1 in 0:flv-1, f2 in 0:flv-1
        for i in 1:N, j in 1:N
            m.temp[i, j] += IG[i + f1*N, i + f1*N] * IG[j + f2*N, j + f2*N] +
                            IG[j + f2*N, i + f1*N] *  G[i + f1*N, j + f2*N]
        end
    end
    push!(m.obs, m.temp)
end


    )
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



"""
    MagnetizationMeasurement(mc::DQMC, model)

Measures:
* `x`, `y`, `z`: the average onsite magnetization in x, y, or z direction
"""
struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
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

    MagnetizationMeasurement(m1x, m1y, m1z)
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # Magnetization
    # c_{i, up}^† c_{i, down} + c_{i, down}^† c_{i, up}
    mx = [- G[i+N, i] - G[i, i+N]           for i in 1:N]
    # -i [c_{i, up}^† c_{i, down} - c_{i, down}^† c_{i, up}]
    my = [-1im * (G[i, i+N] - G[i+N, i])    for i in 1:N]
    # c_{i, up}^† c_{i, up} - c_{i, down}^† c_{i, down}
    mz = [G[i+N, i+N] - G[i, i]             for i in 1:N]
    push!(m.x, mx)
    push!(m.y, my)
    push!(m.z, mz)
end



"""
    SpinDensityCorrelationMeasurement(mc::DQMC, model)

Measures:
* `x`, `y`, `z`: the average spin density correlation between any two sites
"""
struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
end
function SpinDensityCorrelationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    m2x = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation x", "Observables.jld", "sdc-x"
    )
    m2y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation y", "Observables.jld", "sdc-y"
    )
    m2z = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation z", "Observables.jld", "sdc-z"
    )

    SpinDensityCorrelationMeasurement(m2x, m2y, m2z)
end
function measure!(m::SpinDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # NOTE
    # these maybe wrong, maybe IG -> G
    # Spin Density Correlation
    m2x = zeros(eltype(G), N, N)
    m2y = zeros(eltype(G), N, N)
    m2z = zeros(eltype(G), N, N)
    for i in 1:N, j in 1:N
        m2x[i, j] = (
            IG[i+N, i] * IG[j+N, j] + IG[j+N, i] * G[i+N, j] +
            IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
            IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] +
            IG[i, i+N] * IG[j, j+N] + IG[j, i+N] * G[i, j+N]
        )
        m2y[i, j] = (
            - IG[i+N, i] * IG[j+N, j] - IG[j+N, i] * G[i+N, j] +
              IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
              IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] -
              IG[i, i+N] * IG[j, j+N] - IG[j, i+N] * G[i, j+N]
        )
        m2z[i, j] = (
            IG[i, i] * IG[j, j] + IG[j, i] * G[i, j] -
            IG[i, i] * IG[j+N, j+N] - IG[j+N, i] * G[i+N, j] -
            IG[i+N, i+N] * IG[j, j] - IG[j, i+N] * G[i, j+N] +
            IG[i+N, i+N] * IG[j+N, j+N] + IG[j+N, i+N] * G[i+N, j+N]
        )
    end
    push!(m.x, m2x)
    push!(m.y, m2y)
    push!(m.z, m2z)
end



"""
    PairingCorrelationMeasurement(mc::DQMC, model)

Measures the s-wave equal-time pairing correlation matrix (`.mat`) and its uniform
Fourier transform (`.uniform_fourier`).
"""
struct PairingCorrelationMeasurement{
        OT1 <: AbstractObservable,
        OT2 <: AbstractObservable,
        T
    } <: SpinOneHalfMeasurement
    mat::OT1
    uniform_fourier::OT2
    temp::Matrix{T}
end
function PairingCorrelationMeasurement(mc::DQMC, model)
    T = eltype(mc.s.greens)
    N = nsites(model)

    obs1 = LightObservable(
        LogBinner(zeros(T, N, N)),
        "Equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s"
    )
    obs2 = LightObservable(
        LogBinner(T),
        "Uniform Fourier tranforms of equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s Fourier"
    )
    temp = zeros(T, N, N)

    PairingCorrelationMeasurement(obs1, obs2, temp)
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc)
    N = nsites(model)
    m.temp .= G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N]
    push!(m.mat, m.temp)
    push!(m.uniform_fourier, sum(m.temp) / N)
end
