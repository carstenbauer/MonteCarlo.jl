"""
Famous repulsive (positive U) Hubbard model on a lattice.
Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the 
spin/magnetic channel, such that HS-field is real.

    HubbardModelRepulsive(; dims, L[, kwargs...])

Create an repulsive Hubbard model on `dims`-dimensional lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `U::Float64=1.0`: onsite interaction strength, "Hubbard U"
 * `t::Float64=1.0`: hopping energy
"""
@with_kw_noshow struct HubbardModelRepulsive{LT<:AbstractLattice} <: HubbardModel
    # user mandatory
    dims::Int
    L::Int

    # user optional
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0

    # non-user fields
    l::LT = choose_lattice(HubbardModelRepulsive, dims, L)
    flv::Int = 2

    # to avoid allocations (TODO always real?)
    IG::Matrix{Float64}  = Matrix{Float64}(undef, 2length(l), 2)
    IGR::Matrix{Float64} = Matrix{Float64}(undef, 2length(l), 2)
    R::Matrix{Float64}   = Matrix{Float64}(undef, 2, 2)
    Δ::Diagonal{Float64, Vector{Float64}} = Diagonal(Vector{Float64}(undef, 2))
    RΔ::Matrix{Float64}  = Matrix{Float64}(undef, 2, 2)
end


"""
    HubbardModelRepulsive(params::Dict)
    HubbardModelRepulsive(params::NamedTuple)

Create an repulsive Hubbard model with (keyword) parameters as specified in the
dictionary/named tuple `params`.
"""
HubbardModelRepulsive(params::Dict{Symbol, T}) where T = HubbardModelRepulsive(; params...)
HubbardModelRepulsive(params::NamedTuple) = HubbardModelRepulsive(; params...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelRepulsive) = "$(model.dims)D repulsive Hubbard model"
Base.show(io::IO, model::HubbardModelRepulsive) = print(io, "$(model.dims)D repulsive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelRepulsive) = print(io, model)


# Convenience
@inline parameters(m::HubbardModelRepulsive) = (L = m.L, t = m.t, U = m.U)

# optional optimization
hopping_matrix_type(::Type{DQMC}, ::HubbardModelRepulsive) = BlockDiagonal{Float64, 2, Matrix{Float64}}
greens_matrix_type( ::Type{DQMC}, ::HubbardModelRepulsive) = BlockDiagonal{Float64, 2, Matrix{Float64}}



"""
    hopping_matrix(mc::DQMC, m::HubbardModelRepulsive)

Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

# TODO
# Note that since we have a time reversal symmetry relating spin-up
# to spin-down we only consider one spin sector (one flavor) for the repulsive
# Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, model::HubbardModelRepulsive)
    N = length(model.l)
    T = zeros(N, N)

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(model.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -model.t
        end
    end

    return BlockDiagonal(T, copy(T))
end


"""
    interaction_matrix_exp!(
        mc::DQMC, model::HubbardModelRepulsive, result, conf, slice, power
    )

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline @bm function interaction_matrix_exp!(mc::DQMC, model::HubbardModelRepulsive,
            result, conf::HubbardConf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(0.5 * model.U * dtau))

    # z = zero(eltype(result))
    # @inbounds for j in eachindex(result)
    #     result[j] = z
    # end
    # N = length(lattice(model))
    # @inbounds for i in 1:N
    #     result[i, i] = exp(sign(power) * lambda * conf[i, slice])
    # end
    # @inbounds for i in 1:N
    #     result[i+N, i+N] = exp(-sign(power) * lambda * conf[i, slice])
    # end
    N = length(lattice(model))
    @inbounds for i in 1:N
        result.diag[i] = exp(sign(power) * lambda * conf[i, slice])
    end
    @inbounds for i in 1:N
        result.diag[i+N] = exp(-sign(power) * lambda * conf[i, slice])
    end
    nothing
end

@inline @inbounds @bm function propose_local(
        mc::DQMC, model::HubbardModelRepulsive, i::Int, slice::Int, conf::HubbardConf
    )
    N = length(model.l)
    G = mc.s.greens
    Δτ = mc.p.delta_tau
    Δ = model.Δ
    R = model.R

    # 
    α = acosh(exp(0.5Δτ * model.U))
    ΔE_Boson = -2.0α * conf[i, slice]
    Δ[1, 1] = exp(ΔE_Boson) - 1.0
    Δ[2, 2] = exp(-ΔE_Boson) - 1.0

    # Unrolled R = I + Δ * (I - G)
    R[1, 1] = 1.0 + Δ[1, 1] * (1.0 - G[i, i])
    R[1, 2] = - Δ[1, 1] * G[i, i+N]
    R[2, 1] = - Δ[2, 2] * G[i+N, i]
    R[2, 2] = 1.0 + Δ[2, 2] * (1.0 - G[i+N, i+N])

    # Calculate det of 2x2 Matrix
    # det() vs unrolled: 206ns -> 2.28ns
    detratio = R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1]

    # There is no bosonic part (exp(-ΔE_Boson)) to the partition function.
    # Therefore pass 0.0
    return detratio, 0.0, nothing
end

@inline @inbounds @bm function accept_local!(
        mc::DQMC, model::HubbardModelRepulsive, i::Int, slice::Int, 
        conf::HubbardConf, detratio, args...
    )

    @bm "accept_local (init)" begin
        N = length(model.l)
        G = mc.s.greens
        IG = model.IG
        IGR = model.IGR
        Δ = model.Δ
        R = model.R
        RΔ = model.RΔ
    end

    # inverting R in-place, using that R is 2x2
    # speed up: 470ns -> 2.6ns
    @bm "accept_local (inversion)" begin
        inv_div = 1.0 / detratio
        R[1, 2] = -R[1, 2] * inv_div
        R[2, 1] = -R[2, 1] * inv_div
        x = R[1, 1]
        R[1, 1] = R[2, 2] * inv_div
        R[2, 2] = x * inv_div
    end

    # Compute (I - G) R^-1 Δ
    @bm "accept_local (IG, R)" begin
        @avx for m in axes(IG, 1)
            IG[m, 1] = -G[m, i]
            IG[m, 2] = -G[m, i+N]
        end
        IG[i, 1] += 1.0
        IG[i+N, 2] += 1.0
        vmul!(RΔ, R, Δ)
        vmul!(IGR, IG, RΔ)
    end

    # TODO SSSLLOOOWWW  -  explicit @avx loop?
    @bm "accept_local (finalize computation)" begin
        # G = G - IG * (R * Δ) * G[i:N:end, :]

        # @avx for m in axes(G, 1), n in axes(G, 2)
        #     mc.s.greens_temp[m, n] = IGR[m, 1] * G[i, n] + IGR[m, 2] * G[i+N, n]
        # end

        # @avx for m in axes(G, 1), n in axes(G, 2)
        #     G[m, n] = G[m, n] - mc.s.greens_temp[m, n]
        # end

        # BlockDiagonal version
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.s.greens_temp.blocks[1]
        temp2 = mc.s.greens_temp.blocks[2]

        @avx for m in axes(G1, 1), n in axes(G1, 2)
            temp1[m, n] = IGR[m, 1] * G1[i, n]
        end
        @avx for m in axes(G2, 1), n in axes(G2, 2)
            temp2[m, n] = IGR[m+N, 2] * G2[i, n]
        end

        @avx for m in axes(G1, 1), n in axes(G1, 2)
            G1[m, n] = G1[m, n] - temp1[m, n]
        end
        @avx for m in axes(G2, 1), n in axes(G2, 2)
            G2[m, n] = G2[m, n] - temp2[m, n]
        end

        # Always
        conf[i, slice] *= -1
    end

    nothing
end


"""
Calculate energy contribution of the boson, i.e. Hubbard-Stratonovich/Hirsch field.
"""
@inline function energy_boson(mc::DQMC, m::HubbardModelRepulsive, hsfield::HubbardConf)
    # dtau = mc.p.delta_tau
    # lambda = acosh(exp(m.U * dtau/2))
    # return lambda * sum(hsfield)
    0.0
end

greens(mc::DQMC, model::HubbardModelRepulsive) = greens(mc)

function save_model(
        file::JLDFile,
        m::HubbardModelRepulsive,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))

    write(file, entryname * "/dims", m.dims)
    write(file, entryname * "/L", m.L)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")
    write(file, entryname * "/flv", m.flv)

    nothing
end

#     load_parameters(data, ::Type{<: DQMCParameters})
#
# Loads a DQMCParameters object from a given `data` dictionary produced by
# `JLD.load(filename)`.
function _load(data, ::Type{T}) where T <: HubbardModelRepulsive
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HubbardModelRepulsive version $(data["VERSION"])"))
    end

    l = _load(data["l"], data["l"]["type"])
    data["type"](
        dims = data["dims"],
        L = data["L"],
        U = data["U"],
        t = data["t"],
        l = l,
        flv = data["flv"]
    )
end
