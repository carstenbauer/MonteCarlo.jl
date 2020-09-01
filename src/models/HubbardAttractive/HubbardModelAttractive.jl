const HubbardConf = Array{Int8, 2} # conf === hsfield === discrete Hubbard Stratonovich field (Hirsch field)
const HubbardDistribution = Int8[-1,1]

"""
Famous attractive (negative U) Hubbard model on a cubic lattice.
Discrete Hubbard Stratonovich transformation (Hirsch transformation) in the density/charge channel,
such that HS-field is real.

    HubbardModelAttractive(; dims, L[, kwargs...])

Create an attractive Hubbard model on `dims`-dimensional cubic lattice
with linear system size `L`. Additional allowed `kwargs` are:

 * `mu::Float64=0.0`: chemical potential
 * `U::Float64=1.0`: onsite interaction strength, "Hubbard U"
 * `t::Float64=1.0`: hopping energy
"""
@with_kw_noshow struct HubbardModelAttractive{LT<:AbstractLattice} <: Model
    # user mandatory
    dims::Int
    L::Int

    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0

    # non-user fields
    l::LT = choose_lattice(HubbardModelAttractive, dims, L)
    flv::Int = 1

    # to avoid allocations (TODO always real?)
    IG::Vector{Float64} = Vector{Float64}(undef, length(l))
    G::Vector{Float64} = Vector{Float64}(undef, length(l))
end


function choose_lattice(::Type{HubbardModelAttractive}, dims::Int, L::Int)
    if dims == 1
        return Chain(L)
    elseif dims == 2
        return SquareLattice(L)
    else
        return CubicLattice(dims, L)
    end
end

"""
    HubbardModelAttractive(params::Dict)
    HubbardModelAttractive(params::NamedTuple)

Create an attractive Hubbard model with (keyword) parameters as specified in the
dictionary/named tuple `params`.
"""
HubbardModelAttractive(params::Dict{Symbol, T}) where T = HubbardModelAttractive(; params...)
HubbardModelAttractive(params::NamedTuple) = HubbardModelAttractive(; params...)

# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelAttractive) = "$(model.dims)D attractive Hubbard model"
Base.show(io::IO, model::HubbardModelAttractive) = print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelAttractive) = print(io, model)




# implement `Model` interface
@inline nsites(m::HubbardModelAttractive) = length(m.l)


# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::HubbardModelAttractive, nslices::Int) = rand(HubbardDistribution, nsites(m), nslices)


"""
Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.

Note that since we have a time reversal symmetry relating spin-up
to spin-down we only consider one spin sector (one flavor) for the attractive
Hubbard model in the DQMC simulation.

This isn't a performance critical method as it is only used once before the
actual simulation.
"""
function hopping_matrix(mc::DQMC, m::HubbardModelAttractive{L}) where {L<:AbstractLattice}
    N = nsites(m)
    T = diagm(0 => fill(-m.mu, N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(m.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -m.t
        end
    end

    return T
end


"""
Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline @bm function interaction_matrix_exp!(mc::DQMC, m::HubbardModelAttractive,
            result::Matrix, conf::HubbardConf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(0.5 * m.U * dtau))

    z = zero(eltype(result))
    @inbounds for j in eachindex(result)
        result[j] = z
    end
    N = size(result, 1)
    @inbounds for i in 1:N
        result[i, i] = exp(sign(power) * lambda * conf[i, slice])
    end
    nothing
end


@inline @bm function propose_local(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf)
    # see for example dos Santos (2002)
    greens = mc.s.greens
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))

    @inbounds ΔE_boson = -2. * lambda * conf[i, slice]
    γ = exp(ΔE_boson) - 1
    @inbounds detratio = (1 + γ * (1 - greens[i,i]))^2 # squared because of two spin sectors.

    return detratio, ΔE_boson, γ
end

@inline @bm function accept_local!(
        mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, 
        detratio, ΔE_boson, γ)
    greens = mc.s.greens

    # Unoptimized Version
    # u = -greens[:, i]
    # u[i] += 1.
    # # OPT: speed check, maybe @views/@inbounds
    # greens .-= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * greens[i, :]))
    # conf[i, slice] .*= -1.

    # Optimized
    # copy! and `.=` allocate, this doesn't. Synced loop is marginally faster
    @avx for j in eachindex(m.IG)
        m.IG[j] = -greens[j, i]
        m.G[j] = greens[i, j]
    end
    @inbounds m.IG[i] += 1.0
    # This is way faster for small systems and still ~33% faster at L = 15
    # Also no allocations here
    @inbounds x = γ / (1.0 + γ * m.IG[i])
    @avx for k in eachindex(m.IG), l in eachindex(m.G)
        greens[k, l] -= m.IG[k] * x * m.G[l]
    end
    @inbounds conf[i, slice] *= -1
    nothing
end




# implement DQMC interface: optional
"""
Green's function is real for the attractive Hubbard model.
"""
@inline greenseltype(::Type{DQMC}, m::HubbardModelAttractive) = Float64

"""
Calculate energy contribution of the boson, i.e. Hubbard-Stratonovich/Hirsch field.
"""
@inline function energy_boson(mc::DQMC, m::HubbardModelAttractive, hsfield::HubbardConf)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end

# See configurations.jl - compression of configurations
compress(::DQMC, ::HubbardModelAttractive, c) = BitArray(c .== 1)
function decompress(
        mc::DQMC{M, CB, CT}, ::HubbardModelAttractive, c
    ) where {M, CB, CT}
    CT(2c .- 1)
end

function save_model(
        file::JLD.JldFile,
        m::HubbardModelAttractive,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))

    write(file, entryname * "/dims", m.dims)
    write(file, entryname * "/L", m.L)
    write(file, entryname * "/mu", m.mu)
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
function load_model(data::Dict, ::Type{T}) where T <: HubbardModelAttractive
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HubbardModelAttractive version $(data["VERSION"])"))
    end

    l = load_lattice(data["l"], data["l"]["type"])
    data["type"](
        dims = data["dims"],
        L = data["L"],
        mu = data["mu"],
        U = data["U"],
        t = data["t"],
        l = l,
        flv = data["flv"]
    )
end
