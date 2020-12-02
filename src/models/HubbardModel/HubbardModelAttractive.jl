"""
    HubbardModelAttractive(lattice; params...)
    HubbardModelAttractive(L, dims; params...)
    HubbardModelAttractive(params::Dict)
    HubbardModelAttractive(params::NamedTuple)
    HubbardModelAttractive(; params...)

Defines an attractive (negative `U`) Hubbard model on a given (or derived) 
`lattice`. If a linear system size `L` and dimensionality `dims` is given, the
`lattice` will be a Cubic lattice of fitting size.

Additional parameters (keyword arguments) include:
* `l::AbstractLattice = lattice`: The lattice the model uses. The keyword 
argument takes precedence over the argument `lattice`.
* `U::Float64 = 1.0 > 0.0` is the absolute value of the Hubbard Interaction.
* `t::Float64 = 1.0` is the hopping strength.
* `mu::Float64` is the chemical potential.

Internally, a discrete Hubbard Stratonovich transformation (Hirsch 
transformation) is used in the spin/magnetic channel to enable DQMC. The 
resulting Hubbard Stratonovich fiels is real.
Furthermore, we use spin up/down symmetry to speed up the simulation. As a 
result the greens matrix is of size (N, N) with N the number of sites, and the
element G[i, j] corresponds to the up-up and down-down element. 
"""
@with_kw_noshow struct HubbardModelAttractive{LT<:AbstractLattice} <: HubbardModel
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0
    l::LT

    # non-user fields
    flv::Int = 1
    # to avoid allocations (TODO always real?)
    IG::Vector{Float64} = Vector{Float64}(undef, length(l))
    G::Vector{Float64} = Vector{Float64}(undef, length(l))
end


HubbardModelAttractive(params::Dict{Symbol}) = HubbardModelAttractive(; params...)
HubbardModelAttractive(params::NamedTuple) = HubbardModelAttractive(; params...)
function HubbardModelAttractive(lattice::AbstractLattice; kwargs...)
    HubbardModelAttractive(l = lattice; kwargs...)
end
function HubbardModelAttractive(L, dims; kwargs...)
    l = choose_lattice(HubbardModelAttractive, dims, L)
    HubbardModelAttractive(l = l; kwargs...)
end


# cosmetics
import Base.summary
import Base.show
Base.summary(model::HubbardModelAttractive) = "$(model.dims)D attractive Hubbard model"
function Base.show(io::IO, model::HubbardModelAttractive)
    print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
end
Base.show(io::IO, m::MIME"text/plain", model::HubbardModelAttractive) = print(io, model)


# Convenience
@inline parameters(m::HubbardModelAttractive) = (L = m.L, t = m.t, U = m.U, mu = m.mu)


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
    N = length(m.l)
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
            result::Diagonal, conf::HubbardConf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(0.5 * m.U * dtau))

    N = size(result, 1)
    @inbounds for i in 1:N
        result.diag[i] = exp(sign(power) * lambda * conf[i, slice])
    end
    nothing
end


@inline @bm function propose_local(
        mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf
    )
    # see for example dos Santos Introduction to quantum Monte-Carlo
    greens = mc.s.greens
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))

    @inbounds ΔE_boson = -2. * lambda * conf[i, slice]
    γ = exp(ΔE_boson) - 1
    @inbounds detratio = (1 + γ * (1 - greens[i,i]))^2 
    # squared because of two spin sectors

    return detratio, ΔE_boson, γ
end

@inline @bm function accept_local!(
        mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, 
        detratio, ΔE_boson, γ)
    greens = mc.s.greens

    # Unoptimized Version
    # u = -greens[:, i]
    # u[i] += 1.
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


"""
Calculate energy contribution of the boson, i.e. Hubbard-Stratonovich/Hirsch field.
"""
@inline function energy_boson(mc::DQMC, m::HubbardModelAttractive)
    hsfiled = conf(mc)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end


function greens(mc::DQMC, model::HubbardModelAttractive)
    G = greens!(mc)
    vcat(hcat(G, zeros(size(G))), hcat(zeros(size(G)), G))
end

function save_model(
        file::JLDFile,
        m::HubbardModelAttractive,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))

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
function _load(data, ::Type{T}) where T <: HubbardModelAttractive
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HubbardModelAttractive version $(data["VERSION"])"))
    end

    l = _load(data["l"], data["l"]["type"])
    data["type"](
        mu = data["mu"],
        U = data["U"],
        t = data["t"],
        l = l,
        flv = data["flv"]
    )
end



################################################################################
### Measurement overloads
################################################################################



# Need some measurement overwrites because nflavors = 1
checkflavors(::HubbardModelAttractive) = nothing

function cdc_kernel(mc, ::HubbardModelAttractive, i, j, G::AbstractArray)
    # spin up and down symmetric, so (i+N, i+N) = (i, i); (i+N, i) drops
    4 * (1 - G[i, i]) * (1 - G[j, j]) + 2 * (I[j, i] - G[j, i]) * G[i, j]
end
function cdc_kernel(mc, ::HubbardModelAttractive, i, j, pg::NTuple{4})
    G00, G0l, Gl0, Gll = pg
    # spin up and down symmetric, so (i+N, i+N) = (i, i); (i+N, i) drops
    4 * (1 - Gll[i,i]) * (1 - G00[j,j]) - 2 * G0l[j,i] * Gl0[i,j]
end

mx_kernel(mc, ::HubbardModelAttractive, i, G::AbstractArray) = 0.0
my_kernel(mc, ::HubbardModelAttractive, i, G::AbstractArray) = 0.0
mz_kernel(mc, ::HubbardModelAttractive, i, G::AbstractArray) = 0.0

sdc_x_kernel(mc, ::HubbardModelAttractive, i, j, G::AbstractArray) = 2(I[j,i] - G[j,i]) * G[i,j]
sdc_y_kernel(mc, ::HubbardModelAttractive, i, j, G::AbstractArray) = 2(I[j,i] - G[j,i]) * G[i,j]
sdc_z_kernel(mc, ::HubbardModelAttractive, i, j, G::AbstractArray) = 2(I[j,i] - G[j,i]) * G[i,j]

sdc_x_kernel(mc, ::HubbardModelAttractive, i, j, pg::NTuple{4}) = -2 * pg[2][j,i] * pg[3][i,j]
sdc_y_kernel(mc, ::HubbardModelAttractive, i, j, pg::NTuple{4}) = -2 * pg[2][j,i] * pg[3][i,j]
sdc_z_kernel(mc, ::HubbardModelAttractive, i, j, pg::NTuple{4}) = -2 * pg[2][j,i] * pg[3][i,j]

function pc_kernel(mc, ::HubbardModelAttractive, src1, trg1, src2, trg2, G::AbstractArray)
    G[src1, src2] * G[trg1, trg2]
end
function pc_kernel(mc, ::HubbardModelAttractive, src1, trg1, src2, trg2, pg::NTuple{4})
    pg[3][src1, src2] * pg[3][trg1, trg2]
end

function cc_kernel(mc, ::HubbardModelAttractive, src1, trg1, src2, trg2, pg::NTuple{4})
    G00, G0l, Gl0, Gll = pg
    N = length(lattice(mc))
    T = mc.s.hopping_matrix

    # up-up counts, down-down counts, mixed only on 11s or 22s
    s1 = src1; t1 = trg1
    s2 = src2; t2 = trg2
    output = 4.0 * 
        (T[s1, t1] * Gll[t1, s1] - T[t1, s1] * Gll[s1, t1]) * 
        (T[s2, t2] * G00[t2, s2] - T[t2, s2] * G00[s2, t2]) +
        2.0 * T[t1, s1] * T[t2, s2] * (- G0l[s2, t1]) * Gl0[s1, t2] -
        2.0 * T[s1, t1] * T[t2, s2] * (- G0l[s2, s1]) * Gl0[t1, t2] -
        2.0 * T[t1, s1] * T[s2, t2] * (- G0l[t2, t1]) * Gl0[s1, s2] +
        2.0 * T[s1, t1] * T[s2, t2] * (- G0l[t2, s1]) * Gl0[t1, s2]

    output
end