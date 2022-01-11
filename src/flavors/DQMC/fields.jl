mutable struct StandardFieldCache{T1, T2, T3, T4} <: AbstractFieldCache
    Δ::T1
    R::T2
    invRΔ::T2
    IG::T3
    IGR::T3
    G::T3
    detratio::T4
end

function FieldCache(field, model)
    N = size(field.conf, 1)
    flv = max(nflavors(field), nflavors(model))
    T = interaction_eltype(field)
    GET = greens_eltype(field, model)
    VT = vector_type(GET)
    MT = matrix_type(GET)

    Δ = nflavors(field) == 1 ? zero(T) : vector_type(T)(undef, nflavors(field))

    if flv == 1
        # full greens matrix is flavor symmetric
        R   = zero(GET)
        invRΔ = zero(GET)
        IG  = VT(undef, N)
        IGR = VT(undef, N)
        G   = VT(undef, N)

    else
        if greens_matrix_type(field, model) <: BlockDiagonal
            # only blocks on diagonals matter
            R   = VT(undef, flv)
            invRΔ = VT(undef, flv)
            # fast iteration on first index
            IG  = MT(undef, N, flv)
            IGR = MT(undef, N, flv)
            G   = MT(undef, N, flv)
        else
            # all matter
            R   = MT(undef, flv, flv)
            invRΔ = MT(undef, flv, flv)
            # fast iteration on first index
            IG  = MT(undef, N*flv, flv)
            IGR = MT(undef, N*flv, flv)
            G   = MT(undef, N*flv, flv)
        end
        
    end

    detratio = zero(GET)

    return StandardFieldCache(Δ, R, invRΔ, IG, IGR, G, detratio)
end


################################################################################
### math for propose_local
################################################################################

# everything should be derivable from Δ and G

# This requires Δ to be set
function calculate_detratio!(cache::StandardFieldCache, G, i)
    calculate_detratio!(cache, cache.Δ, G, i)
end

# TODO Not sure if optimizing this is worth anything
function calculate_detratio!(cache::StandardFieldCache, Δ::Number, G::Union{FMat64, CMat64}, i)
    @inbounds cache.R = 1.0 + Δ * (1.0 - G[i, i])
    cache.detratio = cache.R * cache.R
end


function calculate_detratio!(cache::StandardFieldCache, Δ::FVec64, G::BlockDiagonal{Float64}, i)
    # Unrolled R = I + Δ * (I - G)
    @inbounds for b in eachindex(G.blocks)
        cache.R[b] = 1.0 + Δ[b] * (1.0 - G.blocks[b][i, i])
    end
    # determinant of Diagonal
    cache.detratio = prod(cache.R)
end
# function calculate_detratio!(cache::StandardFieldCache, Δ::FVec64, G::BlockDiagonal{ComplexF64}, i)
#     # Unrolled R = I + Δ * (I - G)
#     @inbounds for b in eachindex(G.blocks)
#         cache.R.re[b] = 1.0 + Δ[b] * (1.0 - G.blocks[b].re[i, i])
#     end
#     @inbounds for b in eachindex(G.blocks)
#         cache.R.im[b] = - Δ[b] * G.blocks.im[b][i, i]
#     end
#     # determinant of Diagonal (don't think this is worth optimizing)
#     cache.detratio = prod(cache.R)
# end
function calculate_detratio!(cache::StandardFieldCache, Δ::CVec64, G::BlockDiagonal{ComplexF64}, i)
    # Unrolled R = I + Δ * (I - G)
    @inbounds for b in eachindex(G.blocks)
        cache.R.re[b] = 1.0 + Δ.re[b] * (1.0 - G.blocks[b].re[i, i])
    end
    @inbounds for b in eachindex(G.blocks)
        cache.R.re[b] += Δ.im[b] * G.blocks[b].im[i, i]
    end
    @inbounds for b in eachindex(G.blocks)
        cache.R.im[b] = Δ.im[b] * (1.0 - G.blocks[b].re[i, i])
    end
    @inbounds for b in eachindex(G.blocks)
        cache.R.im[b] -= Δ.re[b] * G.blocks[b].im[i, i]
    end
    # determinant of Diagonal
    cache.detratio = prod(cache.R)
end


function calculate_detratio!(cache::StandardFieldCache, Δ::FVec64, G::FMat64, i)
    # TODO this is not nice
    N = div(size(G, 1), 2)
    
    # Unrolled R = I + Δ * (I - G)
    cache.R[1, 1] = 1.0 + Δ[1] * (1.0 - G[i, i])
    cache.R[1, 2] =     - Δ[1] * G[i, i+N]
    cache.R[2, 1] =     - Δ[2] * G[i+N, i]
    cache.R[2, 2] = 1.0 + Δ[2] * (1.0 - G[i+N, i+N])

    # Calculate det of 2x2 Matrix
    # det() vs unrolled: 206ns -> 2.28ns
    cache.detratio = cache.R[1, 1] * cache.R[2, 2] - cache.R[1, 2] * cache.R[2, 1]
end


################################################################################
### Math for accept_local
################################################################################


function update_greens!(cache::StandardFieldCache, G, i)
    update_greens!(cache::StandardFieldCache, cache.Δ, G, i)
end

function update_greens!(cache::StandardFieldCache, Δ::Float64, G::FMat64, i)
    # calculate Δ R⁻¹
    cache.invRΔ = Δ / cache.R
    
    # calculate (I - G)[:, i] * (Δ R⁻¹)
    @inbounds for j in eachindex(cache.IG)
        cache.IGR[j] = - cache.invRΔ * G[j, i]
    end
    cache.IGR[i] += cache.invRΔ

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    @inbounds for j in eachindex(cache.G)
        cache.G[j] = G[i, j]
    end

    # update greens function
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G)
        G[k, l] -= cache.IGR[k] * cache.G[l]
    end

    nothing
end
function update_greens!(cache::StandardFieldCache, Δ::ComplexF64, G::CMat64, i)
    # calculate Δ R⁻¹
    cache.invRΔ = Δ / cache.R
    
    # calculate (I - G)[:, i] * (Δ R⁻¹)
    @inbounds for j in eachindex(cache.IG)
        cache.IG.re[j] = - G.re[j, i]
    end
    @inbounds for j in eachindex(cache.IG)
        cache.IG.im[j] = - G.im[j, i]
    end
    cache.IG.re[i] += 1.0

    # copy G
    @inbounds for j in eachindex(cache.G)
        cache.G.re[j] = real(cache.invRΔ) * G.re[i, j]
    end
    @inbounds for j in eachindex(cache.G)
        cache.G.re[j] -= imag(cache.invRΔ) * G.im[i, j]
    end
    @inbounds for j in eachindex(cache.G)
        cache.G.im[j] = real(cache.invRΔ) * G.im[i, j]
    end
    @inbounds for j in eachindex(cache.G)
        cache.G.im[j] += imag(cache.invRΔ) * G.re[i, j]
    end

    # update greens function
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G)
        G.re[k, l] -= cache.IG.re[k] * cache.G.re[l]
    end
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G)
        G.re[k, l] += cache.IG.im[k] * cache.G.im[l]
    end
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G)
        G.im[k, l] -= cache.IG.im[k] * cache.G.re[l]
    end
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G)
        G.im[k, l] -= cache.IG.re[k] * cache.G.im[l]
    end

    nothing
end

function update_greens!(cache::StandardFieldCache, Δ::FVec64, G::FMat64, i)
    # TODO this is not nice
    N = div(size(G, 1), 2)

    # invert R inplace
    # using M^-1 = [a b; c d]^-1 = 1/det(M) [d -b; -c a]
    # TODO merge this with R⁻¹ * Δ
    @inbounds begin
        inv_div = 1.0 / cache.detratio
        cache.R[1, 2] = - cache.R[1, 2] * inv_div
        cache.R[2, 1] = - cache.R[2, 1] * inv_div
        x = cache.R[1, 1]
        cache.R[1, 1] = cache.R[2, 2] * inv_div
        cache.R[2, 2] = x * inv_div
    end
    
    # compute R⁻¹ Δ
    # TODO: merge with inversion of R
    vmul!(cache.invRΔ, cache.R, Diagonal(Δ))

    # copy (I - G)[:, i]
    @inbounds for j in eachindex(cache.IG)
        cache.IG[j, 1] = - G[j, i]
    end
    @inbounds for j in eachindex(cache.IG)
        cache.IG[j, 2] = - G[j, i+N]
    end
    @inbounds cache.IG[i, 1] += 1.0
    @inbounds cache.IG[i+N, 2] += 1.0

    # calculate (I - G)[:, i] (R⁻¹ Δ)
    vmul!(cache.IGR, cache.IG, cache.invRΔ)

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    @inbounds for j in eachindex(cache.G)
        cache.G[j, 1] = G[i, j]
    end
    @inbounds for j in eachindex(cache.G)
        cache.G[j, 2] = G[i+N, j]
    end

    # update greens function
    @turbo for k in eachindex(cache.IG), l in eachindex(cache.G), m in 1:2
        G[k, l] -= cache.IGR[k, m] * cache.G[l, m]
    end

    nothing
end

function update_greens!(cache::StandardFieldCache, Δ::FVec64, G::BlockDiagonal{Float64}, i)
    # TODO this is not nice
    N = size(G.blocks[1], 1)

    # invert R inplace
    # using M^-1 = [a b; c d]^-1 = 1/det(M) [d -b; -c a]
    # TODO merge this with R⁻¹ * Δ
    @inbounds begin
        cache.invRΔ[1] = Δ[1] / cache.R[1]
        cache.invRΔ[2] = Δ[2] / cache.R[2]
    end
    
    # copy (I - G)[:, i]
    @inbounds for b in eachindex(G.blocks)
        for j in 1:N
            cache.IG[j, b] = - G.blocks[b][j, i]
        end
    end
    @inbounds cache.IG[i, 1] += 1.0
    @inbounds cache.IG[i, 2] += 1.0

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    @inbounds for b in eachindex(G.blocks)
        for j in 1:N
            cache.G[j, b] = cache.invRΔ[b] * G.blocks[b][i, j]
        end
    end

    # update greens function
    @inbounds for b in eachindex(G.blocks)
        @turbo for k in 1:N, l in 1:N
            G.blocks[b][k, l] -= cache.IG[k, b] * cache.G[l, b]
        end
    end

    nothing
end
function update_greens!(cache::StandardFieldCache, Δ::CVec64, G::BlockDiagonal{ComplexF64}, i)
    # TODO this is not nice
    N = size(G.blocks[1], 1)

    # invert R inplace
    # Reminder: 1/c = c* / (cc*) (complex conjugate)
    @inbounds begin
        f1 = 1.0 / (cache.R.re[1] * cache.R.re[1] + cache.R.im[1] * cache.R.im[1])
        f2 = 1.0 / (cache.R.re[2] * cache.R.re[2] + cache.R.im[2] * cache.R.im[2])
        cache.invRΔ.re[1] = f1 * (Δ.re[1] * cache.R.re[1] + Δ.im[1] * cache.R.im[1])
        cache.invRΔ.re[2] = f2 * (Δ.re[2] * cache.R.re[2] + Δ.im[2] * cache.R.im[2])
        cache.invRΔ.im[1] = f1 * (Δ.im[1] * cache.R.re[1] - Δ.re[1] * cache.R.im[1])
        cache.invRΔ.im[2] = f2 * (Δ.im[2] * cache.R.re[2] - Δ.re[2] * cache.R.im[2])
    end
    
    # copy (I - G)[:, i]
    @inbounds for b in eachindex(G.blocks), fn in (:re, :im)
        trg = getproperty(cache.IG, fn); src = getproperty(G.blocks[b], fn)
        for j in 1:N
            trg[j, b] = - src[j, i]
        end
    end
    @inbounds cache.IG.re[i, 1] += 1.0
    @inbounds cache.IG.re[i, 2] += 1.0

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    @inbounds for b in eachindex(G.blocks)
        for j in 1:N
            cache.G.re[j, b] = cache.invRΔ.re[b] * G.blocks[b].re[i, j]
        end
        for j in 1:N
            cache.G.re[j, b] -= cache.invRΔ.im[b] * G.blocks[b].im[i, j]
        end
        for j in 1:N
            cache.G.im[j, b] = cache.invRΔ.re[b] * G.blocks[b].im[i, j]
        end
        for j in 1:N
            cache.G.im[j, b] += cache.invRΔ.im[b] * G.blocks[b].re[i, j]
        end
    end

    # update greens function
    @inbounds for b in eachindex(G.blocks)
        @turbo for k in 1:N, l in 1:N
            G.blocks[b].re[k, l] -= cache.IG.re[k, b] * cache.G.re[l, b]
        end
        @turbo for k in 1:N, l in 1:N
            G.blocks[b].re[k, l] += cache.IG.im[k, b] * cache.G.im[l, b]
        end
        @turbo for k in 1:N, l in 1:N
            G.blocks[b].im[k, l] -= cache.IG.im[k, b] * cache.G.re[l, b]
        end
        @turbo for k in 1:N, l in 1:N
            G.blocks[b].im[k, l] -= cache.IG.re[k, b] * cache.G.im[l, b]
        end
    end

    nothing
end


################################################################################


maybe_to_float(c::ComplexF64) = abs(imag(c)) < 10eps(real(c)) ? real(c) : c

conf(f::AbstractField) = f.conf
conf!(f::AbstractField, c) = conf(f) .= c
temp_conf(f::AbstractField) = f.temp_conf
Base.length(f::AbstractField) = length(conf(f))

function save_field(file, field::AbstractField, entryname="field")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/name", nameof(typeof(field)))
    write(file, entryname * "/conf", conf(field))
end

function load_field(data, ::Val{:Field}, param, model)
    name = data["name"]
    c = data["conf"]
    field = if name == "DensityHirschField"
        DensityHirschField(param, model)
    elseif name == "MagneticHirschField"
        MagneticHirschField(param, model)
    else
        @eval $(Symbol(name))($param, $model)
    end
    copyto!(conf(field), c)

    return field
end


################################################################################
### Hirsch Fields
################################################################################



abstract type AbstractHirschField{T} <: AbstractField end

Base.rand(f::AbstractHirschField) = rand((Int8(-1), Int8(1)), size(f.conf))
Random.rand!(f::AbstractHirschField) = rand!(f.conf, (Int8(-1), Int8(1)))
compress(f::AbstractHirschField) = BitArray(f.conf .== 1)
compressed_conf_type(::AbstractHirschField) = BitArray
decompress(::AbstractHirschField, c) = Int8(2c .- 1)
decompress!(f::AbstractHirschField, c) = f.conf .= Int8.(2c .- 1)

interaction_eltype(::AbstractHirschField{T}) where {T} = T
interaction_matrix_type(::AbstractHirschField{Float64}, ::Model) = Diagonal{Float64, FVec64}
interaction_matrix_type(::AbstractHirschField{ComplexF64}, ::Model) = Diagonal{ComplexF64, CVec64}
function init_interaction_matrix(f::AbstractHirschField{Float64}, m::Model)
    flv = max(nflavors(f), nflavors(m))
    Diagonal(FVec64(undef, flv * size(f.conf, 1)))
end
function init_interaction_matrix(f::AbstractHirschField{ComplexF64}, m::Model)
    flv = max(nflavors(f), nflavors(m))
    Diagonal(CVec64(undef, flv * size(f.conf, 1)))
end


########################################
# Density Channel
########################################


"""
    DensityHirschField

Represents the density channel Hirsch transformation of an on-site interaction
term.

exp(Δτ U (n↑ - 0.5)(n↓ - 0.5)) ~ ∑ₓ exp(α x (n↑ + n↓ + 1))
with α = acosh(exp(0.5 Δτ U)) and x ∈ {-1, 1}

This definition uses *positive* U for the attractive case
"""
struct DensityHirschField{T} <: AbstractHirschField{T}
    α::T

    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end

function DensityHirschField(param::DQMCParameters, model::Model, U::Number = model.U)
    DensityHirschField(
        maybe_to_float(acosh(exp(0.5 * param.delta_tau * ComplexF64(U)))),
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::DensityHirschField) = 1

@inline function interaction_matrix_exp!(f::DensityHirschField, result::Diagonal, slice, power)
    # TODO this will index out of bound with 2 flavors
    @inbounds for i in eachindex(result.diag)
        result.diag[i] = exp(power * f.α * f.conf[i, slice])
    end
    nothing
end

function propose_local(mc, f::DensityHirschField, i, slice)
    @inbounds ΔE_boson = -2.0 * f.α * f.conf[i, slice]
    mc.stack.field_cache.Δ = exp(ΔE_boson) - 1
    detratio = calculate_detratio!(mc.stack.field_cache, mc.stack.greens, i)
    return detratio, ΔE_boson, nothing
end

function accept_local!(mc, f::DensityHirschField, i, slice, args...)
    update_greens!(mc.stack.field_cache, mc.stack.greens, i)
    @inbounds f.conf[i, slice] *= -1
    nothing
end

@inline energy_boson(f::DensityHirschField, conf = f.conf) = f.α * sum(conf)

@bm function propose_global_from_conf(mc::DQMC, ::Model, f::DensityHirschField)
    # I don't think we need this...
    @assert mc.stack.current_slice == 1
    @assert mc.stack.direction == 1

    # This should be just after calculating greens, so mc.s.Dl is from the UDT
    # decomposed G
    copyto!(mc.stack.tempvf, mc.stack.Dl)

    # -1?
    inv_det(mc, current_slice(mc)-1, f)

    # This helps with stability
    detratio = 1.0
    for i in eachindex(mc.stack.tempvf)
        detratio *= mc.stack.tempvf[i] * mc.stack.Dr[i]
    end
    ΔE_Boson = energy_boson(f, conf(f)) - energy_boson(f, temp_conf(f))

    return detratio^2, ΔE_Boson, nothing
end


########################################
# Spin Channel
########################################


"""
    MagneticHirschField

Represents the spin channel Hirsch transformation of an on-site interaction
term.

exp(Δτ U (n↑ - 0.5)(n↓ - 0.5)) ~ ∑ₓ exp(α x (n↑ - n↓))
with α = acosh(exp(-0.5 Δτ U)) and x ∈ {-1, 1}
"""
struct MagneticHirschField{T} <: AbstractHirschField{T}
    α::T

    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end

function MagneticHirschField(param::DQMCParameters, model::Model, U::Number = model.U)
    MagneticHirschField(
        maybe_to_float(acosh(exp(-0.5 * param.delta_tau * ComplexF64(U)))),
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::MagneticHirschField) = 2

@inline function interaction_matrix_exp!(f::MagneticHirschField, result::Diagonal, slice, power)
    N = size(f.conf, 1)
    @inbounds for i in axes(f.conf, 1)
        result.diag[i]   = exp( power * f.α * f.conf[i, slice])
    end
    @inbounds for i in axes(f.conf, 1)
        result.diag[i+N] = exp(-power * f.α * f.conf[i, slice])
    end
    nothing
end

function propose_local(mc, f::MagneticHirschField, i, slice)
    ΔE_Boson = -2.0 * f.α * f.conf[i, slice]
    mc.stack.field_cache.Δ[1] = exp(+ΔE_Boson) - 1.0
    mc.stack.field_cache.Δ[2] = exp(-ΔE_Boson) - 1.0
    detratio = calculate_detratio!(mc.stack.field_cache, mc.stack.greens, i)

    # There is no bosonic part (exp(-ΔE_Boson)) to the partition function.
    # Therefore pass 0.0
    return detratio, 0.0, nothing
end

function accept_local!(mc, f::MagneticHirschField, i, slice, args...)
    update_greens!(mc.stack.field_cache, mc.stack.greens, i)
    @inbounds f.conf[i, slice] *= -1

    nothing
end

@inline energy_boson(f::MagneticHirschField, conf = f.conf) = 0.0

# DEFAULT
# @bm function propose_global_from_conf(f::MagneticHirschField, mc::DQMC)
# end



################################################################################
### Gauß-Hermite Quadrature
################################################################################


abstract type AbstractGHQField{T} <: AbstractField end


# These represent (-2, -1, +1, 2)
const _GHQVALS = (Int8(1), Int8(2), Int8(3), Int8(4))
Base.rand(f::AbstractGHQField) = rand(_GHQVALS, size(f.conf))
Random.rand!(f::AbstractGHQField) = rand!(f.conf, _GHQVALS)
compressed_conf_type(::AbstractGHQField) = BitArray
function compress(f::AbstractGHQField)
    # converts (1, 2, 3, 4) -> (00, 01, 10, 11)
    BitArray((div(v-1, 2), (v-1) % 2)[step] for v in f.conf for step in (1, 2))
end
function decompress(::AbstractGHQField, c)
    # converts (00, 01, 10, 11) -> (1, 2, 3, 4)
    map(1:2:length(c)) do i
        #  1    +    2 * bit1    +    bit2
        Int8(1) + Int8(2) * c[i] + Int8(c[i+1])
    end
end
function decompress!(f::AbstractGHQField, c)
    for i in eachindex(f.conf)
        #  1    +    2 * bit1    +    bit2
        Int8(1) + Int8(2) * c[i] + Int8(c[i+1])
    end
end

interaction_eltype(::AbstractGHQField{T}) where {T} = T
interaction_matrix_type(::AbstractGHQField{Float64}, ::Model) = Diagonal{Float64, FVec64}
interaction_matrix_type(::AbstractGHQField{ComplexF64}, ::Model) = Diagonal{ComplexF64, CVec64}
function init_interaction_matrix(f::AbstractGHQField{Float64}, m::Model)
    flv = max(nflavors(f), nflavors(m))
    Diagonal(FVec64(undef, flv * size(f.conf, 1)))
end
function init_interaction_matrix(f::AbstractGHQField{ComplexF64}, m::Model)
    flv = max(nflavors(f), nflavors(m))
    Diagonal(CVec64(undef, flv * size(f.conf, 1)))
end


"""
    MagneticGHQField

This represents a field generated by using (4 node) Gauss Hermite quadrature to 
solve the integral form of the Hubbard Stratonovich transformed interaction. It 
is a more general solution than the Hirsch transform, allowing any interaction 
that can eb written as V ~ Â² though the current implementation only allows
Hubbard interactions.

In general:
exp(Δτ U Â²) ~ ∑ₓ γ(x) exp(α η(x) Â)
with α = sqrt(Δτ U) and x ∈ {-2, -1, 1, 2}
"""
struct MagneticGHQField{T} <: AbstractGHQField{T}
    α::T
    γ::FVec64
    η::FVec64
    choices::Matrix{Int8}

    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end

function MagneticGHQField(param::DQMCParameters, model::Model, U::Number = model.U)
    α = maybe_to_float(sqrt(-0.5 * param.delta_tau * ComplexF64(U)))
    s6 = sqrt(6)
    gammas = Float64[1 - s6/3, 1 + s6/3, 1 + s6/3, 1 - s6/3]
    etas = Float64[-sqrt(6 + 2s6), -sqrt(6 - 2s6), sqrt(6 - 2s6), sqrt(6 + 2s6)]
    choices = Int8[2 3 4; 1 3 4; 1 2 4; 1 2 3]

    MagneticGHQField(
        α, gammas, etas, choices,
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::MagneticGHQField) = 2
energy_boson(mc, ::MagneticGHQField, conf=nothing) = 0.0


# TODO: Maybe worth adding a complex method?
@inline @bm function interaction_matrix_exp!(
        f::MagneticGHQField, result::Diagonal, slice, power
    )
    N = size(f.conf, 1)
    @inbounds for i in 1:N
        result.diag[i]   = exp(+power * f.α * f.η[f.conf[i, slice]])
    end
    @inbounds for i in 1:N
        result.diag[i+N] = exp(-power * f.α * f.η[f.conf[i, slice]])
    end

    nothing
end


@inline @bm function MonteCarlo.propose_local(mc, f::MagneticGHQField, i, slice)
    x_old = f.conf[i, slice]
    x_new = @inbounds f.choices[x_old, rand(1:3)]

    exp_ratio = exp(f.α * (f.η[x_new] - f.η[x_old]))
    mc.stack.field_cache.Δ[1] = exp_ratio - 1.0
    mc.stack.field_cache.Δ[2] = 1 / exp_ratio - 1.0
    detratio = calculate_detratio!(mc.stack.field_cache, mc.stack.greens, i)
    
    return detratio * f.γ[x_new] / f.γ[x_old], 0.0, x_new
end

@inline @bm function MonteCarlo.accept_local!(
        mc, f::MagneticGHQField, i, slice, detratio, ΔE_boson, x_new
    )
    update_greens!(mc.stack.field_cache, mc.stack.greens, i)

    # update conf
    @inbounds f.conf[i, slice] = x_new
end


################################################################################
### Util
################################################################################


function ConfigRecorder(::Type{<: AbstractField}, rate = 10)
    ConfigRecorder{BitArray}(rate)
end

function Base.push!(c::ConfigRecorder, field::AbstractField, sweep)
    (sweep % c.rate == 0) && push!(c.configs, compress(field))
    nothing
end

function BufferedConfigRecorder(::Type{<: AbstractField}, filename; rate = 10, chunk_size = 1000)
    BufferedConfigRecorder{BitArray}(filename, rate, chunk_size)
end

function Base.push!(cr::BufferedConfigRecorder, field::AbstractField, sweep)
    (sweep % cr.rate == 0) && _push!(cr, compress(field))
    nothing
end