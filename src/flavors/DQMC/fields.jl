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
            IG  = ntuple(_ -> VT(undef, N), flv)
            IGR = ntuple(_ -> VT(undef, N), flv)
            G   = ntuple(_ -> VT(undef, N), flv)
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

"""
    vsubkron!(G, L, R)

Calculates `G[i, j] = G[i, j] - (∑ₘ L[i, m] R[j, m])` where `m` can be omitted,
meaning L and R can be Vectors. Used for local updates.
"""
function vsubkron!(G::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    @turbo for k in eachindex(L), l in eachindex(R)
        G[k, l] -= L[k] * R[l]
    end  
end
function vsubkron!(G::Matrix{Float64}, L::Matrix{Float64}, R::Matrix{Float64})
    @turbo for k in axes(L, 1), l in axes(R, 1), m in axes(L, 2)
        G[k, l] -= L[k, m] * R[l, m]
    end  
end
function vsubkron!(G::CMat64, L::CVec64, R::CVec64)
    @turbo for k in eachindex(L), l in eachindex(R)
        G.re[k, l] -= L.re[k] * R.re[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.re[k, l] += L.im[k] * R.im[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.im[k, l] -= L.im[k] * R.re[l]
    end
    @turbo for k in eachindex(L), l in eachindex(R)
        G.im[k, l] -= L.re[k] * R.im[l]
    end
end
function vsubkron!(G::BlockDiagonal, L::Tuple, R::Tuple)
    # TODO Vector{Vector} for L and R?
    @inbounds for b in eachindex(G.blocks)
        vsubkron!(G.blocks[b], L[b], R[b])
    end
end

"""
    vsub!(trg, ::UniformScaling, src, slices)

Calculates `trg[i, j] = I[i, slices[j]] - src[i, slices[j]]` where 
`j = eachindex(slices)` and `slices` is a tuple of indices. 

`slices` can also be an integer when only one slice is relevant. In this case 
`trg` is assumed to be a Vector.
"""
function vsub!(trg::FVec64, ::UniformScaling, src::FMat64, i::Int)
    @turbo for j in eachindex(trg)
        trg[j] = - src[j, i]
    end
    @inbounds trg[i] += 1.0
    nothing
end
function vsub!(trg::CVec64, ::UniformScaling, src::CMat64, i::Int)
    @turbo for j in eachindex(trg.re)
        trg.re[j] = - src.re[j, i]
    end
    @inbounds trg.re[i] += 1.0
    @turbo for j in eachindex(trg.im)
        trg.im[j] = - src.im[j, i]
    end
    nothing
end
function vsub!(trg::FMat64, ::UniformScaling, src::FMat64, slices::NTuple)
    @inbounds for (k, i) in enumerate(slices)
        @turbo for j in axis(trg, 1)
            trg[j, k] = - G[j, i]
        end
        trg[i, k] += 1.0
    end
    nothing
end
function vsub!(trg::Tuple, ::UniformScaling, src::BlockDiagonal, i::Int)
    @inbounds for b in eachindex(trg)
        vsub!(trg[b], I, src.blocks[b], i)
    end
    nothing
end

"""
    vmul!(trg, M, src, slices)

Calculates `trg[i, j] = ∑ₖ M[j, k] * src[slices[k], i]` where 
`j, k = eachindex(slices)` and `slices` is a tuple of indices. 

`slices` can also be an integer when only one slice is relevant. In this case 
`trg` is assumed to be a Vector and `M` is assumed to be a Number.
"""
function vmul!(trg::FVec64, M::Float64, src::FMat64, i::Int)
    @turbo for j in eachindex(trg)
        trg[j] = M * src[i, j]
    end
    nothing
end
function vmuladd!(trg::FVec64, M::Float64, src::FMat64, i::Int)
    @turbo for j in eachindex(trg)
        trg[j] += M * src[i, j]
    end
    nothing
end
function vmul!(trg::CVec64, M::ComplexF64, src::CMat64, i::Int)
    vmul!(trg.re, real(M), src.re, i)
    vmuladd!(trg.re, -imag(M), src.im, i)
    vmul!(trg.im, real(M), src.im, i)
    vmuladd!(trg.im, imag(M), src.re, i)
    nothing
end
function vmul!(trg::FMat64, M::FMat64, src::FMat64, slices::NTuple)
    @inbounds for i in axes(trg, 1), j in eachindex(slices)
        tij = 0.0
        for (k, l) in enumerate(slices)
            tij += M[j, k] * src[l, i]
        end
        trg[i, j] = tij
    end
    nothing
end
function vmul!(trg::NTuple, M::Union{FVec64, CVec64}, src::BlockDiagonal, i::Int)
    @inbounds for b in eachindex(trg)
        vmul!(trg[b], M[b], src.blocks[b], i)
    end
    nothing
end

# vldiv!(invRΔ, R, Δ)


function update_greens!(cache::StandardFieldCache, G, i)
    update_greens!(cache::StandardFieldCache, cache.Δ, G, i)
end

function update_greens!(cache::StandardFieldCache, Δ::Float64, G::FMat64, i)
    # calculate Δ R⁻¹
    cache.invRΔ = Δ / cache.R
    
    # calculate (I - G)[:, i]
    vsub!(cache.IG, I, G, i)

    # calculate (Δ R⁻¹) * G[i, :]
    vmul!(cache.G, cache.invRΔ, G, i)

    # update greens function
    vsubkron!(G, cache.IG, cache.G)

    nothing
end
function update_greens!(cache::StandardFieldCache, Δ::ComplexF64, G::CMat64, i)
    # calculate Δ R⁻¹
    cache.invRΔ = Δ / cache.R
    
    # calculate (I - G)[:, i]
    vsub!(cache.IG, I, G, i)

    # calculate (Δ R⁻¹) G[i, :]
    vmul!(cache.G, cache.invRΔ, G, i)

    # update greens function
    vsubkron!(G, cache.IG, cache.G)

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

    # copy (I - G)[:, i:N:2N]
    vsub!(cache.IG, I, G, (i, i+N))

    # calculate (R⁻¹ Δ) * G[i:N:end, :]
    vmul!(cache.G, cache.invRΔ, G, (i, i+N))

    # update greens function
    vsubkron!(G, cache.IG, cache.G)

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
    vsub!(cache.IG, I, G, i)

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    vmul!(cache.G, cache.invRΔ, G, i)

    # update greens function
    vsubkron!(G, cache.IG, cache.G)

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
    vsub!(cache.IG, I, G, i)

    # copy G (necessary to avoid overwriten? probably also helps with cache misses)
    vmul!(cache.G, cache.invRΔ, G, i)

    # update greens function
    vsubkron!(G, cache.IG, cache.G)

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

    return nothing
end


@inline @bm function propose_local(mc, f::MagneticGHQField, i, slice)
    x_old = f.conf[i, slice]
    x_new = @inbounds f.choices[x_old, rand(1:3)]

    exp_ratio = exp(f.α * (f.η[x_new] - f.η[x_old]))
    mc.stack.field_cache.Δ[1] = exp_ratio - 1.0
    mc.stack.field_cache.Δ[2] = 1 / exp_ratio - 1.0
    detratio = calculate_detratio!(mc.stack.field_cache, mc.stack.greens, i)
    
    return detratio * f.γ[x_new] / f.γ[x_old], 0.0, x_new
end

@inline @bm function accept_local!(
        mc, f::MagneticGHQField, i, slice, detratio, ΔE_boson, x_new
    )
    update_greens!(mc.stack.field_cache, mc.stack.greens, i)

    # update conf
    @inbounds f.conf[i, slice] = x_new
    return nothing
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