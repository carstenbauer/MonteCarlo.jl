# What should be here
# - interaction_matrix_exp  <--
# - interaction_matrix_type  <--
# - propose_local  <--
# - accept_local  <--
# - energy_boson  <--
# - propose_global_from_conf  <--
# - rand  <--

# accepted
# - de/compress  <--

# maybe
# - this should replace conf?  <-- this seems like a good idea?
#   - this would simplify de/compress code, I think (mc, model independent)
#   - could have conf = get_default(model)
#   - this could probably be delayed enough to hold the constants it needs
#   - would stop this annoying value-less type dispatch
#   - this could also hold temp arrays BUT then temp_conf should not be like this
#       - actually maybe temp conf should be part of this?

# denied
# - U (not needed after construction)
# - move constant to stack (probably with eltype of interaction?)
# - this should replace conf?  <-- this seems like a good idea?
#   - if this knows U model would become largely irrelevant?
#     ... maybe but U is not useful here
#   - maybe we should even have detratio etc in there?
#     ... because we track positive, imaginary etc

# TODO
# - do we need mc in propose_local? accept_local?
# - update the world (see <-- above)
# - save_field

# TODO
# - simplify Diagonal to vectors?

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



abstract type AbstractHirschField <: AbstractField end

Base.rand(f::AbstractHirschField) = rand((Int8(-1), Int8(1)), size(f.conf))
Random.rand!(f::AbstractHirschField) = rand!(f.conf, (Int8(-1), Int8(1)))
compress(f::AbstractHirschField) = BitArray(f.conf .== 1)
compressed_conf_type(::AbstractHirschField) = BitArray
decompress(::AbstractHirschField, c) = Int8(2c .- 1)
decompress!(f::AbstractHirschField, c) = f.conf .= Int8.(2c .- 1)


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
struct DensityHirschField{T} <: AbstractHirschField
    α::T
    IG::Vector{T}
    G::Vector{T}
    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end

function DensityHirschField(param::DQMCParameters, model, U = model.U) # TODO type
    α = acosh(exp(0.5 * U * param.delta_tau))
    DensityHirschField(
        α,
        Vector{typeof(α)}(undef, length(lattice(model))),
        Vector{typeof(α)}(undef, length(lattice(model))),
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::DensityHirschField) = 1
interaction_matrix_type(::DensityHirschField{Float64}) = Diagonal{Float64, Vector{Float64}}
# interaction_matrix_type(::DensityHirschField{ComplexF64}) = Diagonal{ComplexF64, CVec64}
init_interaction_matrix(f::DensityHirschField{Float64}) = Diagonal(Vector{Float64}(undef, size(f.conf, 1)))
# init_interaction_matrix(f::DensityHirschField{ComplexF64}) = Diagonal(CVec64(undef, size(f.conf, 1)))

@inline function interaction_matrix_exp!(f::DensityHirschField, result::Diagonal, slice, power)
    @inbounds for i in eachindex(result.diag)
        result.diag[i] = exp(power * f.α * f.conf[i, slice])
    end
    nothing
end

# TODO
# both of these could be a little bit faster if we used the zeros imposed by
# BlockDiagonal (i.e. G[i, i+N] = 0)

function propose_local(mc, f::DensityHirschField, greens, i, slice)
    # see for example dos Santos Introduction to quantum Monte-Carlo
    # This assumes the Greens (, hopping and interaction) matrix to have the form
    # A 0
    # 0 A
    # where the blocks represent different spin sectors (up-up, up-down, etc).
    # For the determinannt ratio we would usually take the value at (i, i) from
    # each sector to calculate the determinant, but with this structure it 
    # simplifies to the code below.

    @inbounds ΔE_boson = -2.0 * f.α * f.conf[i, slice]
    γ = exp(ΔE_boson) - 1
    @inbounds detratio = (1 + γ * (1 - greens[i,i]))^2 

    return detratio, ΔE_boson, γ
end

function accept_local!(mc, f::DensityHirschField, greens, i, slice, detratio, ΔE_boson, γ)
    # uses: 
    # - mc.stack.greens
    greens = mc.stack.greens

    # copy! and `.=` allocate, this doesn't. Synced loop is marginally faster
    @turbo for j in eachindex(f.IG)
        f.IG[j] = -greens[j, i]
        f.G[j]  =  greens[i, j]
    end
    @inbounds f.IG[i] += 1.0
    
    # This is way faster for small systems and still ~33% faster at L = 15
    # Also no allocations here
    @inbounds x = γ / (1.0 + γ * f.IG[i])
    @turbo for k in eachindex(f.IG), l in eachindex(f.G)
        greens[k, l] -= f.IG[k] * x * f.G[l]
    end
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
struct MagneticHirschField{T} <: AbstractHirschField
    α::T
    
    IG::Matrix{T}
    IGR::Matrix{T}
    R::Matrix{T}
    Δ::Diagonal{T, Vector{T}}
    RΔ::Matrix{T}

    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end


function MagneticHirschField(param::DQMCParameters, model, U = model.U) # TODO type
    α = acosh(exp(-0.5 * U * param.delta_tau))
    N = length(lattice(model))
    MagneticHirschField(
        α,
        Matrix{eltype(α)}(undef, 2N, 2),
        Matrix{eltype(α)}(undef, 2N, 2),
        Matrix{eltype(α)}(undef, 2, 2),
        Diagonal(Vector{eltype(α)}(undef, 2)),
        Matrix{eltype(α)}(undef, 2, 2),
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::MagneticHirschField) = 2
interaction_matrix_type(::MagneticHirschField{Float64}) = Diagonal{Float64, Vector{Float64}}
# interaction_matrix_type(::MagneticHirschField{ComplexF64}) = Diagonal{ComplexF64, CVec64}
init_interaction_matrix(f::MagneticHirschField{Float64}) = Diagonal(Vector{Float64}(undef, 2size(f.conf, 1)))
# init_interaction_matrix(f::MagneticHirschField{ComplexF64}) = Diagonal(CVec64(undef, size(f.conf, 1)))

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

function propose_local(mc, f::MagneticHirschField, G, i, slice)
    # BlockDiagonal would get rid of R[1, 2] and R[2, 1]
    N = size(f.conf, 1)

    ΔE_Boson = -2.0 * f.α * f.conf[i, slice]
    f.Δ[1, 1] = exp(+ΔE_Boson) - 1.0
    f.Δ[2, 2] = exp(-ΔE_Boson) - 1.0

    # Unrolled R = I + Δ * (I - G)
    f.R[1, 1] = 1.0 + f.Δ[1, 1] * (1.0 - G[i, i])
    f.R[1, 2] =     - f.Δ[1, 1] * G[i, i+N]
    f.R[2, 1] =     - f.Δ[2, 2] * G[i+N, i]
    f.R[2, 2] = 1.0 + f.Δ[2, 2] * (1.0 - G[i+N, i+N])

    # Calculate det of 2x2 Matrix
    # det() vs unrolled: 206ns -> 2.28ns
    detratio = f.R[1, 1] * f.R[2, 2] - f.R[1, 2] * f.R[2, 1]

    # There is no bosonic part (exp(-ΔE_Boson)) to the partition function.
    # Therefore pass 0.0
    return detratio, 0.0, nothing
end

function accept_local!(mc, f::MagneticHirschField, G, i, slice, detratio, ΔE_boson, passthrough)
    # uses:
    # - mc.stack.greens
    # - mc.stack.greens_temp
    @bm "accept_local (init)" begin
        N = size(f.conf, 1)
        IG  = f.IG
        IGR = f.IGR
        Δ   = f.Δ
        R   = f.R
        RΔ  = f.RΔ
    end

    # inverting R in-place, using that R is 2x2, i.e.:
    # M^-1 = [a b; c d]^-1 = 1/det(M) [d -b; -c a]
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
        @inbounds for m in axes(IG, 1)
            IG[m, 1] = -G[m, i]
            IG[m, 2] = -G[m, i+N]
        end
        IG[i, 1] += 1.0
        IG[i+N, 2] += 1.0
        vmul!(RΔ, R, Δ)
        vmul!(IGR, IG, RΔ)
    end

    # Slowest part, don't think there's much more we can do
    @bm "accept_local (finalize computation)" begin
        # G = G - IG * (R * Δ) * G[i:N:end, :]

        # @turbo for m in axes(G, 1), n in axes(G, 2)
        #     mc.s.greens_temp[m, n] = IGR[m, 1] * G[i, n] + IGR[m, 2] * G[i+N, n]
        # end

        # @turbo for m in axes(G, 1), n in axes(G, 2)
        #     G[m, n] = G[m, n] - mc.s.greens_temp[m, n]
        # end

        # BlockDiagonal version
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.stack.greens_temp.blocks[1]
        temp2 = mc.stack.greens_temp.blocks[2]

        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1[m, n] = IGR[m, 1] * G1[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2[m, n] = IGR[m+N, 2] * G2[i, n]
        end

        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1[m, n] = G1[m, n] - temp1[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2[m, n] = G2[m, n] - temp2[m, n]
        end

        # Always
        f.conf[i, slice] *= -1
    end

    nothing
end

@inline energy_boson(f::MagneticHirschField, conf = f.conf) = 0.0

# DEFAULT
# @bm function propose_global_from_conf(f::MagneticHirschField, mc::DQMC)
# end



################################################################################
### Gauß-Hermite Quadrature
################################################################################


abstract type AbstractGHQField <: AbstractField end


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
struct MagneticGHQField{T} <: AbstractGHQField
    α::T
    γ::Vector{Float64}
    η::Vector{Float64}
    choices::Matrix{Int8}

    # TODO CVec/Mat vs normal arrays
    IG::Matrix{T}
    IGR::Matrix{T}
    R::Matrix{T}
    Δ::Diagonal{T, Vector{T}} # TODO - just vector?
    RΔ::Matrix{T} # TODO - probably unnecessary

    temp_conf::Matrix{Int8}
    conf::Matrix{Int8}
end

function MagneticGHQField(param::DQMCParameters, model, U = model.U)
    _α = sqrt(-0.5 * param.delta_tau * ComplexF64(U))
    α = abs(imag(_α)) < 1e-12 ? real(_α) : _α
    s6 = sqrt(6)
    gammas = Float64[1 - s6/3, 1 + s6/3, 1 + s6/3, 1 - s6/3]
    etas = Float64[-sqrt(6 + 2s6), -sqrt(6 - 2s6), sqrt(6 - 2s6), sqrt(6 + 2s6)]
    choices = Int8[2 3 4; 1 3 4; 1 2 4; 1 2 3]

    N = length(lattice(model))
    MagneticGHQField(
        α, gammas, etas, choices,
        Matrix{eltype(α)}(undef, 2N, 2),
        Matrix{eltype(α)}(undef, 2N, 2),
        Matrix{eltype(α)}(undef, 2, 2),
        Diagonal(Vector{eltype(α)}(undef, 2)),
        Matrix{eltype(α)}(undef, 2, 2),
        Matrix{Int8}(undef, length(lattice(model)), param.slices),
        Matrix{Int8}(undef, length(lattice(model)), param.slices)
    )
end

nflavors(::MagneticGHQField) = 2
interaction_matrix_type(::MagneticGHQField{Float64}) = Diagonal{Float64, Vector{Float64}}
interaction_matrix_type(::MagneticGHQField{ComplexF64}) = Diagonal{ComplexF64, CVec64}
init_interaction_matrix(f::MagneticGHQField{Float64}) = Diagonal(Vector{Float64}(undef, 2size(f.conf, 1)))
init_interaction_matrix(f::MagneticGHQField{ComplexF64}) = Diagonal(CVec64(undef, size(f.conf, 1)))
energy_boson(mc, ::MagneticGHQField, conf=nothing) = 0.0


# TODO: Maybe worth adding a complex method?
@inline @bm function interaction_matrix_exp!(
        f::MagneticGHQField{Float64}, result::Diagonal, slice, power
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


@inline @bm function MonteCarlo.propose_local(mc, f::MagneticGHQField, G, i, slice)
    N = size(f.conf, 1)
    x_old = f.conf[i, slice]
    x_new = @inbounds f.choices[x_old, rand(1:3)]

    exp_ratio = exp(f.α * (f.η[x_new] - f.η[x_old]))
    f.Δ.diag[1] = exp_ratio - 1.0
    f.Δ.diag[2] = 1 / exp_ratio - 1.0

    # Unrolled R = I + Δ * (I - G)
    f.R[1, 1] = 1.0 + f.Δ.diag[1] * (1.0 - G[i, i])
    f.R[1, 2] = - f.Δ.diag[1] * G[i, i+N] # 0 for BlockDiagonal
    f.R[2, 1] = - f.Δ.diag[2] * G[i+N, i] # 0 for BlockDiagonal
    f.R[2, 2] = 1.0 + f.Δ.diag[2] * (1.0 - G[i+N, i+N])
    
    detratio = f.R[1, 1] * f.R[2, 2] - f.R[1, 2] * f.R[2, 1]
    
    return detratio * f.γ[x_new] / f.γ[x_old], 0.0, (x_new, detratio)
end


@inline @bm function MonteCarlo.accept_local!(
        mc, f::MagneticGHQField, G, i, slice, detratio, ΔE_boson, passthrough
        # mc::DQMC, model::RepulsiveGHQHubbardModel, i::Int, slice::Int, conf::GHQConf, 
        # weight, ΔE_boson, passthrough
    )

    @bm "accept_local (init)" begin
        N = size(f.conf, 1)
        IG = f.IG
        IGR = f.IGR
        Δ = f.Δ
        R = f.R
        RΔ = f.RΔ
        x_new, detratio = passthrough
    end

    # inverting R in-place, using that R is 2x2
    @bm "accept_local (inversion)" begin
        # This does not include the γ part of the weight... I don't know why, 
        # but with it the results are wrong (regardless of whether we include γ
        # in interaction_matrix_exp)
        inv_div = 1.0 / detratio
        R[1, 2] = -R[1, 2] * inv_div
        R[2, 1] = -R[2, 1] * inv_div
        x = R[1, 1]
        R[1, 1] = R[2, 2] * inv_div
        R[2, 2] = x * inv_div
    end

    # Compute (I - G) R^-1 Δ
    @bm "accept_local (IG, R)" begin
        @inbounds for m in axes(IG, 1)
            IG[m, 1] = -G[m, i]
            IG[m, 2] = -G[m, i+N]
        end
        IG[i, 1] += 1.0
        IG[i+N, 2] += 1.0
        vmul!(RΔ, R, Δ)
        vmul!(IGR, IG, RΔ)
    end

    @bm "accept_local (finalize computation)" begin
        # BlockDiagonal version
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.stack.greens_temp.blocks[1]
        temp2 = mc.stack.greens_temp.blocks[2]

        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1[m, n] = IGR[m, 1] * G1[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2[m, n] = IGR[m+N, 2] * G2[i, n]
        end

        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1[m, n] = G1[m, n] - temp1[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2[m, n] = G2[m, n] - temp2[m, n]
        end

        # update conf
        f.conf[i, slice] = x_new
    end
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