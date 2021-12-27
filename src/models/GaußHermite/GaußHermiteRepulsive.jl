# This implements a different field discretization for the Hubbard model. Rather
# than using an Ising field derived from the Hirsch transformation we use a 
# four-valued field derived from Gaussian quadrature. 
# See https://git.physik.uni-wuerzburg.de/ALF (e.g. documentation, upgrade.f90, 
# Fields_mod.f90) and https://arxiv.org/pdf/2009.04491.pdf for more information

const GHQConf = Array{Int8, 2} 
const GHQDistribution = (Int8(-2), Int8(-1), Int8(1), Int8(2))

"""
    RepulsiveGHQHubbardModel(; l[, U, mu, t])

H = -t ∑ cⱼ^† cᵢ + U ∑ (n↑ - n↓)²                v- we can ignore constants like this
  = -t ∑ cⱼ^† cᵢ - U ∑ [(n↑ - 0.5)(n↓ - 0.5) + 1/4]

"""
MonteCarlo.@with_kw_noshow struct RepulsiveGHQHubbardModel{LT <: AbstractLattice} <: Model
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0
    l::LT

    # non-user fields
    flv::Int = 2

    # to avoid allocations (TODO always real?)
    IG::Matrix{Float64}  = Matrix{Float64}(undef, 2length(l), 2)
    IGR::Matrix{Float64} = Matrix{Float64}(undef, 2length(l), 2)
    R::Matrix{Float64}   = Matrix{Float64}(undef, 2, 2)
    Δ::Diagonal{Float64, Vector{Float64}} = Diagonal(Vector{Float64}(undef, 2))
    RΔ::Matrix{Float64}  = Matrix{Float64}(undef, 2, 2)
end

# TODO constructors

# cosmetics
import Base.summary
import Base.show
Base.summary(model::RepulsiveGHQHubbardModel) = "Gauß-Hermite Quadrature Hubbard model"
function Base.show(io::IO, model::RepulsiveGHQHubbardModel)
    print(io, model.U < 0.0 ? "attractive " : "repulsive ")
    print(io, "Gauß-Hermite Quadrature Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, ::MIME"text/plain", model::RepulsiveGHQHubbardModel) = print(io, model)

@inline MonteCarlo.nflavors(m::RepulsiveGHQHubbardModel) = m.flv
@inline MonteCarlo.lattice(m::RepulsiveGHQHubbardModel) = m.l

@inline function Base.rand(::Type{DQMC}, m::RepulsiveGHQHubbardModel, nslices::Int)
    rand(GHQDistribution, length(m.l), nslices)
end


# TODO: type optimizations
@inline function hopping_matrix_type(::Type{DQMC}, m::RepulsiveGHQHubbardModel)
    return BlockDiagonal{Float64, 2, Matrix{Float64}}
end
@inline function greens_matrix_type( ::Type{DQMC}, m::RepulsiveGHQHubbardModel)
    return BlockDiagonal{Float64, 2, Matrix{Float64}}
end
@inline function interaction_matrix_type(::Type{DQMC}, m::RepulsiveGHQHubbardModel)
    return Diagonal{Float64, Vector{Float64}}
end
@inline greenseltype(::Type{DQMC}, m::RepulsiveGHQHubbardModel) = Float64



function MonteCarlo.hopping_matrix(mc::DQMC, m::RepulsiveGHQHubbardModel{L}) where {L<:AbstractLattice}
    N = length(m.l)
    T = diagm(0 => fill(-m.mu, N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for (src, trg) in neighbors(m.l, Val(true))
            trg == -1 && continue
            T[trg, src] += -m.t
        end
    end

    return BlockDiagonal(T, copy(T))
end

function MonteCarlo.init_interaction_matrix(m::RepulsiveGHQHubbardModel)
    N = length(lattice(m))
    flv = nflavors(m)
    Diagonal(zeros(Float64, N*flv))
end


# These are essentialy the weights (γ) and nodes (η) of Gaussian quadrature.
# Equivalent to
# γ(±1) = 1 + sqrt(6) / 3
# γ(±2) = 1 - sqrt(6) / 3
# BigFloat:
# b = 3.449489742783178098197284074705891391965947480656670128432692567250960377457299
# m = -1.632993161855452065464856049803927594643964987104446752288461711500640251638199
_γ(x) = 3.449489742783178 - 1.632993161855452 * abs(x)

# Equivalent to
# η(±1) = ± sqrt(2 (3 - sqrt(6)) ) 
# η(±2) = ± sqrt(2 (3 + sqrt(6)) ) 
# BigFloat:
# b = -1.202769754670407840230441206902525721924555153129697926453305944908974747056691
# m = 2.25206500122098847599856171457200840475171228346862467025702629579654517230705
_η(x) = sign(x) * (2.2520650012209886 * abs(x) - 1.202769754670408)

@inline @bm function MonteCarlo.interaction_matrix_exp!(
        mc::DQMC, model::RepulsiveGHQHubbardModel,
        result::Diagonal, conf::GHQConf, slice::Int, power::Float64 = 1.
    )
    N = length(lattice(model))
    dtau = mc.parameters.delta_tau
    factor = sqrt(0.5 * dtau * model.U) 

    @inbounds for i in 1:N
        l = conf[i, slice]
        result.diag[i]   = exp(power * factor * _η(l))
    end
    @inbounds for i in 1:N
        l = conf[i, slice]
        result.diag[i+N] = exp(-power * factor * _η(l))
    end

    nothing
end


@inline @bm function MonteCarlo.propose_local(
        mc::DQMC, model::RepulsiveGHQHubbardModel, i::Int, slice::Int, conf::GHQConf
    )
    N = length(model.l)
    G = mc.stack.greens
    Δτ = mc.parameters.delta_tau
    Δ = model.Δ
    R = model.R

    x_old = conf[i, slice]
    # TODO: Might be fast to implement a table new_value[3+old_value, rand(1:3)]?
    if     x_old == -2; x_new = rand((    -1, 1, 2))
    elseif x_old == -1; x_new = rand((-2,     1, 2))
    elseif x_old ==  1; x_new = rand((-2, -1,    2))
    else                x_new = rand((-2, -1, 1   )) # x_old ==  2;
    end

    temp = sqrt(0.5 * Δτ * model.U)
    exp_ratio = exp(temp * (_η(x_new) - _η(x_old)))
    Δ[1, 1] = exp_ratio - 1.0
    Δ[2, 2] = 1 / exp_ratio - 1.0

    # Unrolled R = I + Δ * (I - G)
    R[1, 1] = 1.0 + Δ[1, 1] * (1.0 - G[i, i])
    R[1, 2] = - Δ[1, 1] * G[i, i+N] # 0 for BlockDiagonal
    R[2, 1] = - Δ[2, 2] * G[i+N, i] # 0 for BlockDiagonal
    R[2, 2] = 1.0 + Δ[2, 2] * (1.0 - G[i+N, i+N])
    
    detratio = R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1]
    
    return detratio * _γ(x_new) / _γ(x_old), 0.0, (x_new, detratio)
end


@inline @bm function MonteCarlo.accept_local!(
        mc::DQMC, model::RepulsiveGHQHubbardModel, i::Int, slice::Int, conf::GHQConf, 
        weight, ΔE_boson, passthrough
    )

    @bm "accept_local (init)" begin
        N = length(model.l)
        G = mc.stack.greens
        IG = model.IG
        IGR = model.IGR
        Δ = model.Δ
        R = model.R
        RΔ = model.RΔ
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
        conf[i, slice] = x_new
    end
end

# TODO global (test this)
energy_boson(mc, ::RepulsiveGHQHubbardModel, conf=nothing) = 0.0
function global_update(mc::DQMC, model::RepulsiveGHQHubbardModel, temp_conf::AbstractArray)
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, temp_conf)

    p = exp(- ΔE_boson) * detratio
    old_weight = mapreduce(_γ, *, mc.conf)
    new_weight = mapreduce(_γ, *, temp_conf)
    p *= new_weight / old_weight
    @assert imag(p) == 0.0 "p = $p should always be real because ΔE_boson = $ΔE_boson and detratio = $detratio should always be real..."

    # Gibbs/Heat bath
    # p = p / (1.0 + p)
    # Metropolis
    if p > 1 || rand() < p
        accept_global!(mc, model, temp_conf, passthrough)
        return 1
    end

    return 0
end

# checked
function MonteCarlo.compress(mc::DQMC, ::RepulsiveGHQHubbardModel, c)
    # converts (-2, -1, 1, 2) -> (10, 00, 01, 11)
    # first bit is value (0 -> 1, 1 -> 2), second is sign (0 -> -, 1 -> +)
    bools = [(abs(v) == 2, sign(v) == 1)[step] for v in c for step in (1, 2)]
    BitArray(bools)
end
MonteCarlo.compressed_conf_type(::Type{<: DQMC}, ::Type{<: RepulsiveGHQHubbardModel}) = BitArray
function MonteCarlo.decompress(::DQMC, ::RepulsiveGHQHubbardModel, c)
    map(1:2:length(c)) do i
        # (c[i] ? 2 : 1)       * (c[i+1] ? +1 : -1)
        (Int8(1) + Int8(c[i])) * (Int8(2) * Int8(c[i+1]) - Int8(1))
    end
end

# checked
function MonteCarlo.intE_kernel(mc, model::RepulsiveGHQHubbardModel, G::GreensMatrix)
    model.U * sum((diag(G.val.blocks[1]) .- 0.5) .* (diag(G.val.blocks[2]) .- 0.5))
end

function save_model(
        file::JLDFile,
        m::RepulsiveGHQHubbardModel,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "RepulsiveGHQHubbardModel")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")

    nothing
end

function _load(data, ::Val{:RepulsiveGHQHubbardModel})
    l = _load(data["l"], to_tag(data["l"]))
    RepulsiveGHQHubbardModel(
        mu = data["mu"],
        U = data["U"],
        t = data["t"],
        l = l
    )
end