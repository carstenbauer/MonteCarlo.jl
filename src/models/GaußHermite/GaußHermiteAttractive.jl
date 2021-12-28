# This implements a different field discretization for the Hubbard model. Rather
# than using an Ising field derived from the Hirsch transformation we use a 
# four-valued field derived from Gaussian quadrature. 
# See https://git.physik.uni-wuerzburg.de/ALF (e.g. documentation, upgrade.f90, 
# Fields_mod.f90) and https://arxiv.org/pdf/2009.04491.pdf for more information

"""
    AttractiveGHQHubbardModel(; l[, U, mu, t])

H = -t ∑ cⱼ^† cᵢ - U ∑ (n↑ - n↓)²                v- we can ignore constants like this
  = -t ∑ cⱼ^† cᵢ + U ∑ [(n↑ - 0.5)(n↓ - 0.5) + 1/4]

"""
MonteCarlo.@with_kw_noshow struct AttractiveGHQHubbardModel{LT <: AbstractLattice} <: Model
    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0
    l::LT

    # non-user fields
    flv::Int = 2

    # to avoid allocations (TODO always real?)
    IG::CMat64  = StructArray(zeros(ComplexF64, 2length(l), 2))
    IGR::CMat64 = StructArray(Matrix{ComplexF64}(undef, 2length(l), 2))
    R::CVec64   = StructArray(Vector{ComplexF64}(undef, 2))
    Δ::CVec64   = StructArray(Vector{ComplexF64}(undef, 2))
end

# TODO constructors

# cosmetics
import Base.summary
import Base.show
Base.summary(model::AttractiveGHQHubbardModel) = "Gauß-Hermite Quadrature Hubbard model"
function Base.show(io::IO, model::AttractiveGHQHubbardModel)
    print(io, "attractive Gauß-Hermite Quadrature Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, ::MIME"text/plain", model::AttractiveGHQHubbardModel) = print(io, model)

@inline MonteCarlo.nflavors(m::AttractiveGHQHubbardModel) = m.flv
@inline MonteCarlo.lattice(m::AttractiveGHQHubbardModel) = m.l

@inline function Base.rand(::Type{DQMC}, m::AttractiveGHQHubbardModel, nslices::Int)
    rand(GHQDistribution, length(m.l), nslices)
end


# TODO: type optimizations
@inline function hopping_matrix_type(::Type{DQMC}, m::AttractiveGHQHubbardModel)
    return BlockDiagonal{Float64, 2, Matrix{Float64}}
end
@inline function greens_matrix_type( ::Type{DQMC}, m::AttractiveGHQHubbardModel)
    return BlockDiagonal{ComplexF64, 2, CMat64}
end
@inline function interaction_matrix_type(::Type{DQMC}, m::AttractiveGHQHubbardModel)
    return Diagonal{ComplexF64, CVec64}
end
@inline greenseltype(::Type{DQMC}, m::AttractiveGHQHubbardModel) = ComplexF64



function MonteCarlo.hopping_matrix(mc::DQMC, m::AttractiveGHQHubbardModel{L}) where {L<:AbstractLattice}
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

function MonteCarlo.init_interaction_matrix(m::AttractiveGHQHubbardModel)
    N = length(lattice(m))
    flv = nflavors(m)
    Diagonal(CVec64(undef, N*flv))
end


@inline @bm function MonteCarlo.interaction_matrix_exp!(
        mc::DQMC, model::AttractiveGHQHubbardModel,
        result::Diagonal, conf::GHQConf, slice::Int, power::Float64 = 1.
    )
    N = length(lattice(model))
    dtau = mc.parameters.delta_tau
    factor = sqrt(0.5 * dtau * ComplexF64(-model.U))

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

# TODO
# comment out previous changes, switch to charge channel?


@inline @bm function MonteCarlo.propose_local(
        mc::DQMC, model::AttractiveGHQHubbardModel, i::Int, slice::Int, conf::GHQConf
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

    temp = sqrt(0.5 * Δτ * ComplexF64(-model.U))
    exp_ratio = exp(temp * (_η(x_new) - _η(x_old)))
    Δ[1] = exp_ratio - 1.0
    Δ[2] = 1 / exp_ratio - 1.0

    # Unrolled R = I + Δ * (I - G)
    # Cross terms 0 with BlockDiagonal
    R.re[1] = 1.0 + Δ.re[1] * (1.0 - G.blocks[1].re[i, i])
    R.re[2] = 1.0 + Δ.re[2] * (1.0 - G.blocks[2].re[i, i])

    R.re[1] += Δ.im[1] * G.blocks[1].im[i, i]
    R.re[2] += Δ.im[2] * G.blocks[2].im[i, i]

    R.im[1] = - Δ.re[1] * G.blocks[1].im[i, i]
    R.im[2] = - Δ.re[2] * G.blocks[2].im[i, i]

    R.im[1] +=  Δ.im[1] * (1.0 - G.blocks[1].re[i, i])
    R.im[2] +=  Δ.im[2] * (1.0 - G.blocks[2].re[i, i])
    
    # detratio = R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1]
    detratio = ComplexF64(
        R.re[1] * R.re[2] - R.im[1] * R.im[2],
        R.re[1] * R.im[2] + R.im[1] * R.re[2]
    )

    # @info detratio
    
    return detratio * _γ(x_new) / _γ(x_old), 0.0, (x_new, detratio)
end


@inline @bm function MonteCarlo.accept_local!(
        mc::DQMC, model::AttractiveGHQHubbardModel, i::Int, slice::Int, conf::GHQConf, 
        weight, ΔE_boson, passthrough
    )

    @bm "accept_local (init)" begin
        N = length(model.l)
        G = mc.stack.greens
        IG = model.IG
        IGR = model.IGR
        Δ = model.Δ
        R = model.R
        x_new, detratio = passthrough
    end

    # Calculates R⁻¹ Δ -> R inplace using
    # - detratio = det(R)
    # - 1/c = c* / (c c*) (complex numbers)
    # - Diagonal shape of R
    # This does not include the γ part of the weight... I don't know why, 
    # but with it the results are wrong (regardless of whether we include γ
    # in interaction_matrix_exp)
    @bm "accept_local (inversion)" begin
        @inbounds @fastmath begin
            inv_div = 1.0 / abs2(detratio)
            Δ[1] *= inv_div
            Δ[2] *= inv_div
            inv_div1_re = Δ.re[1] * real(detratio) + Δ.im[1] * imag(detratio)
            inv_div1_im = Δ.im[1] * real(detratio) - Δ.re[1] * imag(detratio)
            inv_div2_re = Δ.re[2] * real(detratio) + Δ.im[2] * imag(detratio)
            inv_div2_im = Δ.im[2] * real(detratio) - Δ.re[2] * imag(detratio)

            # Need to be saved for imaginary part
            r11 = R.re[1]
            r22 = R.re[2]

            # inv_div_im's enter with negative sign because of implied conj
            # This does re = re² - im²
            R.re[1] = inv_div1_re * R.re[2] - inv_div1_im * R.im[2]
            R.re[2] = inv_div2_re * r11     - inv_div2_im * R.im[1]

            # This does im = re * im + im * re
            r11_im = R.im[1, 1]
            R.im[1] = inv_div1_re * R.im[2] + inv_div1_im * r22
            R.im[2] = inv_div2_re * r11_im  + inv_div2_im * r11
        end
    end

    # Compute (I - G)[:, i:N:end] {{R⁻¹ Δ}} -> IGR
    @bm "accept_local (IG, R)" begin
        # Do IG = I - G (relevant entries only)
        @turbo for m in axes(G.blocks[1], 1)
            IG.re[m, 1] = -G.blocks[1].re[m, i]
        end
        @turbo for m in axes(G.blocks[2], 1)
            IG.re[m+N, 2] = -G.blocks[2].re[m, i]
        end
        @turbo for m in axes(G.blocks[1], 1)
            IG.im[m, 1] = -G.blocks[1].im[m, i]
        end
        @turbo for m in axes(G.blocks[2], 1)
            IG.im[m+N, 2] = -G.blocks[2].im[m, i]
        end
        @inbounds IG.re[i, 1] += 1.0
        @inbounds IG.re[i+N, 2] += 1.0
                
        # Do IGR = IG * R
        vmul!(IGR, IG, Diagonal(R))
    end

    # Calculate G -= {{(I - G)[:, i:N:end] * R⁻¹ Δ}} G[i:N:end, :]
    @bm "accept_local (finalize computation)" begin
        # BlockDiagonal version
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.stack.greens_temp.blocks[1]
        temp2 = mc.stack.greens_temp.blocks[2]

        # IG * (R * Δ) * G[i:N:end, :]
        # real part
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] = IGR.re[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] -= IGR.im[m, 1] * G1.im[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] = IGR.re[m+N, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] -= IGR.im[m+N, 2] * G2.im[i, n]
        end

        # imaginary part
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] = IGR.im[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] += IGR.re[m, 1] * G1.im[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] = IGR.im[m+N, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] += IGR.re[m+N, 2] * G2.im[i, n]
        end

        # G = G - [...] part
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.re[m, n] = G1.re[m, n] - temp1.re[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.re[m, n] = G2.re[m, n] - temp2.re[m, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.im[m, n] = G1.im[m, n] - temp1.im[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.im[m, n] = G2.im[m, n] - temp2.im[m, n]
        end
        
        # update conf
        conf[i, slice] = x_new
    end
end

# TODO global (test this)
energy_boson(mc, ::AttractiveGHQHubbardModel, conf=nothing) = 0.0
function global_update(mc::DQMC, model::AttractiveGHQHubbardModel, temp_conf::AbstractArray)
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
function MonteCarlo.compress(mc::DQMC, ::AttractiveGHQHubbardModel, c)
    # converts (-2, -1, 1, 2) -> (10, 00, 01, 11)
    # first bit is value (0 -> 1, 1 -> 2), second is sign (0 -> -, 1 -> +)
    bools = [(abs(v) == 2, sign(v) == 1)[step] for v in c for step in (1, 2)]
    BitArray(bools)
end
MonteCarlo.compressed_conf_type(::Type{<: DQMC}, ::Type{<: AttractiveGHQHubbardModel}) = BitArray
function MonteCarlo.decompress(::DQMC, ::AttractiveGHQHubbardModel, c)
    map(1:2:length(c)) do i
        # (c[i] ? 2 : 1)       * (c[i+1] ? +1 : -1)
        (Int8(1) + Int8(c[i])) * (Int8(2) * Int8(c[i+1]) - Int8(1))
    end
end

function MonteCarlo.intE_kernel(mc, model::AttractiveGHQHubbardModel, G::GreensMatrix)
    N = length(lattice(model))
    output = zero(eltype(G.val))
    for i in 1:length(lattice(mc))
        output += (G.val[i, i] - 0.5) * (G.val[i+N, i+N] - 0.5)
    end
    -model.U * output
end


function save_model(
        file::JLDFile,
        m::AttractiveGHQHubbardModel,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "AttractiveGHQHubbardModel")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t", m.t)
    save_lattice(file, m.l, entryname * "/l")

    nothing
end

function _load(data, ::Val{:AttractiveGHQHubbardModel})
    l = _load(data["l"], to_tag(data["l"]))
    AttractiveGHQHubbardModel(
        mu = data["mu"],
        U = data["U"],
        t = data["t"],
        l = l
    )
end