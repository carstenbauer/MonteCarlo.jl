################################################################################
### Lattice via LatticePhysics
################################################################################



using LinearAlgebra, LatticePhysics, LatPhysUnitcellLibrary

function LatPhysUnitcellLibrary.getUnitcellSquare(
            unitcell_type  :: Type{U},
            implementation :: Val{17}
        ) :: U where {LS,LB,S<:AbstractSite{LS,2},B<:AbstractBond{LB,2}, U<:AbstractUnitcell{S,B}}

    # return a new Unitcell
    return newUnitcell(
        # Type of the unitcell
        U,

        # Bravais lattice vectors
        [[1.0, +1.0], [1.0, -1.0]],
        
        # Sites
        S[
            newSite(S, [0.0, 0.0], getDefaultLabelN(LS, 1)),
            newSite(S, [0.0, 1.0], getDefaultLabelN(LS, 2))
        ],

        # Bonds
        B[
            # NN, directed
            # bonds from ref plot, π/4 weight for spin up
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (0, 1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 1), (-1, 0)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (+1, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 1), (0, 0)),

            # NN reversal
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (0, -1)),
            newBond(B, 2, 1, getDefaultLabelN(LB, 2), (+1, 0)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (-1, +1)),
            newBond(B, 1, 2, getDefaultLabelN(LB, 2), (0, 0)),
            
            # NNN
            # positive weight (we need forward and backward facing bonds here too)
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (+1, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 3), (-1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, +1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 3), (0, -1)),
            # negative weight
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, +1)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 4), (0, -1)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (+1, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 4), (-1, 0)),
            
            # Fifth nearest neighbors
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, 2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, 2)),  
            # backwards facing bonds
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (-2, 0)),
            newBond(B, 1, 1, getDefaultLabelN(LB, 5), (0, -2)),
            newBond(B, 2, 2, getDefaultLabelN(LB, 5), (0, -2)), 
        ]
    )
end



################################################################################
### Model implementation
################################################################################



using MonteCarlo
using MonteCarlo: StructArray, CMat64, CVec64
using MonteCarlo: AbstractLattice, @bm, HubbardConf, @turbo, BlockDiagonal, vmul!, conf


"""
t5 = 0  <=> F = 0.2
t5 = (1 - sqrt(2)) / 4 <=> F = 0.009
"""
MonteCarlo.@with_kw_noshow struct HBCModel{LT<:AbstractLattice} <: HubbardModel
    # parameters with defaults based on paper
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0 / sqrt(2.0)
    t5::Float64 = (1 - sqrt(2)) / 4

    # lattice
    l::LT

    # two fermion flavors (up, down)
    flv::Int = 2
    
    # temp storage to avoid allocations in propose_local and accept_local
    IG::CMat64  = StructArray(Matrix{ComplexF64}(undef, length(l), 2))
    IGR::CMat64 = StructArray(Matrix{ComplexF64}(undef, length(l), 2))
    R::Diagonal{ComplexF64, CVec64} = Diagonal(StructArray(Vector{ComplexF64}(undef, 2)))
end


# Constructors
HBCModel(params::Dict{Symbol}) = HBCModel(; params...)
HBCModel(params::NamedTuple) = HBCModel(; params...)
function HBCModel(lattice::AbstractLattice; kwargs...)
    HBCModel(l = lattice; kwargs...)
end
function HBCModel(L, dims; kwargs...)
    l = choose_lattice(HBCModel, dims, L)
    HBCModel(l = l; kwargs...)
end


# type information for DQMC stack
MonteCarlo.hoppingeltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.hopping_matrix_type(::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}
MonteCarlo.greenseltype(::Type{DQMC}, ::HBCModel) = ComplexF64
MonteCarlo.greens_matrix_type( ::Type{DQMC}, ::HBCModel) = BlockDiagonal{ComplexF64, 2, CMat64}


# cosmetics
import Base.summary
import Base.show
Base.summary(model::HBCModel) = "Hofmann Berg Hubbard model"
function Base.show(io::IO, model::HBCModel)
    print(io, "Hofmann Berg Hubbard model, $(length(model.l)) sites")
end
Base.show(io::IO, m::MIME"text/plain", model::HBCModel) = print(io, model)


# Convenience
@inline MonteCarlo.parameters(m::HBCModel) = (N = length(m.l), t1 = m.t1, t2 = m.t2, t5 = m.t5, U = -m.U, mu = m.mu)


# generate hopping matrix
function MonteCarlo.hopping_matrix(mc::DQMC, m::HBCModel{<: LatPhysLattice})
    # number of sites
    N = length(m.l)

    # spin up and spin down blocks of T
    tup = diagm(0 => fill(-ComplexF64(m.mu), N))
    tdown = diagm(0 => fill(-ComplexF64(m.mu), N))

    # positive and negative prefactors for t1, t2
    t1p = m.t1 * cis(+pi/4) # ϕ_ij^↑ = + π/4
    t1m = m.t1 * cis(-pi/4) # ϕ_ij^↓ = - π/4
    t2p = + m.t2
    t2m = - m.t2
    
    for b in bonds(m.l.lattice)
        # NN paper direction
        if b.label == 1 
            tup[b.from, b.to]   = - t1p
            tdown[b.from, b.to] = - t1m
        
        # NN reverse direction
        elseif b.label == 2
            tup[b.from, b.to]   = - t1m
            tdown[b.from, b.to] = - t1p
            
        # NNN solid bonds
        elseif b.label == 3
            tup[b.from, b.to]   = - t2p
            tdown[b.from, b.to] = - t2p

        # NNN dashed bonds
        elseif b.label == 4
            tup[b.from, b.to]   = - t2m
            tdown[b.from, b.to] = - t2m

        # Fifth nearest neighbors
        else
            tup[b.from, b.to]   = - m.t5
            tdown[b.from, b.to] = - m.t5
        end
    end

    return BlockDiagonal(StructArray(tup), StructArray(tdown))
end


"""
Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline @bm function MonteCarlo.interaction_matrix_exp!(mc::DQMC, m::HBCModel,
            result::Diagonal, conf::HubbardConf, slice::Int, power::Float64=1.)
    dtau = mc.parameters.delta_tau
    lambda = acosh(exp(0.5 * m.U * dtau))
    N = length(lattice(m))
    
    # spin up block
    @inbounds for i in 1:N
        result.diag[i] = exp(sign(power) * lambda * conf[i, slice])
    end

    # spin down block
    @inbounds for i in 1:N
        result.diag[N+i] = exp(sign(power) * lambda * conf[i, slice])
    end
    nothing
end



@inline @bm function MonteCarlo.propose_local(
        mc::DQMC, model::HBModel, i::Int, slice::Int, conf::HubbardConf
    )
    N = length(model.l)
    G = mc.stack.greens
    Δτ = mc.parameters.delta_tau
    R = model.R

    α = acosh(exp(0.5Δτ * model.U))
    ΔE_boson = -2.0α * conf[i, slice]
    Δ = exp(ΔE_boson) - 1.0

    # Unrolled R = I + Δ * (I - G)
    # up-up term
    R.diag.re[1] = 1.0 + Δ * (1.0 - G.blocks[1].re[i, i])
    R.diag.im[1] = - Δ * G.blocks[1].im[i, i]
    # down-down term
    R.diag.re[2] = 1.0 + Δ * (1.0 - G.blocks[2].re[i, i])
    R.diag.im[2] = - Δ * G.blocks[2].im[i, i]

    # Calculate "determinant"
    detratio = ComplexF64(
        R.diag.re[1] * R.diag.re[2] - R.diag.im[1] * R.diag.im[2],
        R.diag.re[1] * R.diag.im[2] + R.diag.im[1] * R.diag.re[2]
    )
    
    return detratio, ΔE_boson, Δ
end

@inline @bm function MonteCarlo.accept_local!(
        mc::DQMC, model::HBModel, i::Int, slice::Int, conf::HubbardConf, 
        detratio, ΔE_boson, Δ)

    @bm "accept_local (init)" begin
        N = length(model.l)
        G = mc.stack.greens
        IG = model.IG
        IGR = model.IGR
        R = model.R
    end
    
    # compute R⁻¹ Δ, using that R is Diagonal, Δ is Number
    # using Δ / (a + ib) = Δ / (a^2 + b^2) * (a - ib)
    @bm "accept_local (inversion)" begin
        f = Δ / (R.diag.re[1]^2 + R.diag.im[1]^2)
        R.diag.re[1] = +f * R.diag.re[1]
        R.diag.im[1] = -f * R.diag.im[1]
        f = Δ / (R.diag.re[2]^2 + R.diag.im[2]^2)
        R.diag.re[2] = +f * R.diag.re[2]
        R.diag.im[2] = -f * R.diag.im[2]
    end

    # Compute (I - G) R^-1 Δ
    # Note IG is reduced to non-zero entries. Full IG would be
    # (I-G)[:, i]        0
    #     0         (I-G)[:, i+N]
    # our IG is [(I-G)[:, i]  (I-G)[:, i+N]]
    @bm "accept_local (IG, R)" begin
        # Calculate IG = I - G (relevant entries only)
        @turbo for m in axes(IG, 1)
            IG.re[m, 1] = -G.blocks[1].re[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.re[m, 2] = -G.blocks[2].re[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.im[m, 1] = -G.blocks[1].im[m, i]
        end
        @turbo for m in axes(IG, 1)
            IG.im[m, 2] = -G.blocks[2].im[m, i]
        end
        IG.re[i, 1] += 1.0
        IG.re[i, 2] += 1.0
        
        # Calculate IGR = IG * R where R = R⁻¹ Δ from the 
        # previous calculation (relevant entries only)
        # spin up-up block 
        @turbo for m in axes(IG, 1)
            IGR.re[m, 1] = IG.re[m, 1] * R.diag.re[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.re[m, 1] -= IG.im[m, 1] * R.diag.im[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 1] = IG.re[m, 1] * R.diag.im[1]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 1] += IG.im[m, 1] * R.diag.re[1]
        end
        
        # spin down-down block
        @turbo for m in axes(IG, 1)
            IGR.re[m, 2] = IG.re[m, 2] * R.diag.re[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.re[m, 2] -= IG.im[m, 2] * R.diag.im[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 2] = IG.re[m, 2] * R.diag.im[2]
        end
        @turbo for m in axes(IG, 1)
            IGR.im[m, 2] += IG.im[m, 2] * R.diag.re[2]
        end
    end

    # Update G according to G = G - (I - G)[:, i:N:end] * R⁻¹ * Δ * G[i:N:end, :]
    # We already have IG = (I - G)[:, i:N:end] * R⁻¹ * Δ
    @bm "accept_local (finalize computation)" begin
        # get blocks to write less
        G1 = G.blocks[1]
        G2 = G.blocks[2]
        temp1 = mc.stack.greens_temp.blocks[1]
        temp2 = mc.stack.greens_temp.blocks[2]

        # compute temp = IG[:, i:N:end] * G[i:N:end, :]
        # spin up-up block
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] = IGR.re[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.re[m, n] -= IGR.im[m, 1] * G1.im[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] = IGR.im[m, 1] * G1.re[i, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            temp1.im[m, n] += IGR.re[m, 1] * G1.im[i, n]
        end
        
        # spin down-down block
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] = IGR.re[m, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.re[m, n] -= IGR.im[m, 2] * G2.im[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] = IGR.im[m, 2] * G2.re[i, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            temp2.im[m, n] += IGR.re[m, 2] * G2.im[i, n]
        end

        # Calculate G = G - temp
        # spin up-up block
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.re[m, n] = G1.re[m, n] - temp1.re[m, n]
        end
        @turbo for m in axes(G1, 1), n in axes(G1, 2)
            G1.im[m, n] = G1.im[m, n] - temp1.im[m, n]
        end
        
        # spin down-down block
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.re[m, n] = G2.re[m, n] - temp2.re[m, n]
        end
        @turbo for m in axes(G2, 1), n in axes(G2, 2)
            G2.im[m, n] = G2.im[m, n] - temp2.im[m, n]
        end

        # Update configuration
        conf[i, slice] *= -1
    end

    nothing
end


# this enables boson energy measurements and global updates
@inline function MonteCarlo.energy_boson(mc::DQMC, m::HBCModel, hsfield = conf(mc))
    dtau = mc.parameters.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end


# to save only what is necessary and not rely on the model type remaining the same
function MonteCarlo.save_model(
        file::MonteCarlo.JLDFile, m::HBCModel,
        entryname::String="Model"
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "HBCModel")

    write(file, entryname * "/mu", m.mu)
    write(file, entryname * "/U", m.U)
    write(file, entryname * "/t1", m.t1)
    write(file, entryname * "/t2", m.t2)
    write(file, entryname * "/t5", m.t5)
    MonteCarlo.save_lattice(file, m.l, entryname * "/l")
    write(file, entryname * "/flv", m.flv)

    nothing
end

# I used HBModel as a type name and tag before. Adding this will alow the old 
# tag to still work
MonteCarlo._load(data, ::Val{:HBModel}) = MonteCarlo._load(data, Val(:HBCModel))
function MonteCarlo._load(data, ::Val{:HBCModel})
    if !(data["VERSION"] == 1)
        throw(ErrorException("Failed to load HBCModel version $(data["VERSION"])"))
    end

    l = MonteCarlo._load(data["l"], MonteCarlo.to_tag(data["l"]))
    HBCModel(
        mu = data["mu"],
        U = data["U"],
        t1 = data["t1"],
        t2 = data["t2"],
        t5 = data["t5"],
        l = l,
        flv = data["flv"]
    )
end



################################################################################
### Measurement kernels
################################################################################



MonteCarlo.checkflavors(model::HBCModel) = MonteCarlo.checkflavors(model, 2)

function MonteCarlo.intE_kernel(mc, model::HBCModel, G::GreensMatrix)
    # ⟨U (n↑ - 1/2)(n↓ - 1/2)⟩ = ... 
    # = U [G↑↑ G↓↓ - G↓↑ G↑↓ - 0.5 G↑↑ - 0.5 G↓↓ + G↑↓ + 0.25]
    # = U [(G↑↑ - 1/2)(G↓↓ - 1/2) + G↑↓(1 + G↑↓)]
    # with up-up = down-down and up-down = 0
    - model.U * sum((diag(G.val) .- 0.5).^2)
end