################################################################################
### Pre measurement rework
################################################################################


struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function save_measurement(file::JLDFile, m::GreensMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: GreensMeasurement
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
end



struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function save_measurement(file::JLDFile, m::BosonEnergyMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: BosonEnergyMeasurement
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
end



struct OccupationMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function OccupationMeasurement(m::GreensMeasurement{<: LightObservable}; capacity=capacity(m.obs.B))
    N = size(m.obs)[1]
    o = LightObservable(
        LogBinner(zeros(Float64, N), capacity=capacity),
        "State resolved Occupation",
        "Observables.jld",
        "Occ"
    )

    N = min(length(o.B.count), length(m.obs.B.count))
    if m.obs.B.count[N] > 1
        throw(OverflowError("OccupationMeasurement does not have enough capacity!"))
    end
    N = findlast(>(1), m.obs.B.count)
    N === nothing && return OccupationMeasurement{typeof(o)}(o)
    
    for i in 1:N
        o.B.x_sum[i]  .= m.obs.B.count[i] .- diag(m.obs.B.x_sum[i])
        o.B.x2_sum[i] .= m.obs.B.count[i] .- 2diag(m.obs.B.x_sum[i]) .+ diag(m.obs.B.x2_sum[i])
        o.B.count[i]   = m.obs.B.count[i]
        o.B.compressors[i].value .= 1 .- diag(m.obs.B.compressors[i].value)
        o.B.compressors[i].switch = m.obs.B.compressors[i].switch
    end
    OccupationMeasurement{typeof(o)}(o)
end
function save_measurement(file::JLDFile, m::OccupationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: OccupationMeasurement
    if data["VERSION"] == 1
        data["type"](data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
end


_get_shape(model::Model) = (length(lattice(model)),)
_get_shape(mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(mask::DistanceMask) = length(mask)
mask(m::AbstractMeasurement) = m.mask



abstract type SpinOneHalfMeasurement <: AbstractMeasurement end



struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
end
function save_measurement(file::JLDFile, m::ChargeDensityCorrelationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: ChargeDensityCorrelationMeasurement
    if data["VERSION"] == 1
        OccupationMeasurement(data["obs"])
    else
        _load(data, AbstractMeasurement)
    end
end



struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: AbstractArray,
        ATy <: AbstractArray
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    tempy::ATy
end
function save_measurement(file::JLDFile, m::MagnetizationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", MagnetizationMeasurement)
    write(file, entryname * "/x", m.x)
    write(file, entryname * "/y", m.y)
    write(file, entryname * "/z", m.z)
    nothing
end
function _load(data, ::Type{T}) where T <: MagnetizationMeasurement
    if data["VERSION"] == 1
        x = data["x"]
        y = data["y"]
        z = data["z"]
        temp = similar(x.B.x_sum[1])
        tempy = similar(y.B.x_sum[1])
        data["type"](x, y, z, temp, tempy)
    else
        _load(data, AbstractMeasurement)
    end
end



struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    mask::MT
end



struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: DistanceMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
    rsm::RestrictedSourceMask
end



to_logbinner(B::LogBinner) = B
to_logbinner(B::LightObservable) = B.B


function mask2iter(mask, iter::EachSitePairByDistance)
    # A[idxs] should output iter order
    # idxs = [mask_idx@iter1, mask_idx@iter2, ...]
    idxs = Vector{Int64}(undef, ndirections(iter))
    
    N = sqrt(length(iter))
    @assert N ≈ round(Int64, N)
    N = round(Int64, N)
    mask_idxs = Matrix{Int64}(undef, N, N)
    for (idx, src, trg) in getorder(mask)
        mask_idxs[src, trg] = idx
    end

    for i in 1:ndirections(iter)
        pairs = in_direction(iter, i)
        x = first(pairs)
        idx = mask_idxs[x[1], x[2]]
        for (src, trg) in pairs
            if mask_idxs[src, trg] != idx
                error("Mapping is not unqiue: $(mask_idxs[src, trg]) != $(idx)")
            end
        end
        idxs[i] = idx
    end

    idxs
end

function reorder(input::LogBinner, idxs::Vector)
    B = deepcopy(input)
    for lvl in eachindex(B.compressors)
        permute!(B.compressors[lvl].value, idxs)
        permute!(B.x_sum[lvl], idxs)
        permute!(B.x2_sum[lvl], idxs)
    end
    B
end

function reorder(input::LogBinner, idxs1::Vector, idxs2::Vector, idxs3::Vector)
    B = deepcopy(input)
    for lvl in eachindex(B.compressors)
        B.compressors[lvl].value = B.compressors[lvl].value[idxs1, idxs2, idxs3]
        B.x_sum[lvl] = B.x_sum[lvl][idxs1, idxs2, idxs3]
        B.x2_sum[lvl] = B.x2_sum[lvl][idxs1, idxs2, idxs3]
    end
    B
end



function DQMCMeasurement(mc::DQMC, m::GreensMeasurement)
    obs = to_logbinner(m.obs)
    temp = mean(obs)
    DQMCMeasurement{Greens, Nothing}(greens_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::BosonEnergyMeasurement)
    obs = to_logbinner(m.obs)
    DQMCMeasurement{Nothing, Nothing}(energy_boson, obs, 0.0)
end

function DQMCMeasurement(mc::DQMC, m::OccupationMeasurement)
    obs = to_logbinner(m.obs)
    temp = mean(obs)
    DQMCMeasurement{Greens, EachSiteAndFlavor}(occupation_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::MagnetizationMeasurement)
    x = to_logbinner(m.x)
    y = to_logbinner(m.y)
    z = to_logbinner(m.z)

    return (
        DQMCMeasurement{Greens, EachSite}(mx_kernel, x, mean(x)),
        DQMCMeasurement{Greens, EachSite}(my_kernel, y, mean(y)),
        DQMCMeasurement{Greens, EachSite}(mz_kernel, z, mean(z))
    )
end

function DQMCMeasurement(mc::DQMC, m::ChargeDensityCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)
    obs = reorder(to_logbinner(m.obs), idxs)
    temp = mean(obs)
    DQMCMeasurement{Greens, EachSitePairByDistance}(cdc_kernel, obs, temp)
end

function DQMCMeasurement(mc::DQMC, m::SpinDensityCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)

    x = reorder(to_logbinner(m.x), idxs)
    y = reorder(to_logbinner(m.y), idxs)
    z = reorder(to_logbinner(m.z), idxs)
    temp = mean(x)
    tempy = mean(y)
    return (
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_x_kernel, x, temp),
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_y_kernel, y, tempy),
        DQMCMeasurement{Greens, EachSitePairByDistance}(sdc_z_kernel, z, temp)
    )
end

function DQMCMeasurement(mc::DQMC, m::PairingCorrelationMeasurement)
    iter = EachSitePairByDistance(mc, mc.model)
    idxs = mask2iter(m.mask, iter)
    pre = to_logbinner(m.obs)
    temp = mean(pre)
    D = size(temp, 2)
    obs = reorder(pre, idxs, idxs[1:D], idxs[1:D])
    DQMCMeasurement{Greens, EachLocalQuadByDistance{D}}(pc_kernel, obs, mean(obs))
end



################################################################################
### From saving measurement function names
################################################################################



greens_kernel(mc, model, G::AbstractArray) = G


occupation_kernel(mc, model, i::Integer, G::AbstractArray) = 1 - G[i, i]



function cdc_kernel(mc, model, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    N = length(lattice(mc))
    # ⟨n↑n↑⟩
    (1 - G[i, i])       * (1 - G[j, j]) +
    (I[j, i] - G[j, i]) * G[i, j] +
    # ⟨n↑n↓⟩
    (1 - G[i, i]) * (1 - G[j+N, j+N]) -
    G[j+N, i]     * G[i, j + N] +
    # ⟨n↓n↑⟩
    (1 - G[i+N, i+N]) * (1 - G[j, j]) -
    G[j, i+N]         * G[i+N, j] +
    # ⟨n↓n↓⟩
    (1 - G[i+N, i+N])           * (1 - G[j+N, j+N]) +
    (I[j, i] - G[j+N, i+N]) *  G[i+N, j+N]
end

function cdc_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    # ⟨n↑(l)n↑⟩
    (1 - Gll[i, i]) * (1 - G00[j, j]) -
    G0l[j, i] * Gl0[i, j] +
    # ⟨n↑n↓⟩
    (1 - Gll[i, i]) * (1 - G00[j+N, j+N]) -
    G0l[j+N, i] * Gl0[i, j+N] +
    # ⟨n↓n↑⟩
    (1 - Gll[i+N, i+N]) * (1 - G00[j, j]) -
    G0l[j, i+N] * Gl0[i+N, j] +
    # ⟨n↓n↓⟩
    (1 - Gll[i+N, i+N]) * (1 - G00[j+N, j+N]) -
    G0l[j+N, i+N] * Gl0[i+N, j+N]
end



function mx_kernel(mc, model, i, G::AbstractArray)
    N = length(lattice(model))
    -G[i+N, i] - G[i, i+N]
end

function my_kernel(mc, model, i, G::AbstractArray)
    N = length(lattice(model))
    G[i+N, i] - G[i, i+N]
end

function mz_kernel(mc, model, i, G::AbstractArray)
    N = length(lattice(model))
    G[i+N, i+N] - G[i, i]
end



function sdc_x_kernel(mc, model, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    N = length(lattice(model))
    G[i+N, i] * G[j+N, j] - G[j+N, i] * G[i+N, j] +
    G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
    G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] +
    G[i, i+N] * G[j, j+N] - G[j, i+N] * G[i, j+N]
end
function sdc_x_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    Gll[i+N, i] * G00[j+N, j] - G0l[j+N, i] * Gl0[i+N, j] +
    Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
    Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] +
    Gll[i, i+N] * G00[j, j+N] - G0l[j, i+N] * Gl0[i, j+N]
end

function sdc_y_kernel(mc, model, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    N = length(lattice(model))
    - G[i+N, i] * G[j+N, j] + G[j+N, i] * G[i+N, j] +
      G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
      G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] -
      G[i, i+N] * G[j, j+N] + G[j, i+N] * G[i, j+N]
end
function sdc_y_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    - Gll[i+N, i] * G00[j+N, j] + G0l[j+N, i] * Gl0[i+N, j] +
      Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
      Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] -
      Gll[i, i+N] * G00[j, j+N] + G0l[j, i+N] * Gl0[i, j+N]
end

function sdc_z_kernel(mc, model, ij::NTuple{2}, G::AbstractArray)
    i, j = ij
    N = length(lattice(model))
    (1 - G[i, i]) * (1 - G[j, j])         + (I[j, i] - G[j, i]) * G[i, j] -
    (1 - G[i, i]) * (1 - G[j+N, j+N])     + G[j+N, i] * G[i, j+N] -
    (1 - G[i+N, i+N]) * (1 - G[j, j])     + G[j, i+N] * G[i+N, j] +
    (1 - G[i+N, i+N]) * (1 - G[j+N, j+N]) + (I[j, i] - G[j+N, i+N]) * G[i+N, j+N]
end
function sdc_z_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    (1 - Gll[i, i])     * (1 - G00[j, j])     - G0l[j, i] * Gl0[i, j] -
    (1 - Gll[i, i])     * (1 - G00[j+N, j+N]) + G0l[j+N, i] * Gl0[i, j+N] -
    (1 - Gll[i+N, i+N]) * (1 - G00[j, j])     + G0l[j, i+N] * Gl0[i+N, j] +
    (1 - Gll[i+N, i+N]) * (1 - G00[j+N, j+N]) - G0l[j+N, i+N] * Gl0[i+N, j+N]
end




function pc_kernel(mc, model, sites::NTuple{4}, G::AbstractArray)
    pc_kernel(mc, model, sites, (G, G, G, G))
end
function pc_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    # Δ_v(src1, trg1)(τ) Δ_v^†(src2, trg2)(0)
    # G_{i, j}^{↑, ↑}(τ, 0) G_{i+d, j+d'}^{↓, ↓}(τ, 0) - 
    # G_{i, j+d'}^{↑, ↓}(τ, 0) G_{i+d, j}^{↓, ↑}(τ, 0)
    Gl0[src1, src2] * Gl0[trg1+N, trg2+N] - Gl0[src1, trg2+N] * Gl0[trg1+N, src2]
end

function pc_alt_kernel(mc, model, sites::NTuple{4}, G::AbstractArray)
    pc_alt_kernel(mc, model, sites, (G, G, G, G))
end
function pc_alt_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    # Δ_v^†(src1, trg1)(τ) Δ_v(src2, trg2)(0)
    # (I-G)_{j, i}^{↑, ↑}(0, τ) (I-G)_{j+d', i+d}^{↓, ↓}(0, τ) - 
    # (I-G)_{j, i+d}^{↑, ↓}(0, τ) G_{j+d', i}^{↓, ↑}(0, τ)
    (I[trg2, trg1] - G0l[trg2+N, trg1+N]) * (I[src2, src1] - G0l[src2, src1]) -
    (I[src2, trg1] - G0l[src2, trg1+N]) * (I[trg2, src1] - G0l[trg2+N, src1])
end

function pc_combined_kernel(mc, model, sites::NTuple{4}, G)
    # Δ^† Δ + Δ Δ^†
    pc_kernel(mc, model, sites, G) + pc_alt_kernel(mc, model, sites, G)
end


function pc_ref_kernel(mc, model, sites::NTuple{4}, G::AbstractArray)
    # Δ^† Δ + Δ Δ^† but ↑ and ↓ are swapped
    pc_ref_kernel(mc, model, sites, (G, G, G, G))
end
function pc_ref_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    Gl0[src1+N, src2+N] * Gl0[trg1, trg2] - 
    Gl0[src1+N, trg2] * Gl0[trg1, src2+N] +
    (I[trg2, trg1] - G0l[trg2, trg1]) * (I[src2, src1] - G0l[src2+N, src1+N]) -
    (I[src2, trg1] - G0l[src2+N, trg1]) * (I[trg2, src1] - G0l[trg2, src1+N])
end



function cc_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4})
    # Computes
    # ⟨j_{t2-s2}(s2, l) j_{t1-s1}(s1, 0)⟩
    # where t2-s2 (t1-s1) is usually a NN vector/jump, and
    # j_{t2-s2}(s2, l) = i \sum_σ [T_{ts} c_t^†(l) c_s(τ) - T_{st} c_s^†(τ) c_t(τ)]
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    T = mc.stack.hopping_matrix
    output = zero(eltype(G00))

    # Iterate through (spin up, spin down)
    for σ1 in (0, N), σ2 in (0, N)
        s1 = src1 + σ1; t1 = trg1 + σ1
        s2 = src2 + σ2; t2 = trg2 + σ2
        # Note: if H is real and Hermitian, T can be pulled out and the I's cancel
        # Note: This matches crstnbr/dqmc if H real, Hermitian
        # Note: I for G0l and Gl0 auto-cancels
        output -= (
                T[t2, s2] * (I[s2, t2] - Gll[s2, t2]) - 
                T[s2, t2] * (I[t2, s2] - Gll[t2, s2])
            ) * (
                T[t1, s1] * (I[s1, t1] - G00[s1, t1]) - 
                T[s1, t1] * (I[t1, s1] - G00[t1, s1])
            ) +
            - T[t2, s2] * T[t1, s1] * G0l[s1, t2] * Gl0[s2, t1] +
            + T[t2, s2] * T[s1, t1] * G0l[t1, t2] * Gl0[s2, s1] +
            + T[s2, t2] * T[t1, s1] * G0l[s1, s2] * Gl0[t2, t1] +
            - T[s2, t2] * T[s1, t1] * G0l[t1, s2] * Gl0[t2, s1]
    end

    # OLD PARTIALLY OUTDATED

    # This should compute 
    # ⟨j_{trg1-src1}(src1, τ) j_{trg2-src2}(src2, 0)⟩
    # where (trg-src) picks a direction (e.g. NN directions)
    # and (src1-src2) is the distance vector that the Fourier transform applies to
    # From dos Santos: Introduction to Quantum Monte Carlo Simulations
    # j_{trg-src}(src, τ) = it \sum\sigma (c^\dagger(trg,\sigma, \tau) c(src, \sigma, \tau) - c^\dagger(src, \sigma, \tau) c(trg, \sigma \tau))
    # where i -> src, i+x -> trg as a generalization
    # and t is assumed to be hopping matrix element, generalizing to
    # = i \sum\sigma (T[trg, src] c^\dagger(trg,\sigma, \tau) c(src, \sigma, \tau) - T[src, trg] c^\dagger(src, \sigma, \tau) c(trg, \sigma \tau))
    
            # Why no I? - delta_0l = 0
            # T[t1, s1] * T[t2, s2] * (I[s2, t1] - G0l[s2, t1]) * Gl0[s1, t2] -
            # T[s1, t1] * T[t2, s2] * (I[s2, s1] - G0l[s2, s1]) * Gl0[t1, t2] -
            # T[t1, s1] * T[s2, t2] * (I[t2, t1] - G0l[t2, t1]) * Gl0[s1, s2] +
            # T[s1, t1] * T[s2, t2] * (I[t2, s1] - G0l[t2, s1]) * Gl0[t1, s2]

        # Uncompressed Wicks expansion
        # output += T[t1, s1] * T[t2, s2] *
        #     ((I[s1, t1] - Gll[s1, t1]) * (I[s2, t2] - G00[s2, t2]) +
        #     (I[s2, t1] - G0l[s2, t1]) * Gl0[s1, t2])
        # output -= T[s1, t1] * T[t2, s2] *
        #     ((I[t1, s1] - Gll[t1, s1]) * (I[s2, t2] - G00[s2, t2]) +
        #     (I[s2, s1] - G0l[s2, s1]) * Gl0[t1, t2])
        # output -= T[t1, s1] * T[s2, t2] *
        #     ((I[s1, t1] - Gll[s1, t1]) * (I[t2, s2] - G00[t2, s2]) +
        #     (I[t2, t1] - G0l[t2, t1]) * Gl0[s1, s2])
        # output += T[s1, t1] * T[s2, t2] *
        #     ((I[t1, s1] - Gll[t1, s1]) * (I[t2, s2] - G00[t2, s2]) +
        #     (I[t2, s1] - G0l[t2, s1]) * Gl0[t1, s2])
    output
end



@inline function nonintE_kernel(mc, model, G::AbstractArray)
    # <T> = \sum Tji * (Iij - Gij) = - \sum Tji * (Gij - Iij)
    T = mc.stack.hopping_matrix
    nonintE(T, G)
end



function totalE_kernel(mc, model, G::AbstractArray)
    nonintE_kernel(mc, model, G) + intE_kernel(mc, model, G)
end



_li_dimensionality(li::Type{Nothing}) = nothing
_li_dimensionality(li::Type{EachSite}) = 1
_li_dimensionality(li::Type{EachSiteAndFlavor}) = 1
_li_dimensionality(li::Type{OnSite}) = 1
_li_dimensionality(li::Type{EachSitePair}) = 2
_li_dimensionality(li::Type{EachSitePairByDistance}) = 2
_li_dimensionality(li::Type{EachLocalQuadBySyncedDistance}) = 4
_li_dimensionality(li::LatticeIterationWrapper{T}) where T = _li_dimensionality(T)

function function2code(model, li, gi, func)
    N = _li_dimensionality(li)
    if model isa HubbardModelAttractive
        library = _measurement_kernel_code2
    else
        library = _measurement_kernel_code
    end

    if gi == Greens
        if N == Nothing
            func == greens_kernel && return library.greens
            func == boson_energy && return library.boson_energy
            func == nonintE_kernel && return library.noninteracting_energy
            if model isa HubbardModelAttractive
                func == intE_kernel && return quote
                    (mc, model::HubbardModelAttractive, G::AbstractArray) -> - model.U * sum((diag(G) .- 0.5).^2)
                end
                func == totalE_kernel && return quote
                    (mc, model::HubbardModelAttractive, G::AbstractArray) -> begin
                        nonintE(mc.stack.hopping_matrix, G) - model.U * sum((diag(G) .- 0.5).^2)
                end
            end
            elseif model isa HubbardModelRepulsive
                func == intE_kernel && return quote
                    (mc, model::HubbardModelRepulsive, G::AbstractArray) -> 
                        model.U * sum((diag(G.blocks[1]) .- 0.5) .* (diag(G.blocks[2]) .- 0.5))
                end
                func == totalE_kernel && return quote
                    (mc, model::HubbardModelRepulsive, G::AbstractArray) -> begin
                        nonintE(mc.stack.hopping_matrix, G) +
                        model.U * sum((diag(G.blocks[1]) .- 0.5) .* (diag(G.blocks[2]) .- 0.5))
                    end
                end
            end
        elseif N == 1
            func == occupation_kernel && return library.occupation
            func == mx_kernel && return library.Mx
            func == my_kernel && return library.My
            func == mz_kernel && return library.Mz
        elseif N == 2
            func == cdc_kernel && return library.equal_time_charge_density
            func == sdc_x_kernel && return library.equal_time_spin_density_x
            func == sdc_y_kernel && return library.equal_time_spin_density_y
            func == sdc_z_kernel && return library.equal_time_spin_density_z
        elseif N == 4
            func == pc_kernel && return library.equal_time_pairing
        end
    elseif gi == CombinedGreensIterator
        if N == 2
            func == cdc_kernel && return library.unequal_time_charge_density
            func == sdc_x_kernel && return library.unequal_time_spin_density_x
            func == sdc_x_kernel && return library.unequal_time_spin_density_y
            func == sdc_x_kernel && return library.unequal_time_spin_density_z
        elseif N == 4
            func == pc_kernel && return library.unequal_time_pairing
            func == cc_kernel && return library.current_current
        end
    end

    return Expr(:NA)
end