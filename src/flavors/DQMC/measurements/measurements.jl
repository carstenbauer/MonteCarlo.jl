# call stack:
# run/replay 
#   > calls apply(GreensRequirement, MaskRequirement, group, dqmc, model, sweep)
#       > calls measure(MaskRequirement, measurement, dqmc, model, sweep, GreensResults)
#           > calls apply!(MaskRequirement, measurement, dqmc, model, GreensResults)
#               > calls kernel(measurement, dqmc, model, MaskResult, GreensResults)

# Does:
# creates groups, runs simulation
#   > resolves GreensRequirement (calclates G, Gkk, Gkl or whatever)
#       > resolves sweep (skipping, frequency based measurements) and commits
#           > resolves Mask (to Greens indices) 
#               > calculate element from Wicks theorem

# TODO
# - rewrite run/replay with new measurements
# - add BosonEnergyMeasurement
# - deprecate old measurement types
# - _save/_load to autoconvert
# - combine masks to 4-site mask
# - deprecate RawMask
# - re-add Greens -> Occupation maybe?


struct DQMCMeasurement{
        GreensRequirement, MaskRequirement, F <: Function, OT, T
    } <: AbstractMeasurement
    kernel::F
    observable::OT
    output::T
end

function DQMCMeasurement{GR, MR}(kernel::FT, observable::OT, output::T) where {GR, MR, FT, OT, T}
    DQMCMeasurement{GR, MR, FT, OT, T}(kernel, observable, output)
end

function Measurement(
        dqmc, model, GreensRequirement, MaskRequirement, kernel::F;
        capacity = _default_capacity(dqmc), eltype = geltype(dqmc)
    ) where F
    mask = MaskRequirement(dqmc, model)
    shape = _get_shape(model, mask)
    obs = LogBinner(zeros(eltype, shape), capacity=capacity)
    temp = zeros(eltype, shape)
    DQMCMeasurement{GreensRequirement, MaskRequirement}(kernel, obs, temp)
end



################################################################################
### DQMCMeasurement utilities
################################################################################



function Base.show(io::IO, ::MIME"text/plain", m::DQMCMeasurement{GR, MR}) where {GR, MR}
    max = capacity(m.observable)
    current = length(m.observable)
    print(io, "[$current/$max] DQMCMeasurement{$GR, $MR}($(m.kernel))")
end

"""
    _default_capacity(mc::DQMC)

Returns 100_000 if the configuration measurement is not available or empty, and
the length of the configuration measurement if it is not.

This should be useful for `run` - `setup measurements` - `replay` workflow, 
returning the exact capacity needed.
"""
function _default_capacity(mc::DQMC)
    k = if isdefined(mc, :measurements)
        findfirst(v -> v isa ConfigurationMeasurement, mc.measurements)
    else
        nothing
    end
    if k === nothing
        return 100_000
    else
        N = length(mc.measurements[k].obs)
        return N == 0 ? 100_000 : N
    end
end

_get_shape(model, ::Nothing) = (length(lattice(model)),)
_get_shape(model, mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(model, mask::DistanceMask) = length(mask)


# Some type piracy to make things easier
Base.Nothing(::DQMC, ::Model) = nothing

# To identify the requirement of equal-time Greens functions
struct Greens end
Greens(::DQMC, ::Model)= Greens()
# maybe you want to some stuff to get custom indexing?
# How would this integrate with greens iterators?
# greens(mc::DQMC, model::Model) = greens(mc)

greens_requirement(::AbstractMeasurement) = Nothing
mask_requirement(::AbstractMeasurement) = Nothing
greens_requirement(::DQMCMeasurement{GR, MR}) where {GR, MR} = GR
mask_requirement(::DQMCMeasurement{GR, MR}) where {GR, MR} = MR


function generate_groups(mc, model, measurements)
    # resolving greens requirements
    grs = tuple(unique(map(greens_requirement, measurements))...)
    greens = map(req -> req(mc, model), grs)

    # resolving mask requirements
    mask_requirements = tuple(unique(map(mask_requirement, measurements))...)
    masks = map(Mask -> Mask(mc, model), mask_requirements)

    # Greens => ((mask, measurement), (mask, measurement), ...)
    map(eachindex(greens)) do i
        ms = filter(m -> grs[i] == greens_requirement(m), measurements)
        group = map(ms) do m
            MR = mask_requirement(m)
            i = findfirst(==(MR), mask_requirements)
            @assert i !== nothing
            (masks[i], m)
        end
        greens[i] => group
    end
end


function _save(file::JLDFile, m::DQMCMeasurement{GR, MR}, key::String) where {GR, MR}
    write(file, "$key/VERSION", 1)
    write(file, "$key/type", typeof(m))
    # for the future I guess
    # write(file, "$key/ID", "DQMCMeasurement")
    # write(file, "$key/GR", GR)
    # write(file, "$key/MR", MR)
    write(file, "$key/kernel", m.kernel)
    write(file, "$key/obs", m.obs)
    write(file, "$key/ouput", m.output)
end

function _load(data, ::Type{T}) where {T <: DQMCMeasurement}
    # for the future
    # DQMCMeasurement{data["GR"], data["MR"]}(
    #     data["kernel"], data["obs"], data["output"]
    # )
    T(data["kernel"], data["obs"], data["output"])
end



################################################################################
### Greens function related
################################################################################



function apply!(::Nothing, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    for (mask, measurement) in combined
        measure!(mask, measurement, mc, model, sweep)
    end
    nothing
end

function apply!(::Greens, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    G = greens(mc)
    for (mask, measurement) in combined
        measure!(mask, measurement, mc, model, sweep, G)
    end
    nothing
end

function apply!(iter::CombinedGreensIterator, combined::Vector{<: Tuple}, mc::DQMC, model, sweep)
    G00 = greens(mc)
    for (Gkk, Gkl) in iter
        for (mask, measurement) in combined
            measure!(mask, measurement, mc, model, sweep, G00, Gkk, Gkl)
        end
    end
    nothing
end



################################################################################
### measure!
################################################################################



function measure!(mask, measurement, mc::DQMC, model, sweep, args...)
    # ignore sweep
    apply!(mask, measurement, mc, model, args...)
    push!(measurement.observable, measurement.output)
    nothing
end



################################################################################
### Mask related
################################################################################


struct OnSiteMask end
OnSiteMask(mc, model) = OnSiteMask()


# map greens indices to output indices
# Basically the same as RawMask, maybe remove RawMask?
function apply!(::Nothing, measurement, mc::DQMC, model, args...)
    for i in 1:length(lattice(mc))
        measurement.output[i] = measurement.kernel(mc, model, i, args...)
    end
    nothing
end

function apply!(mask::RawMask, measurement, mc::DQMC, model, args...)
    for i in 1:size(mask, 1), j in 1:size(mask, 2)
        measurement.output[i, j] = measurement.kernel(mc, model, i, j, args...)
    end
    nothing
end

# map (src, trg) pairs to an index representing the distance vector between them
function apply!(mask::DistanceMask, measurement, mc::DQMC, model, args...)
    measurement.output .= zero(eltype(measurement.output))
    for (dir, src, trg) in getorder(mask)
        measurement.output[dir] += measurement.kernel(mc, model, src, trg, args...)
    end
    measurement.output ./= length(lattice(model))
    nothing
end

# Just the Diagonal (i.e. on-site)
function apply!(::OnSiteMask, measurement, mc::DQMC, model, args...)
    for i in 1:length(lattice(model))
        measurement.output[i] = measurement.kernel(mc, model, i, args...)
    end
    nothing
end

# TODO four index mask
# function apply!(::TODO, measurement, mc::DQMC, model, args...)
    # output .= zero(eltype(output))
    # # Compute   Δ_v(r_1, Δr_1) Δ_v^†(r_2, Δr_2)
    # # where we write r_2 = r_1 + Δr and sum over r_1
    # for (dir_idx, src1, src2) in getorder(mask)
    #     for (i, trg1) in getorder(rsm, src1)
    #         for (j, trg2) in getorder(rsm, src2)
    #             output[dir_idx, i, j] += kernel(mc, model, src1, src2, trg1, trg2, G)
    #         end
    #     end
    # end
#     nothing
# end


include("equal_time_measurements.jl")



################################################################################
#=


function PairingCorrelationMeasurement(
        mc::DQMC, model; 
        mask = DistanceMask(lattice(model)), 
        directions = 10,
        capacity = _default_capacity(mc)
    )
    mask isa RawMask && @error(
        "The Pairing Correlation Measurement will be extremely large with a RawMask!" *
        " (Estimate: $(ceil(Int64, log2(capacity))*3*length(lattice(model))^4*8 / 1024 / 1024)MB)"
    )
    rsm = RestrictedSourceMask(mask, directions)
    T = greenseltype(DQMC, model)
    shape = (length(mask), directions, directions)

    obs1 = LightObservable(
        LogBinner(zeros(T, shape), capacity=capacity),
        "Equal time pairing correlation matrix",
        "observables.jld",
        "etpc-s"
    )
    temp = zeros(T, shape)
    PairingCorrelationMeasurement(obs1, temp, mask, rsm)
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = length(lattice(model))
    G = greens(mc, model)
    # Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩
    #     = ⟨c_{i, ↑} c_{i+d, ↓} c_{j+d, ↓}^† c_{j, ↑}^†⟩
    #     = ⟨c_{i, ↑} c_{j, ↑}^†⟩ ⟨c_{i+d, ↓} c_{j+d, ↓}^†⟩ -
    #       ⟨c_{i, ↑} c_{j+d, ↓}^†⟩ ⟨c_{i+d, ↓} c_{j, ↑}^†⟩
    #     = G_{i, j}^{↑, ↑} G_{i+d, j+d}^{↓, ↓} - 
    #       G_{i, j+d}^{↑, ↓} G_{i+d, j}^{↓, ↑}

    # Doesn't require IG
    mask_kernel!(m, m.mask, m.rsm, G, G, _pc_s_wave_kernel, m.temp)
    push!(m.obs, m.temp ./ N)
end

function mask_kernel!(
        m::PairingCorrelationMeasurement,
        mask::DistanceMask, rsm::RestrictedSourceMask,
        IG, G, kernel::Function, output
    )
    output .= zero(eltype(output))
    # Compute   Δ_v(r_1, Δr_1) Δ_v^†(r_2, Δr_2)
    # where we write r_2 = r_1 + Δr and sum over r_1
    for (dir_idx, src1, src2) in getorder(mask)
        for (i, trg1) in getorder(rsm, src1)
            for (j, trg2) in getorder(rsm, src2)
                output[dir_idx, i, j] += kernel(IG, G, src1, src2, trg1, trg2)
            end
        end
    end
    output
end
function _pc_s_wave_kernel(IG, G, src1, src2, trg1, trg2)
    N = div(size(IG, 1), 2)
    # verified against ED for each (src1, src2, trg1, trg2)
    # G_{i, j}^{↑, ↑} G_{i+d, j+d}^{↓, ↓} - G_{i, j+d}^{↑, ↓} G_{i+d, j}^{↓, ↑}
    G[src1, src2] * G[trg1+N, trg2+N] - G[src1, trg2+N] * G[trg1+N, src2]
end
=#