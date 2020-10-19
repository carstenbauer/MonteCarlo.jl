################################################################################
### Measurements (constructors + kernels)
################################################################################



function checkflavors(model)
    if nflavors(model) != 2
        @warn(
            "CDC_measurement assumes 2 spin flavors, but $(nflavors(model)) " *
            "flavors have been found. The measurement might be incorrect."
        )
    end
    nothing
end



function greens_measurement(mc::DQMC, model::Model; mask_type = Diagonal, kwargs...)
    Measurement(mc, model, Greens, mask_type, greens_kernel; kwargs...)
end
# TODO remove one (depending on whether we keep Nothing or RawMask)
greens_kernel(mc, model, i, G) = G[i]
greens_kernel(mc, model, i, j, G) = G[i, j]



function occupation_measurement(mc::DQMC, model::Model; mask_type = OnSiteMask, kwargs...)
    Measurement(mc, model, Greens, mask_type, occupation_kernel; kwargs...)
end
occupation_kernel(mc, model, i, G) = 1 - G[i, j]



function CDC_measurement(mc::DQMC, model::Model; mask_type = DistanceMask, kwargs...)
    checkflavors(model)
    Measurement(mc, model, Greens, mask_type, cdc_kernel; kwargs...)
end
function cdc_kernel(mc, model, i, j, G)
    # Assume 1:N = spin up, N+1:2N = spin down
    # I[i, j] has been inlined where possible
    # G[i, j] = ⟨c_i c_j^†⟩     (I - G)[i, j] = ⟨c_j^† c_i⟩
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
    (I[j+N, i+N] - G[j+N, i+N]) *  G[i+N, j+N]
end



"""
    magnetization_measurement(mc, model, dir[; mask_type, kwargs...])

Returns the x, y or z magnetization measurement given `dir = :x`, `:y` or `:z`
respectively.

NOTE: 
We're skipping the multiplication by `-1im` during the measurement of the y 
magnetization. To get the correct result, multiply the final result by `-1im`.
"""
function magnetization_measurement(
        mc::DQMC, model::Model, dir::Symbol; mask_type = OnSiteMask, kwargs...
    )
    checkflavors(model)
    if     dir == :x; kernel = mx_kernel
    elseif dir == :y; kernel = my_kernel
    elseif dir == :z; kernel = mz_kernel
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    Measurement(mc, model, Greens, mask_type, kernel; kwargs...)
end
function mx_kernel(mc, model, i, G)
    N = length(lattice(model))
    -G[i+N, i] - G[i, i+N]
end
function my_kernel(mc, model, i, G)
    N = length(lattice(model))
    G[i+N, i] - G[i, i+N]
end
function mz_kernel(mc, model, i, G)
    N = length(lattice(model))
    G[i+N, i+N] - G[i, i]
end



function SDC_measurement(dqmc, model, dir::Symbol; mask_type = DistanceMask, kwargs...)
    checkflavors(model)
    if     dir == :x; kernel = sdc_x_kernel
    elseif dir == :y; kernel = sdc_y_kernel
    elseif dir == :z; kernel = sdc_z_kernel
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    Measurement(dqmc, model, Greens, mask_type, kernel; kwargs...)
end
function sdc_x_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    G[i+N, i] * G[j+N, j] - G[j+N, i] * G[i+N, j] +
    G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
    G[i, i+N] * G[j+N, j] + (I[j+N, i+N] - G[j+N, i+N]) * G[i, j] +
    G[i, i+N] * G[j, j+N] - G[j, i+N] * G[i, j+N]
end
function sdc_y_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    - G[i+N, i] * G[j+N, j] + G[j+N, i] * G[i+N, j] +
      G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
      G[i, i+N] * G[j+N, j] + (I[j+N, i+N] - G[j+N, i+N]) * G[i, j] -
      G[i, i+N] * G[j, j+N] + G[j, i+N] * G[i, j+N]
end
function sdc_z_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    (1 - G[i, i]) * (1 - G[j, j])         + (I[j, i] - G[j, i]) * G[i, j] -
    (1 - G[i, i]) * (1 - G[j+N, j+N])     + G[j+N, i] * G[i, j+N] -
    (1 - G[i+N, i+N]) * (1 - G[j, j])     + G[j, i+N] * G[i+N, j] +
    (1 - G[i+N, i+N]) * (1 - G[j+N, j+N]) + (1 - G[j+N, i+N]) * G[i+N, j+N]
end



function pc_measurement(dqmc, model;
        directions, mask_type = TODO{directions}, kwargs...
    )
    Measurement(dqmc, model, Greens, mask_type, kernel; kwargs...)
end
function pc_kernel(mc, model, src1, src2, trg1, trg2, G)
    N = length(lattice(model))
    # verified against ED for each (src1, src2, trg1, trg2)
    # G_{i, j}^{↑, ↑} G_{i+d, j+d}^{↓, ↓} - G_{i, j+d}^{↑, ↓} G_{i+d, j}^{↓, ↑}
    G[src1, src2] * G[trg1+N, trg2+N] - G[src1, trg2+N] * G[trg1+N, src2]
end



######################################



struct BosonEnergyMeasurement{OT} <: AbstractMeasurement
    obs::OT
end
function boson_energy_measurement(dqmc, model; capacity = _default_capacity(dqmc))
    BosonEnergyMeasurement(LogBinner(Float64, capacity=capacity))
end


function measure!(mask, m::BosonEnergyMeasurement, mc::DQMC, model, sweep, args...)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end