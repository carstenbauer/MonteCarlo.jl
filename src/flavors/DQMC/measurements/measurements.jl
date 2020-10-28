################################################################################
### Measurements (constructors + kernels)
################################################################################



# Notes on math and stuff:
#=
- Use Wicks theorem to go from one expectation value with 2N operators to 
  N expectation values with 2 operators each
- use G_{ij} = ⟨c_i c_j^†⟩
- use ⟨c_i^† c_j⟩ = δ_{ij} - ⟨c_j c_i^†⟩ = (I - G)_{ji}
- we define G[i,j] as spin up, G[i+N,j+N] as spin down (N = number of sites)
- we used (I - G)[i, j+N] = G[i, j+N] etc when possible
- Gl0_{i,j} = ⟨c_i(l) c_j(0)^†⟩
- ⟨c_i(0) c_j(l)^†⟩ = ⟨(c_j(l) c_i(0)^†)^†⟩ = (⟨c_j(l) c_i(0)^†⟩)^† 
                    = Gl0_{j,i}^† = Gl0_{j,i}^*
=#
function checkflavors(model, N=2)
    if nflavors(model) != N
        @warn(
            "$N flavors are required, but $(nflavors(model)) have been found"
        )
    end
    nothing
end



# This has lattice_iteratorator = Nothing, because it straight up copies G
function greens_measurement(mc::DQMC, model::Model, greens_iterator=Greens; kwargs...)
    N = length(lattice(model)) * nflavors(model)
    Measurement(
        mc, model, greens_iterator, Nothing, greens_kernel, shape = (N, N); kwargs...
    )
end
greens_kernel(mc, model, G) = G



function occupation(mc::DQMC, model::Model, kwargs...)
    Measurement(mc, model, Greens, EachSiteAndFlavor, occupation_kernel; kwargs...)
end
occupation_kernel(mc, model, i, G) = 1 - G[i, i]



function charge_density(
        mc::DQMC, model::Model, greens_iterator; 
        lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    Measurement(mc, model, greens_iterator, lattice_iterator, cdc_kernel; kwargs...)
end
charge_density_correlation(mc, m; kwargs...) = charge_density(mc, m, Greens; kwargs...)
function charge_density_susceptibility(mc, m; kwargs...)
    charge_density(mc, m, CombinedGreensIterator; kwargs...)
end

function cdc_kernel(mc, model, i, j, G)
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
function cdc_kernel(mc, model, i, j, G00, G0l, Gl0, Gll)
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



"""
    magnetization_measurement(mc, model, dir[; lattice_iterator, kwargs...])

Returns the x, y or z magnetization measurement given `dir = :x`, `:y` or `:z`
respectively.

NOTE: 
We're skipping the multiplication by `-1im` during the measurement of the y 
magnetization. To get the correct result, multiply the final result by `-1im`.
"""
function magnetization(
        mc::DQMC, model::Model, dir::Symbol; 
        lattice_iterator = EachSite, kwargs...
    )
    checkflavors(model)
    if     dir == :x; kernel = mx_kernel
    elseif dir == :y; kernel = my_kernel
    elseif dir == :z; kernel = mz_kernel
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    Measurement(mc, model, Greens, lattice_iterator, kernel; kwargs...)
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



function spin_density(
        dqmc, model, dir::Symbol, greens_iterator; 
        lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    if     dir == :x; kernel = sdc_x_kernel
    elseif dir == :y; kernel = sdc_y_kernel
    elseif dir == :z; kernel = sdc_z_kernel
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    Measurement(dqmc, model, greens_iterator, lattice_iterator, kernel; kwargs...)
end
spin_density_correlation(args...; kwargs...) = spin_density(args..., Greens; kwargs...)
function spin_density_susceptibility(args...; kwargs...)
    spin_density(args..., CombinedGreensIterator; kwargs...)
end

function sdc_x_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    G[i+N, i] * G[j+N, j] - G[j+N, i] * G[i+N, j] +
    G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
    G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] +
    G[i, i+N] * G[j, j+N] - G[j, i+N] * G[i, j+N]
end
function sdc_x_kernel(mc, model, i, j, G00, G0l, Gl0, Gll)
    N = length(lattice(model))
    Gll[i+N, i] * G00[j+N, j] - G0l[j+N, i] * Gl0[i+N, j] +
    Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
    Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] +
    Gll[i, i+N] * G00[j, j+N] - G0l[j, i+N] * Gl0[i, j+N]
end

function sdc_y_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    - G[i+N, i] * G[j+N, j] + G[j+N, i] * G[i+N, j] +
      G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
      G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] -
      G[i, i+N] * G[j, j+N] + G[j, i+N] * G[i, j+N]
end
function sdc_y_kernel(mc, model, i, j, G00, G0l, Gl0, Gll)
    N = length(lattice(model))
    - Gll[i+N, i] * G00[j+N, j] + G0l[j+N, i] * Gl0[i+N, j] +
      Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
      Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] -
      Gll[i, i+N] * G00[j, j+N] + G0l[j, i+N] * Gl0[i, j+N]
end

function sdc_z_kernel(mc, model, i, j, G)
    N = length(lattice(model))
    (1 - G[i, i]) * (1 - G[j, j])         + (I[j, i] - G[j, i]) * G[i, j] -
    (1 - G[i, i]) * (1 - G[j+N, j+N])     + G[j+N, i] * G[i, j+N] -
    (1 - G[i+N, i+N]) * (1 - G[j, j])     + G[j, i+N] * G[i+N, j] +
    (1 - G[i+N, i+N]) * (1 - G[j+N, j+N]) + (I[j, i] - G[j+N, i+N]) * G[i+N, j+N]
end
function sdc_z_kernel(mc, model, i, j, G00, G0l, Gl0, Gll)
    N = length(lattice(model))
    (1 - Gll[i, i])     * (1 - G00[j, j])     - G0l[j, i] * Gl0[i, j] -
    (1 - Gll[i, i])     * (1 - G00[j+N, j+N]) + G0l[j+N, i] * Gl0[i, j+N] -
    (1 - Gll[i+N, i+N]) * (1 - G00[j, j])     + G0l[j, i+N] * Gl0[i+N, j] +
    (1 - Gll[i+N, i+N]) * (1 - G00[j+N, j+N]) - G0l[j+N, i+N] * Gl0[i+N, j+N]
end




function pairing(
        dqmc::DQMC, model::Model, greens_iterator; 
        K = 1+length(neighbors(lattice(model), 1)),
        lattice_iterator = EachLocalQuadByDistance{K}, kwargs...
    )
    Measurement(dqmc, model, Greens, lattice_iterator, pc_kernel; kwargs...)
end
pairing_correlation(mc, m; kwargs...) = pairing(mc, m, Greens; kwargs...)
pairing_susceptibility(mc, m; kwargs...) = pairing(mc, m, CombinedGreensIterator; kwargs...)
function pc_kernel(mc, model, src1, trg1, src2, trg2, G)
    N = length(lattice(model))
    # verified against ED for each (src1, src2, trg1, trg2)
    # Δ_v(src1, trg1) Δ_v^†(src2, trg2)
    # G_{i, j}^{↑, ↑} G_{i+d, j+d}^{↓, ↓} - G_{i, j+d}^{↑, ↓} G_{i+d, j}^{↓, ↑}
    G[src1, src2] * G[trg1+N, trg2+N] - G[src1, trg2+N] * G[trg1+N, src2]
end
function pc_kernel(mc, model, src1, trg1, src2, trg2, G00, G0l, Gl0, Gll)
    N = length(lattice(model))
    Gl0[src1, src2] * Gl0[trg1+N, trg2+N] - Gl0[src1, trg2+N] * Gl0[trg1+N, src2]
end



function boson_energy_measurement(dqmc, model; kwargs...)
    Measurement(dqmc, model, Nothing, Nothing, energy_boson; kwargs...)
end