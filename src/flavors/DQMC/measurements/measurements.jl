################################################################################
### Measurement constructors
################################################################################



function checkflavors(model, N=2)
    if nflavors(model) != N
        @warn(
            "$N flavors are required, but $(nflavors(model)) have been found"
        )
    end
    nothing
end



# This has lattice_iteratorator = Nothing, because it straight up copies G
function greens_measurement(
        mc::DQMC, model::Model, greens_iterator=Greens; 
        capacity = _default_capacity(mc), eltype = geltype(mc),
        obs = let
            N = length(lattice(model)) * nflavors(model)
            LogBinner(zeros(eltype, (N, N)), capacity=capacity)
        end, kwargs...
    )
    Measurement(
        mc, model, greens_iterator, Nothing, greens_kernel, 
        obs = obs; kwargs...
    )
end



function occupation(mc::DQMC, model::Model; wrapper = nothing, kwargs...)
    li = wrapper === nothing ? EachSiteAndFlavor : wrapper{EachSiteAndFlavor}
    Measurement(mc, model, Greens, li, occupation_kernel; kwargs...)
end



function charge_density(
        mc::DQMC, model::Model, greens_iterator; 
        wrapper = nothing, lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    Measurement(mc, model, greens_iterator, li, cdc_kernel; kwargs...)
end

charge_density_correlation(mc, m; kwargs...) = charge_density(mc, m, Greens; kwargs...)
charge_density_susceptibility(mc, m; kwargs...) = charge_density(mc, m, CombinedGreensIterator; kwargs...)



"""
    magnetization(mc, model, dir[; lattice_iterator, kwargs...])

Returns the x, y or z magnetization measurement given `dir = :x`, `:y` or `:z`
respectively.

NOTE: 
We're skipping the multiplication by `-1im` during the measurement of the y 
magnetization. To get the correct result, multiply the final result by `-1im`.
"""
function magnetization(
        mc::DQMC, model::Model, dir::Symbol; 
        wrapper = nothing, lattice_iterator = EachSite, kwargs...
    )
    checkflavors(model)
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    if dir == :x; 
        return Measurement(mc, model, Greens, li, mx_kernel; kwargs...)
    elseif dir == :y; 
        return Measurement(mc, model, Greens, li, my_kernel; kwargs...)
    elseif dir == :z; 
        return Measurement(mc, model, Greens, li, mz_kernel; kwargs...)
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    
end



function spin_density(
        dqmc, model, dir::Symbol, greens_iterator; 
        wrapper = nothing, lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    dir in (:x, :y, :z) || throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    if     dir == :x
        return Measurement(dqmc, model, greens_iterator, li, sdc_x_kernel; kwargs...)
    elseif dir == :y
        return Measurement(dqmc, model, greens_iterator, li, sdc_y_kernel; kwargs...)
    elseif dir == :z
        return Measurement(dqmc, model, greens_iterator, li, sdc_z_kernel; kwargs...)
    else
        error("$greens_iterator not recongized")
    end
end
spin_density_correlation(args...; kwargs...) = spin_density(args..., Greens; kwargs...)
spin_density_susceptibility(args...; kwargs...) = spin_density(args..., CombinedGreensIterator; kwargs...)



function pairing(
        dqmc::DQMC, model::Model, greens_iterator; 
        K = 1+length(neighbors(lattice(model), 1)), wrapper = nothing, 
        lattice_iterator = EachLocalQuadByDistance{K}, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    Measurement(dqmc, model, greens_iterator, li, pc_kernel; kwargs...)
end
pairing_correlation(mc, m; kwargs...) = pairing(mc, m, Greens; kwargs...)
pairing_susceptibility(mc, m; kwargs...) = pairing(mc, m, CombinedGreensIterator; kwargs...)



#=
Λ^L = Λxx(qx->0, qy=0, iω=0)
    = ∑_r ∫[0, ß] dτ ⟨j_x(r, τ) j_x(0, 0)⟩ exp(iqr) exp(iωτ)
j_x(r, τ) = it ∑_σ [c_{r+e_x, σ}^†(τ) c_{r, σ}(τ) - c_{r, σ}^†(τ) c_{r+e_x, σ}(τ)]

# reframe as:
0 -> r, r -> r + Δr, e_x -> r+e_x, r+e_x -> r+e_x+Δr
then use EachLocalQuadByDistance
Well actually no, because we want e_x to only pick one neighbor.

Λ_{trg-src}(Δr) = \sum_src1 cc_kernel(...)
∑_Δr exp(iq⋅Δr) real(Λ_{trg-src}(Δr)) # Why real? this is silly...
where q = Δq_i = k_1, k_2 (or k_1+k_2?)

                       v- (0, qy)
ρs = 1/8 * (real(Λxxq[1,2]) - real(Λxxq[2,1]) + real(Λyyq[1,2]) - real(Λyyq[2,1]))
                                (qx, 0) -^
This should correspond to
ρ = -K_s - Λ^T = Λxx^L - Λxx^T = Λxx((0, qy)) - Λxx((qx, 0))
  ~ real(Λxxq[1,2]) + real(Λyyq[1,2]) - real(Λxxq[2,1]) - real(Λyyq[2,1])
    ^-        sum over NNs?        -^

So I need to compute reciprocal lattice vectors I guess
So what is qx, qy then? I assume those reciprocal lattice vectors? Or literally
just a dx/dy?

Any real-space runtime computation could be EachLocalQuadByDistance, we can do
summing, direction picking and Fourier afterwards...

To be more efficient:
Should implement new lattice iterator that picks trg1, trg2 to be in the same
direction and skips trg == src
EachSyncedNNQuadByDistance{K}?
=#
function current_current_susceptibility(
        dqmc::DQMC, model::Model; 
        K = 1+length(neighbors(lattice(model), 1)),
        greens_iterator = CombinedGreensIterator, wrapper = nothing,
        lattice_iterator = EachLocalQuadBySyncedDistance{K}, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    Measurement(dqmc, model, greens_iterator, li, cc_kernel; kwargs...)
end
function superfluid_density(
        dqmc::DQMC, model::Model, Ls = size(lattice(model)); 
        K = 1+length(neighbors(lattice(model), 1)), 
        capacity = _default_capacity(dqmc),
        obs = LogBinner(ComplexF64(0), capacity=capacity),
        kwargs...
    )
    @assert K > 1
    dirs = directions(lattice(model))

    # Note: this only works with 2d vecs
    # Note: longs and trans are epsilon-vectors, i.e. they're the smallest 
    # step we can take in discrete reciprocal space
    lvecs = lattice_vectors(lattice(model))
    uc_vecs = lvecs ./ Ls
    prefactor = 2pi / dot(lvecs...)
    rvecs = map(lv -> prefactor * cross([lv; 0], [0, 0, 1])[[1, 2]], uc_vecs)

    # Note: this only works for 2d lattices... maybe
    longs = []
    trans = []
    dir_idxs = []
    for i in 2:K
        if -uc_vecs[1] ≈ dirs[i] || uc_vecs[1] ≈ dirs[i]
            push!(longs, rvecs[1])
            push!(trans, rvecs[2])
            push!(dir_idxs, i)
        elseif -uc_vecs[2] ≈ dirs[i] || uc_vecs[2] ≈ dirs[i]
            push!(longs, rvecs[2])
            push!(trans, rvecs[1])
            push!(dir_idxs, i)
        else
            # We kinda need the i, j from R = i*a_1 + j*a_2 here so we can
            # construct the matching reciprocal vectors here
            @error("Skipping $(dirs[i]) - not a nearest neighbor")
        end
    end

    #=
    longs = normalize.(dirs[2:K]) * 1/L
    trans = map(dirs[2:K]) do v
        n = [normalize(v)..., 0]
        u = cross([0,0,1], n)
        u[1:2] / L
    end
    longs .*= 2pi
    trans .*= 2pi
    =#
    li = SuperfluidDensity{EachLocalQuadBySyncedDistance{K}}(
        dir_idxs, longs, trans
    )
    # Measurement(dqmc, model, CombinedGreensIterator, li, cc_kernel, obs=obs; kwargs...)
    current_current_susceptibility(dqmc, model, lattice_iterator = li)
end



function boson_energy_measurement(dqmc, model; kwargs...)
    Measurement(dqmc, model, Nothing, Nothing, energy_boson; kwargs...)
end



function noninteracting_energy(dqmc, model; kwargs...)
    Measurement(dqmc, model, Greens, Nothing, nonintE_kernel; kwargs...)
end


# These require the model to implement intE_kernel
function interacting_energy(dqmc, model; kwargs...)
    Measurement(dqmc, model, Greens, Nothing, intE_kernel; kwargs...)
end

function total_energy(dqmc, model; kwargs...)
    Measurement(dqmc, model, Greens, Nothing, totalE_kernel; kwargs...)
end



################################################################################
### Wicks expanded Kernel functions
################################################################################

# Notes on math and stuff:
#=
- Use Wicks theorem to go from one expectation value with 2N operators to 
  N expectation values with 2 operators each
- use G_{ij} = ⟨c_i c_j^†⟩
- use ⟨c_i^† c_j⟩ = δ_{ij} - ⟨c_j c_i^†⟩ = (I - G)_{ji} = dagger(::GreensMatrix)[i, j]
- we define G[i,j] as spin up, G[i+N,j+N] as spin down (N = number of sites)
- we used (I - G)[i, j+N] = G[i, j+N] etc when possible
- Gl0_{i,j} = ⟨c_i(l) c_j(0)^†⟩
- ⟨c_i(0) c_j(l)^†⟩ = ⟨(c_j(l) c_i(0)^†)^†⟩ = (⟨c_j(l) c_i(0)^†⟩)^† 
                    = Gl0_{j,i}^† = Gl0_{j,i}^*
=#

"""
    greens_kernel(mc, model, G::GreensMatrix)

Returns the unprocessed Greens function `greens(mc) = {⟨cᵢcⱼ^†⟩}`.

* Lattice Iterators: `nothing` (zero index)
* Greens Iterators: `Greens` or `GreensAt`
"""
greens_kernel(mc, model, G::GreensMatrix) = G.val


"""
    occupation_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per index occupation `⟨nᵢ⟩`.

* Lattice Iterators: `EachSiteAndFlavor`, `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
occupation_kernel(mc, model, i::Integer, G::GreensMatrix) = 1 - G[i, i]


"""
    cdc_kernel(mc, model, ij::NTuple{2, Integer}, G::GreensMatrix)
    cdc_kernel(mc, model, ij::NTuple{2, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair charge density `⟨nᵢ(τ) nⱼ(0)⟩ - ⟨nᵢ(τ)⟩⟨nⱼ(0)⟩` where 
`τ = 0` for the first signature.

* Lattice Iterators: `OnSite`, `EachSitePair` or `EachSitePairByDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function cdc_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix)
    i, j = ij
    N = length(lattice(mc))
    # ⟨n↑n↑⟩
    dagger(G)[i, i] * dagger(G)[j, j] +
    dagger(G)[i, j] * G[i, j] +
    # ⟨n↑n↓⟩
    dagger(G)[i, i] * dagger(G)[j+N, j+N] +
    dagger(G)[i, j+N] * G[i, j + N] +
    # ⟨n↓n↑⟩
    dagger(G)[i+N, i+N] * dagger(G)[j, j] +
    dagger(G)[i+N, j] * G[i+N, j] +
    # ⟨n↓n↓⟩
    dagger(G)[i+N, i+N]  * dagger(G)[j+N, j+N] +
    dagger(G)[i+N, j+N] * G[i+N, j+N]
end

function cdc_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(mc))
    # ⟨n↑(l)n↑(0)⟩
    dagger(Gll)[i, i] * dagger(G00)[j, j] +
    dagger(G0l)[i, j] * Gl0[i, j] +
    # ⟨n↑(l)n↓(0)⟩
    dagger(Gll)[i, i] * dagger(G00)[j+N, j+N] +
    dagger(G0l)[i, j+N] * Gl0[i, j + N] +
    # ⟨n↓(l)n↑(0)⟩
    dagger(Gll)[i+N, i+N] * dagger(G00)[j, j] +
    dagger(G0l)[i+N, j] * Gl0[i+N, j] +
    # ⟨n↓(l)n↓(0)⟩
    dagger(Gll)[i+N, i+N] * dagger(G00)[j+N, j+N] +
    dagger(G0l)[i+N, j+N] * Gl0[i+N, j+N]
end


"""
    mx_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site x-magnetization `⟨cᵢ↑^† cᵢ↓ + cᵢ↓^† cᵢ↑⟩`.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function mx_kernel(mc, model, i, G::GreensMatrix)
    N = length(lattice(model))
    -G[i+N, i] - G[i, i+N]
end

"""
    my_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site y-magnetization `-⟨cᵢ↑^† cᵢ↓ - cᵢ↓^† cᵢ↑⟩` without the 
imaginary prefactor.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function my_kernel(mc, model, i, G::GreensMatrix)
    N = length(lattice(model))
    G[i+N, i] - G[i, i+N]
end

"""
    mz_kernel(mc, model, i::Integer, G::GreensMatrix)

Returns the per-site z-magnetization `⟨nᵢ↑ - nᵢ↓⟩`.
    
* Lattice Iterators: `EachSite`
* Greens Iterators: `Greens` or `GreensAt`
"""
function mz_kernel(mc, model, i, G::GreensMatrix)
    N = length(lattice(model))
    G[i+N, i+N] - G[i, i]
end


"""
    sdc_x_kernel(mc, model, ij::NTuple{2, Integer}, G::GreensMatrix)
    sdc_x_kernel(mc, model, ij::NTuple{2, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair x-spin density `⟨mxᵢ(τ) mxⱼ(0)⟩ - ⟨mxᵢ(τ)⟩⟨mxⱼ(0)⟩` 
where `τ = 0` for the first signature.

* Lattice Iterators: `OnSite`, `EachSitePair` or `EachSitePairByDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function sdc_x_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix)
    i, j = ij
    N = length(lattice(model))
    G[i+N, i] * G[j+N, j] + G[i+N, i] * G[j, j+N] + 
    G[i, i+N] * G[j+N, j] + G[i, i+N] * G[j, j+N] + 
    dagger(G)[i, j+N] * G[i+N, j] + dagger(G)[i, j] * G[i+N, j+N] +
    dagger(G)[i+N, j+N] * G[i, j] + dagger(G)[i+N, j] * G[i, j+N]
end
function sdc_x_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    Gll[i+N, i] * G00[j+N, j] + Gll[i+N, i] * G00[j, j+N] + 
    Gll[i, i+N] * G00[j+N, j] + Gll[i, i+N] * G00[j, j+N] + 
    dagger(G0l)[i, j+N] * Gl0[i+N, j] + dagger(G0l)[i, j] * Gl0[i+N, j+N] +
    dagger(G0l)[i+N, j+N] * Gl0[i, j] + dagger(G0l)[i+N, j] * Gl0[i, j+N]
end

"""
    sdc_y_kernel(mc, model, ij::NTuple{2, Integer}, G::GreensMatrix)
    sdc_y_kernel(mc, model, ij::NTuple{2, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair x-spin density `⟨myᵢ(τ) myⱼ(0)⟩ - ⟨myᵢ(τ)⟩⟨myⱼ(0)⟩` 
where `τ = 0` for the first signature.

* Lattice Iterators: `OnSite`, `EachSitePair` or `EachSitePairByDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function sdc_y_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix)
    i, j = ij
    N = length(lattice(model))
    - G[i+N, i] * G[j+N, j] + G[i+N, i] * G[j, j+N] + 
      G[i, i+N] * G[j+N, j] - G[i, i+N] * G[j, j+N] + 
    - dagger(G)[i, j+N] * G[i+N, j] + dagger(G)[i, j] * G[i+N, j+N] +
      dagger(G)[i+N, j+N] * G[i, j] - dagger(G)[i+N, j] * G[i, j+N]
end
function sdc_y_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    - Gll[i+N, i] * G00[j+N, j] + Gll[i+N, i] * G00[j, j+N] + 
      Gll[i, i+N] * G00[j+N, j] - Gll[i, i+N] * G00[j, j+N] + 
    - dagger(G0l)[i, j+N] * Gl0[i+N, j] + dagger(G0l)[i, j] * Gl0[i+N, j+N] +
      dagger(G0l)[i+N, j+N] * Gl0[i, j] - dagger(G0l)[i+N, j] * Gl0[i, j+N]
end

"""
    sdc_z_kernel(mc, model, ij::NTuple{2, Integer}, G::GreensMatrix)
    sdc_z_kernel(mc, model, ij::NTuple{2, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair x-spin density `⟨mzᵢ(τ) mzⱼ(0)⟩ - ⟨mzᵢ(τ)⟩⟨mzⱼ(0)⟩` 
where `τ = 0` for the first signature.

* Lattice Iterators: `OnSite`, `EachSitePair` or `EachSitePairByDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function sdc_z_kernel(mc, model, ij::NTuple{2}, G::GreensMatrix)
    i, j = ij
    N = length(lattice(model))
    dagger(G)[i, i]     * dagger(G)[j, j] - 
    dagger(G)[i, i]     * dagger(G)[j+N, j+N] -
    dagger(G)[i+N, i+N] * dagger(G)[j, j] + 
    dagger(G)[i+N, i+N] * dagger(G)[j+N, j+N] + 
    dagger(G)[i, j] * G[i, j]     - dagger(G)[i, j+N] * G[i, j+N] -
    dagger(G)[i+N, j] * G[i+N, j] + dagger(G)[i+N, j+N] * G[i+N, j+N]
end
function sdc_z_kernel(mc, model, ij::NTuple{2}, packed_greens::NTuple{4})
    i, j = ij
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    dagger(Gll)[i, i] * dagger(G00)[j, j] - 
    dagger(Gll)[i, i] * dagger(G00)[j+N, j+N] -
    dagger(Gll)[i+N, i+N] * dagger(G00)[j, j] + 
    dagger(Gll)[i+N, i+N] * dagger(G00)[j+N, j+N] + 
    dagger(G0l)[i, j] * Gl0[i, j]     - dagger(G0l)[i, j+N] * Gl0[i, j+N] -
    dagger(G0l)[i+N, j] * Gl0[i+N, j] + dagger(G0l)[i+N, j+N] * Gl0[i+N, j+N]
end


"""
    pc_kernel(mc, model, ij::NTuple{4, Integer}, G::GreensMatrix)
    pc_kernel(mc, model, ij::NTuple{4, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair pairing `⟨Δᵢⱼ(τ) Δₖₗ^†(0)⟩` where `τ = 0` for the 
first signature and `Δᵢⱼ(τ) = 0.5 cᵢ↑(τ) cⱼ↓(τ)`. The Delta operator is a 
deferred version of the  pair field operator `Δᵢ = 0.5 ∑ₐ f(a) cᵢ↑ cᵢ₊ₐ↓` which 
leaves the execution of the sum for after the simulation. 

* Lattice Iterators: `EachLocalQuadByDistance` or `EachLocalQuadBySyncedDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function pc_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix)
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

"""
    pc_kernel(mc, model, ij::NTuple{4, Integer}, G::GreensMatrix)
    pc_kernel(mc, model, ij::NTuple{4, Integer}, Gs::NTuple{4, GreensMatrix})

Returns the per-site-pair pairing `⟨Δᵢⱼ^†(τ) Δₖₗ(0)⟩` where `τ = 0` for the 
first signature and `Δᵢⱼ(τ) = 0.5 cᵢ↑(τ) cⱼ↓(τ)`. The Delta operator is a 
deferred version of the  pair field operator `Δᵢ = 0.5 ∑ₐ f(a) cᵢ↑ cᵢ₊ₐ↓` which 
leaves the execution of the sum for after the simulation. 

* Lattice Iterators: `EachLocalQuadByDistance` or `EachLocalQuadBySyncedDistance`
* Greens Iterators: `Greens`, `GreensAt` or `CombinedGreensIterator`
"""
function pc_alt_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix)
    pc_alt_kernel(mc, model, sites, (G, G, G, G))
end
function pc_alt_kernel(mc, model, sites::NTuple{4}, packed_greens::NTuple{4})
    src1, trg1, src2, trg2 = sites
	G00, G0l, Gl0, Gll = packed_greens
    N = length(lattice(model))
    # Δ_v^†(src1, trg1)(τ) Δ_v(src2, trg2)(0)
    # (I-G)_{j, i}^{↑, ↑}(0, τ) (I-G)_{j+d', i+d}^{↓, ↓}(0, τ) - 
    # (I-G)_{j, i+d}^{↑, ↓}(0, τ) G_{j+d', i}^{↓, ↑}(0, τ)
    dagger(G0l)[trg1+N, trg2+N] * dagger(G0l)[src1, src2] -
    G0l[src2, trg1+N] * G0l[trg2+N, src1]
end

function pc_combined_kernel(mc, model, sites::NTuple{4}, G)
    # Δ^† Δ + Δ Δ^†
    pc_kernel(mc, model, sites, G) + pc_alt_kernel(mc, model, sites, G)
end


function pc_ref_kernel(mc, model, sites::NTuple{4}, G::GreensMatrix)
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



@inline function nonintE_kernel(mc, model, G::GreensMatrix)
    # <T> = \sum Tji * (Iij - Gij) = - \sum Tji * (Gij - Iij)
    T = mc.stack.hopping_matrix
    nonintE(T, G.val)
end

# TODO should this be moved/work differently?
nonintE(T::AbstractArray, G::GreensMatrix) = nonintE(T, G.val)
function nonintE(T::AbstractArray, G::AbstractArray)
    output = zero(eltype(G))
    for i in axes(G, 1), j in axes(G, 2)
        output += T[j, i] * (I[i, j] - G[i, j])
    end
    # 2 because we're using spin up/down symmetry
    2.0 * output
end
function nonintE(T::BlockDiagonal{X, N}, G::BlockDiagonal{X, N}) where {X, N}
    output = zero(eltype(G))
    @inbounds n = size(T.blocks[1], 1)
    @inbounds for i in 1:N
        t = T.blocks[i]
        g = G.blocks[i]
        @avx for k in 1:n, l in 1:n
            output += t[k,l] * (ifelse(k==l, 1.0, 0.0) - g[k,l])
        end
    end
    output
end


function totalE_kernel(mc, model, G::GreensMatrix)
    nonintE_kernel(mc, model, G) + intE_kernel(mc, model, G)
end
