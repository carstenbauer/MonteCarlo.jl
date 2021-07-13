function _expand_indices(N)
    if N == 1
        return :(i = idxs), Integer
    elseif N == 2
        return :(i = idxs[1]; j = idxs[2]), NTuple{2}
    elseif N == 4
        return :(i = idxs[1]; j = idxs[2]; k = idxs[3]; l = idxs[4]), NTuple{4}
    else
        error("Cannot splat $N indices.")
    end
end

function equal_time_kernel(N, name, code)
    splatting, T = _expand_indices(N)
    # unequal time -> equal time just means that every G is at the same time
    # We define G00 etc here to more compactly define kernels
    quote
        function $name(mc, model, idxs::$T, G::AbstractArray)
            $splatting
            N = length(lattice(mc))
            G00 = G0l = Gl0 = Gll = G
            $code
        end
    end
end

function unequal_time_kernel(N, name, code)
    splatting, T = _expand_indices(N)
    quote
        function $name(mc, model, idxs::$T, packed_greens::NTuple{4})
            $splatting
            G00, G0l, Gl0, Gll = packed_greens
            N = length(lattice(mc))
            $code
        end
    end
end



module _measurement_kernel_code
    using ..MonteCarlo: equal_time_kernel, unequal_time_kernel

    const greens = :(_greens_kernel(mc, m, G::GreensMatrix) = G.val)

    const occupation = :(_occupation_kernel(mc, m, i::Integer, G::AbstractArray) = 1 - G[i, i])

    const equal_time_charge_density = equal_time_kernel(2, :_etcd_kernel, quote
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
    end)
    
    # Some I[i, j]'s drop here because of time-deltas
    const unequal_time_charge_density = unequal_time_kernel(2, :_etcd_kernel, quote
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
    end)

    const Mx = equal_time_kernel(1, :_etMx_kernel, :(-G[i+N, i] - G[i, i+N]))
    const My = equal_time_kernel(1, :_etMy_kernel, :(G[i+N, i] - G[i, i+N]))
    const Mz = equal_time_kernel(1, :_etMz_kernel, :(G[i+N, i+N] - G[i, i]))

    const equal_time_spin_density_x = equal_time_kernel(2, :_etsdx_kernel, quote
        G[i+N, i] * G[j+N, j] - G[j+N, i] * G[i+N, j] +
        G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
        G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] +
        G[i, i+N] * G[j, j+N] - G[j, i+N] * G[i, j+N]
    end)
    const unequal_time_spin_density_x = unequal_time_kernel(2, :_utsdx_kernel, quote
        Gll[i+N, i] * G00[j+N, j] - G0l[j+N, i] * Gl0[i+N, j] +
        Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
        Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] +
        Gll[i, i+N] * G00[j, j+N] - G0l[j, i+N] * Gl0[i, j+N]
    end)

    const equal_time_spin_density_y = equal_time_kernel(2, :_etsdy_kernel, quote
        - G[i+N, i] * G[j+N, j] + G[j+N, i] * G[i+N, j] +
          G[i+N, i] * G[j, j+N] + (I[j, i] - G[j, i]) * G[i+N, j+N] +
          G[i, i+N] * G[j+N, j] + (I[j, i] - G[j+N, i+N]) * G[i, j] -
          G[i, i+N] * G[j, j+N] + G[j, i+N] * G[i, j+N]
    end)
    const unequal_time_spin_density_y = unequal_time_kernel(2, :_utsdy_kernel, quote
        - Gll[i+N, i] * G00[j+N, j] + G0l[j+N, i] * Gl0[i+N, j] +
          Gll[i+N, i] * G00[j, j+N] - G0l[j, i] * Gl0[i+N, j+N] +
          Gll[i, i+N] * G00[j+N, j] - G0l[j+N, i+N] * Gl0[i, j] -
          Gll[i, i+N] * G00[j, j+N] + G0l[j, i+N] * Gl0[i, j+N]
    end)

    const equal_time_spin_density_z = equal_time_kernel(2, :_etsdz_kernel, quote
        (1 - G[i, i]) * (1 - G[j, j])         + (I[j, i] - G[j, i]) * G[i, j] -
        (1 - G[i, i]) * (1 - G[j+N, j+N])     + G[j+N, i] * G[i, j+N] -
        (1 - G[i+N, i+N]) * (1 - G[j, j])     + G[j, i+N] * G[i+N, j] +
        (1 - G[i+N, i+N]) * (1 - G[j+N, j+N]) + (I[j, i] - G[j+N, i+N]) * G[i+N, j+N]
    end)
    const unequal_time_spin_density_z = unequal_time_kernel(2, :_utsdz_kernel, quote
        (1 - Gll[i, i])     * (1 - G00[j, j])     - G0l[j, i] * Gl0[i, j] -
        (1 - Gll[i, i])     * (1 - G00[j+N, j+N]) + G0l[j+N, i] * Gl0[i, j+N] -
        (1 - Gll[i+N, i+N]) * (1 - G00[j, j])     + G0l[j, i+N] * Gl0[i+N, j] +
        (1 - Gll[i+N, i+N]) * (1 - G00[j+N, j+N]) - G0l[j+N, i+N] * Gl0[i+N, j+N]
    end)

    # i, j, k, L
    # src, trg, src, trg
    # Δ_v(i, j)(τ) Δ_v^†(k, l)(0)
    # G_{i, j}^{↑, ↑}(τ, 0) G_{i+d, j+d'}^{↓, ↓}(τ, 0) - 
    # G_{i, j+d'}^{↑, ↓}(τ, 0) G_{i+d, j}^{↓, ↑}(τ, 0)
    const equal_time_pairing, unequal_time_pairing = let
        wicks = :(Gl0[i, k] * Gl0[j+N, l+N] - Gl0[i, l+N] * Gl0[j+N, k])
        equal_time_kernel(4, :_etp_kernel, wicks), unequal_time_kernel(4, :_utp_kernel, wicks)
    end

    const current_current = unequal_time_kernel(4, :_utcc_kernel, quote
        T = mc.stack.hopping_matrix
        output = zero(eltype(G00))

        # Iterate through (spin up, spin down)
        for σ1 in (0, N), σ2 in (0, N)
            s1 = i + σ1; t1 = j + σ1
            s2 = k + σ2; t2 = l + σ2
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
        output
    end)

    const boson_energy = :(be_kernel(mc, m) = energy_boson(mc, m))

    const noninteracting_energy = :(_niE_kernel(mc, m, G::AbstractArray) = nonintE(mc.stack.hopping_matrix, G))
end



################################################################################
### Measurement constructors
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
function greens_measurement(
        mc::DQMC, model::Model, greens_iterator=Greens; 
        capacity = _default_capacity(mc), eltype = geltype(mc),
        obs = let
            N = length(lattice(model)) * nflavors(model)
            LogBinner(zeros(eltype, (N, N)), capacity=capacity)
        end, kwargs...
    )
    code = _measurement_kernel_code.greens
    Measurement(
        mc, model, greens_iterator, Nothing, code, 
        obs = obs; kwargs...
    )
end



function occupation(mc::DQMC, model::Model; wrapper = nothing, kwargs...)
    li = wrapper === nothing ? EachSiteAndFlavor : wrapper{EachSiteAndFlavor}
    code = _measurement_kernel_code.occupation
    Measurement(mc, model, Greens, li, code; kwargs...)
end



function charge_density(
        mc::DQMC, model::Model, greens_iterator; 
        wrapper = nothing, lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    code = if greens_iterator == CombinedGreensIterator
        _measurement_kernel_code.unequal_time_charge_density
    elseif greens_iterator == Greens
        _measurement_kernel_code.equal_time_charge_density
    else
        error("$greens_iterator not recongized")
    end
    Measurement(mc, model, greens_iterator, li, code; kwargs...)
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
    if dir == :x; 
        code = _measurement_kernel_code.Mx
    elseif dir == :y; 
        code = _measurement_kernel_code.My
    elseif dir == :z; 
        code = _measurement_kernel_code.Mz
    else throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    end
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    Measurement(mc, model, Greens, li, code; kwargs...)
end



function spin_density(
        dqmc, model, dir::Symbol, greens_iterator; 
        wrapper = nothing, lattice_iterator = EachSitePairByDistance, kwargs...
    )
    checkflavors(model)
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    dir in (:x, :y, :z) || throw(ArgumentError("`dir` must be :x, :y or :z, but is $dir"))
    code = if greens_iterator == CombinedGreensIterator
        if     dir == :x;   _measurement_kernel_code.unequal_time_spin_density_x
        elseif dir == :y;   _measurement_kernel_code.unequal_time_spin_density_y
        else                _measurement_kernel_code.unequal_time_spin_density_z end
    elseif greens_iterator == Greens
        if     dir == :x;   _measurement_kernel_code.equal_time_spin_density_x
        elseif dir == :y;   _measurement_kernel_code.equal_time_spin_density_y
        else                _measurement_kernel_code.equal_time_spin_density_z end
    else
        error("$greens_iterator not recongized")
    end
    Measurement(dqmc, model, greens_iterator, li, code; kwargs...)
end
spin_density_correlation(args...; kwargs...) = spin_density(args..., Greens; kwargs...)
spin_density_susceptibility(args...; kwargs...) = spin_density(args..., CombinedGreensIterator; kwargs...)



function pairing(
        dqmc::DQMC, model::Model, greens_iterator; 
        K = 1+length(neighbors(lattice(model), 1)), wrapper = nothing, 
        lattice_iterator = EachLocalQuadByDistance{K}, kwargs...
    )
    li = wrapper === nothing ? lattice_iterator : wrapper{lattice_iterator}
    code = if greens_iterator == CombinedGreensIterator
        _measurement_kernel_code.unequal_time_pairing
    elseif greens_iterator == Greens
        _measurement_kernel_code.equal_time_pairing
    else
        error("$greens_iterator not recongized")
    end
    Measurement(dqmc, model, greens_iterator, li, code; kwargs...)
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
    code = _measurement_kernel_code.current_current
    Measurement(dqmc, model, greens_iterator, li, code; kwargs...)
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
    current_current_susceptibility(dqmc, model,lattice_iterator = li)
end



function boson_energy_measurement(dqmc, model; kwargs...)
    code = _measurement_kernel_code.boson_energy
    Measurement(dqmc, model, Nothing, Nothing, code; kwargs...)
end



function noninteracting_energy(dqmc, model; kwargs...)
    code = _measurement_kernel_code.noninteracting_energy
    Measurement(dqmc, model, Greens, Nothing, code; kwargs...)
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