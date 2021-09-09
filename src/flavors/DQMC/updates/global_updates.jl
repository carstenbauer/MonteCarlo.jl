################################################################################
### Linalg additions
################################################################################

# Notes:
# * det(A B) = det(A) det(B)
# * det(A^-1) = 1/det(A)
# * |det(U)| = 1 (complex norm, U unitary)
# * det(T) = 1 (T unit-triangular like our T's)
# * our UDT decomposition always makes D real positive
# * all weights (det(G)) should be real positive
# * local updates already check ratios (det(G)/det(G')) so it's probably OK to 
#   ignore phases here!?



@bm function calculate_inv_greens_udt(Ul, Dl, Tl, Ur, Dr, Tr, G, pivot, temp)
    # G = [I + Ul Dl Tl Tr^† Dr Ur^†]^-1
    vmul!(G, Tl, adjoint(Tr))
    vmul!(Tr, G, Diagonal(Dr))
    vmul!(G, Diagonal(Dl), Tr)
    #   = [I + Ul G Ur^†]^-1
    udt_AVX_pivot!(Tr, Dr, G, pivot, temp, Val(false)) # Dl available
    #   = [I + Ul Tr Dr G Ur^†]^-1  w/ Tr unitary, G triangular
    #   = Ur G^-1 [(Ul Tr)^† Ur G^-1 + Dr]^-1 (Ul Tr)^†
    vmul!(Tl, Ul, Tr)
    rdivp!(Ur, G, Ul, pivot) # requires unpivoted udt decompostion (Val(false))
    #   = Ur [Tl^† Ur + Dr]^-1 Tl^†  w/ Tl unitary, Ur not
    vmul!(Tr, adjoint(Tl), Ur)
    rvadd!(Tr, Diagonal(Dr))
    #   = Ur Tr^-1 Tl^†
    udt_AVX_pivot!(Ul, Dr, Tr, pivot, temp, Val(false)) # Dl available
    #   = Ur Tr^-1 Dr^-1 Ul^† Tl^†
    #   = (old_Ur / G) Tr^-1 Dr^-1 Ul^† Tl^†
    # with old_Ur, Ul, Tl unitary and G, Tr triangular
    # det(G) = phase1 / 1 / 1 / det(Dr) / phase2 / phase3
    # where we ignore phases because they should be 1 and we already check this
    # in local updates.
    return
end

# after the above without modifying Ur, Tr, Tl, Ul, Dr
@bm function finish_calculate_greens(Ul, Dl, Tl, Ur, Dr, Tr, G, pivot, temp)
    # G = Ur Tr^-1 Dr^-1 Ul^† Tl^†
    rdivp!(Ur, Tr, G, pivot) # requires unpivoted udt decompostion (false)
    vmul!(Tr, Tl, Ul)
    #   = Ur Dr^-1 Tr^†
    vinv!(Dl, Dr)
    vmul!(Ul, Ur, Diagonal(Dl))
    vmul!(G, Ul, adjoint(Tr))
    return G
end

# This calculates the UDT stack stuff from scratch, but doesn't calculate greens
# fully. We use that det(UDT) = prod(D), i.e. that det(U) = 1 by definition and
# det(T) = 1 because T is unit-triangular by construction
@bm function inv_det(
        mc::DQMC, slice::Int, 
        conf::AbstractArray = mc.conf, safe_mult::Int = mc.parameters.safe_mult
    )
    copyto!(mc.stack.curr_U, I)
    copyto!(mc.stack.Ur, I)
    mc.stack.Dr .= one(eltype(mc.stack.Dr))
    copyto!(mc.stack.Tr, I)

    # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
    if slice+1 <= mc.parameters.slices
        start = slice+1
        stop = mc.parameters.slices
        for k in reverse(start:stop)
            if mod(k,safe_mult) == 0
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dr))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dr, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tr)
                vmul!(mc.stack.Tr, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
            end
        end
        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dr))
        udt_AVX_pivot!(mc.stack.Ur, mc.stack.Dr, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
        copyto!(mc.stack.tmp2, mc.stack.Tr)
        vmul!(mc.stack.Tr, mc.stack.tmp1, mc.stack.tmp2)
    end


    copyto!(mc.stack.curr_U, I)
    copyto!(mc.stack.Ul, I)
    mc.stack.Dl .= one(eltype(mc.stack.Dl))
    copyto!(mc.stack.Tl, I)

    # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
    if slice >= 1
        start = 1
        stop = slice
        for k in start:stop
            if mod(k,safe_mult) == 0
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
                vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dl))
                udt_AVX_pivot!(mc.stack.curr_U, mc.stack.Dl, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
                copyto!(mc.stack.tmp2, mc.stack.Tl)
                vmul!(mc.stack.Tl, mc.stack.tmp1, mc.stack.tmp2)
            else
                multiply_slice_matrix_left!(mc, mc.model, k, mc.stack.curr_U, conf)
            end
        end
        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.Dl))
        udt_AVX_pivot!(mc.stack.Ul, mc.stack.Dl, mc.stack.tmp1, mc.stack.pivot, mc.stack.tempv)
        copyto!(mc.stack.tmp2, mc.stack.Tl)
        vmul!(mc.stack.Tl, mc.stack.tmp1, mc.stack.tmp2)
    end

    calculate_inv_greens_udt(
        mc.stack.Ul, mc.stack.Dl, mc.stack.Tl, mc.stack.Ur, mc.stack.Dr, mc.stack.Tr, 
        mc.stack.greens_temp, mc.stack.pivot, mc.stack.tempv
    )
    return mc.stack.Dr
end



################################################################################
### Global Metropolis update
################################################################################



@bm function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray)
    # I don't think we need this...
    @assert mc.stack.current_slice == 1
    @assert mc.stack.direction == 1

    # This should be just after calculating greens, so mc.stack.Dl is from the UDT
    # decomposed G
    # We need an independent temp vector here as inv_det changes Dl, Dr and tempv
    mc.stack.tempvf .= mc.stack.Dl

    # This is essentially reverse_build_stack + partial calculate_greens
    # after this: G = Ur Tr^-1 Dr^-1 Ul^† Tl^†
    # where Ur, Ul, Tl only contribute complex phases and Tr contributes 1
    # Since we are already checking for sign problems in local updates we ignore
    # the phases here and set det(G) = 1 / det(Dr)
    inv_det(mc, current_slice(mc)-1, conf)

    # This loop helps with stability - it multiplies large and small numbers
    # whihc avoid reaching extremely large or small (typemin/max) floats
    detratio = 1.0
    for i in eachindex(mc.stack.tempvf)
        detratio *= mc.stack.tempvf[i] * mc.stack.Dr[i]
    end
    ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    
    # @info detratio
    return detratio, ΔE_Boson, nothing
end

@bm function accept_global!(mc::DQMC, m::Model, conf, passthrough)
    # for checking
    # new_G = finish_calculate_greens(
    #     mc.stack.Ul, mc.stack.Dl, mc.stack.Tl, mc.stack.Ur, mc.stack.Dr, mc.stack.Tr,
    #     mc.stack.greens_temp, mc.stack.pivot, mc.stack.tempv
    # )

    copyto!(mc.conf, conf)
    # Need a full stack rebuild
    reverse_build_stack(mc, mc.stack)
    # This calculates greens
    propagate(mc)

    # @info mc.stack.current_slice, mc.stack.direction
    # which should match new_G
    # display(new_G .- mc.stack.greens)
    # @assert new_G ≈ mc.stack.greens
    nothing
end



# This does a MC update with the given temp_conf as the proposed new_conf
function global_update(mc::DQMC, model::Model, temp_conf::AbstractArray)
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, temp_conf)

    p = exp(- ΔE_boson) * detratio
    @assert imag(p) == 0.0 "p = $p should always be real because ΔE_boson = $ΔE_boson and detratio = $detratio should always be real..."
    # @info p

    # Gibbs/Heat bath
    # p = p / (1.0 + p)
    # Metropolis
    if p > 1 || rand() < p
        accept_global!(mc, model, temp_conf, passthrough)
        return 1
    end

    return 0
end



################################################################################
### Update Interface
################################################################################



abstract type AbstractGlobalUpdate <: AbstractUpdate end



"""
    GlobalFlip([mc, model])

A global update that flips the configuration (±1 -> ∓1).
"""
struct GlobalFlip <: AbstractGlobalUpdate end
GlobalFlip(mc, model) = GlobalFlip()
name(::GlobalFlip) = "GlobalFlip"

@bm function update(u::GlobalFlip, mc, model)
    c = conf(mc)
    @. mc.temp_conf = -c
    return global_update(mc, model, mc.temp_conf)
end



"""
    GlobalShuffle([mc, model])

A global update that shuffles the current configuration. Note that this is not 
local to a time slice.
"""
struct GlobalShuffle <: AbstractGlobalUpdate end
GlobalShuffle(mc, model) = GlobalShuffle()
name(::GlobalShuffle) = "GlobalShuffle"


@bm function update(u::GlobalShuffle, mc, model)
    copyto!(mc.temp_conf, conf(mc))
    shuffle!(mc.temp_conf)
    return global_update(mc, model, mc.temp_conf)
end



"""
    SpatialShuffle([mc, model])

A global update that randomly swaps spatial indices of a configuration without 
changing the temporal indices. I.e. it may set 
`new_conf[i, :] = current_conf[j, :]` where the second index is a time slice 
index.
"""
struct SpatialShuffle <: AbstractGlobalUpdate 
    indices::Vector{Int}
    SpatialShuffle() = new(Vector{Int}(undef, 0))
end
SpatialShuffle(mc, model) = SpatialShuffle()
function init!(mc, u::SpatialShuffle)
    resize!(u.indices, length(lattice(mc)))
    u.indices .= 1:length(lattice(mc))
    nothing
end
name(::SpatialShuffle) = "SpatialShuffle"


@bm function update(u::SpatialShuffle, mc, model)
    shuffle!(u.indices)
    for slice in 1:nslices(mc), (i, j) in enumerate(u.indices)
        mc.temp_conf[i, slice] = mc.conf[j, slice]
    end
    return global_update(mc, model, mc.temp_conf)
end



"""
    TemporalShuffle([mc, model])

A global update that randomly swaps time slice indices of a configuration 
without changing the spatial indices. I.e. it may set 
`new_conf[:, k] = current_conf[:, l]` where the second index is a time slice 
index.
"""
struct TemporalShuffle <: AbstractGlobalUpdate 
    indices::Vector{Int}
    TemporalShuffle() = new(Vector{Int}(undef, 0))
end
TemporalShuffle(mc, model) = TemporalShuffle()
function init!(mc, u::TemporalShuffle)
    resize!(u.indices, nslices(mc))
    u.indices .= 1:nslices(mc)
    nothing
end
name(::TemporalShuffle) = "TemporalShuffle"


@bm function update(u::TemporalShuffle, mc, model)
    shuffle!(u.indices)
    for (k, l) in enumerate(u.indices), i in 1:length(lattice(mc))
        mc.temp_conf[i, k] = mc.conf[i, l]
    end
    return global_update(mc, model, mc.temp_conf)
end



"""
    Denoise([mc, model])

This global update attempts to remove noise, i.e. it attempts to build spatial
domains. This is done by setting each site to dominant surrounding value.
"""
struct Denoise <: AbstractGlobalUpdate end
Denoise(mc, model) = Denoise()
name(::Denoise) = "Denoise"


@bm function update(::Denoise, mc, model)
    for slice in 1:nslices(mc), i in 1:length(lattice(mc))
        average = mc.conf[i, slice]
        for j in neighbors(lattice(model), i)
            average += 2 * mc.conf[j, slice]
        end
        mc.temp_conf[i, slice] = sign(average)
    end
    return global_update(mc, model, mc.temp_conf)
end



"""
    DenoiseFlip([mc, model])

This update is similar to `Denoise` but sets each site to the opposite value to 
its surrounding.
"""
struct DenoiseFlip <: AbstractGlobalUpdate end
DenoiseFlip(mc, model) = DenoiseFlip()
name(::DenoiseFlip) = "DenoiseFlip"


@bm function update(::DenoiseFlip, mc, model)
    for slice in 1:nslices(mc), i in 1:length(lattice(mc))
        average = mc.conf[i, slice]
        for j in neighbors(lattice(model), i)
            average += 2 * mc.conf[j, slice]
        end
        mc.temp_conf[i, slice] = -sign(average)
    end
    return global_update(mc, model, mc.temp_conf)
end


"""
    StaggeredDenoise([mc, model])

This update is similar to `Denoise` but adds a multiplier based on lattice site
index. Even sites get multiplied by `+1`, odd sites by `-1`.
"""
struct StaggeredDenoise <: AbstractGlobalUpdate end
StaggeredDenoise(mc, model) = StaggeredDenoise()
name(::StaggeredDenoise) = "StaggeredDenoise"


@bm function update(::StaggeredDenoise, mc, model)
    for slice in 1:nslices(mc), i in 1:length(lattice(mc))
        average = mc.conf[i, slice]
        for j in neighbors(lattice(model), i)
            average += 2 * mc.conf[j, slice]
        end
        mc.temp_conf[i, slice] = (1 - 2 * (i % 2)) * sign(average)
    end
    return global_update(mc, model, mc.temp_conf)
end