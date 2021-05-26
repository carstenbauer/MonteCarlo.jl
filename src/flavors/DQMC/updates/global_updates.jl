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
    @avx for i in eachindex(Dr)
        Dl[i] = 1.0 / Dr[i]
    end
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


@bm function reverse_build_stack(mc::DQMC, ::DQMCStack)
    copyto!(mc.stack.u_stack[end], I)
    mc.stack.d_stack[end] .= one(eltype(mc.stack.d_stack[end]))
    copyto!(mc.stack.t_stack[end], I)

    @inbounds for i in length(mc.stack.ranges):-1:1
        add_slice_sequence_right(mc, i)
    end

    mc.stack.current_slice = 0
    mc.stack.direction = 1

    nothing
end



################################################################################
### Global update (working)
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
    @info p

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
### Updates
################################################################################



# TODO: rewrite
"""
    AbstractGlobalUpdate

A global update should be a struct inhereting from AbstractGlobalUpdate, i.e.

```
struct MyGlobalUpdate <: AbstractGlobalUpdate
    ...
end
```

with whatever fields are required. It should implement a method

```
function global_update(u::MyGlobalUpdate, mc, model, temp_conf, temp_vec)
    temp_conf = ...
    return global_update(mc, model, temp_conf, temp_vec)
end
```

which performs the update by creating a new conf in `temp_conf` (which you may
overwrite and assume to be overwritten before the next call) and passing that 
to the standard global Metropolis update implented in 
`global_update(mc, model, temp_conf, temp_vec)`. 

Behind the scenes, the scheduler may wrap `MyGlobalUpdate` in 
`AcceptanceStatistics` which collects the number of requested and accepted 
updates. It is expected that you return `0` if the update is denied or `1` if it
is accepted (as does the `global_update` returned above). 
"""
abstract type AbstractGlobalUpdate end
Base.show(io::IO, u::AbstractGlobalUpdate) = print(io, name(u))



"""
    NoUpdate([mc, model], [sampling_rate = 1e-10])

A global update that does nothing. Mostly used internally to keep the adaptive 
scheduler running if all (other) updates are ignored.
"""
struct NoUpdate <: AbstractGlobalUpdate end
NoUpdate(mc, model) = NoUpdate()
function global_update(u::NoUpdate, args...)
    # we count this as "denied" global update
    return 0
end
name(::NoUpdate) = "NoUpdate"



"""
    GlobalFlip([mc, model], [sampling_rate = 0.5])

A global update that flips the configuration (±1 -> ∓1).
"""
struct GlobalFlip <: AbstractGlobalUpdate end
GlobalFlip(mc, model) = GlobalFlip()
name(::GlobalFlip) = "GlobalFlip"

function global_update(u::GlobalFlip, mc, model, temp_conf)
    c = conf(mc)
    @. temp_conf = -c
    return global_update(mc, model, temp_conf)
end



"""
    GlobalShuffle([mc, model, [sampling_rate = 0.5])

A global update that shuffles the current configuration. Note that this is not 
local to a time slice.
"""
struct GlobalShuffle <: AbstractGlobalUpdate end
GlobalShuffle(mc, model) = GlobalShuffle()
name(::GlobalShuffle) = "GlobalShuffle"


function global_update(u::GlobalShuffle, mc, model, temp_conf)
    copyto!(temp_conf, conf(mc))
    shuffle!(temp_conf)
    return global_update(mc, model, temp_conf)
end