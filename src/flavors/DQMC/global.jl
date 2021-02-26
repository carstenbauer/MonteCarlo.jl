################################################################################
### Linalg additions
################################################################################


@bm function calculate_inv_greens_udt(Ul, Dl, Tl, Ur, Dr, Tr, G, pivot, temp)
    vmul!(G, Tl, adjoint(Tr))
    vmul!(Tr, G, Diagonal(Dr))
    vmul!(G, Diagonal(Dl), Tr)
    udt_AVX_pivot!(Tr, Dr, G, pivot, temp, Val(false)) # Dl available

    vmul!(Tl, Ul, Tr)
    rdivp!(Ur, G, Ul, pivot) # requires unpivoted udt decompostion (Val(false))
    vmul!(Tr, adjoint(Tl), Ur)

    rvadd!(Tr, Diagonal(Dr))
    udt_AVX_pivot!(Ul, Dr, Tr, pivot, temp, Val(false)) # Dl available
    return Ul, Dr, Tr
end

# after the above without modifying Ur, Tr, Tl, Ul, Dr
@bm function finish_calculate_greens(Ul, Dl, Tl, Ur, Dr, Tr, G, pivot, temp)
    rdivp!(Ur, Tr, G, pivot) # requires unpivoted udt decompostion (false)
    vmul!(Tr, Tl, Ul)

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
        conf::AbstractArray = mc.conf, safe_mult::Int = mc.p.safe_mult
    )
    copyto!(mc.s.curr_U, I)
    copyto!(mc.s.Ur, I)
    mc.s.Dr .= one(eltype(mc.s.Dr))
    copyto!(mc.s.Tr, I)

    # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
    if slice+1 <= mc.p.slices
        start = slice+1
        stop = mc.p.slices
        for k in reverse(start:stop)
            if mod(k,safe_mult) == 0
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U, conf)
                vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dr))
                udt_AVX_pivot!(mc.s.curr_U, mc.s.Dr, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
                copyto!(mc.s.tmp2, mc.s.Tr)
                vmul!(mc.s.Tr, mc.s.tmp1, mc.s.tmp2)
            else
                multiply_daggered_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U, conf)
            end
        end
        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dr))
        udt_AVX_pivot!(mc.s.Ur, mc.s.Dr, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
        copyto!(mc.s.tmp2, mc.s.Tr)
        vmul!(mc.s.Tr, mc.s.tmp1, mc.s.tmp2)
    end


    copyto!(mc.s.curr_U, I)
    copyto!(mc.s.Ul, I)
    mc.s.Dl .= one(eltype(mc.s.Dl))
    copyto!(mc.s.Tl, I)

    # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
    if slice >= 1
        start = 1
        stop = slice
        for k in start:stop
            if mod(k,safe_mult) == 0
                multiply_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U, conf)
                vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dl))
                udt_AVX_pivot!(mc.s.curr_U, mc.s.Dl, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
                copyto!(mc.s.tmp2, mc.s.Tl)
                vmul!(mc.s.Tl, mc.s.tmp1, mc.s.tmp2)
            else
                multiply_slice_matrix_left!(mc, mc.model, k, mc.s.curr_U, conf)
            end
        end
        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.Dl))
        udt_AVX_pivot!(mc.s.Ul, mc.s.Dl, mc.s.tmp1, mc.s.pivot, mc.s.tempv)
        copyto!(mc.s.tmp2, mc.s.Tl)
        vmul!(mc.s.Tl, mc.s.tmp1, mc.s.tmp2)
    end

    calculate_inv_greens_udt(
        mc.s.Ul, mc.s.Dl, mc.s.Tl, mc.s.Ur, mc.s.Dr, mc.s.Tr, 
        mc.s.greens_temp, mc.s.pivot, mc.s.tempv
    )
    return mc.s.Dr
end


@bm function reverse_build_stack(mc::DQMC, ::DQMCStack)
    copyto!(mc.s.u_stack[end], I)
    mc.s.d_stack[end] .= one(eltype(mc.s.d_stack[end]))
    copyto!(mc.s.t_stack[end], I)

    @inbounds for i in length(mc.s.ranges):-1:1
        add_slice_sequence_right(mc, i)
    end

    mc.s.current_slice = 0
    mc.s.direction = 1

    nothing
end



################################################################################
### Global update (working)
################################################################################

#=
# function propose_global_from_green(mc::DQMC, m::Model, new_G::AbstractArray)
#     # weight = w_C' / w_C = det|G'| / det|G| = det|G' G^-1|
#     # det(new_G * inv(old_G))
#     inverted = inv!(mc.s.greens_temp, mc.s.greens)
#     mul!(mc.s.tmp1, new_G, inverted)
#     det(mc.s.tmp1) 
# end

function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray)
    # I don't think we need this...
    # @assert mc.s.current_slice == mc.p.slices + 1
    # @assert mc.s.direction == -1

    # @info current_slice(mc), mc.s.direction
    # calculate_greens(mc, mc.s.greens_temp) # we only care about getting an up to date Dl
    # G = mc.s.Ur * Diagonal(mc.s.Dl) * adjoint(mc.s.Tr)
    G = calculate_greens(mc, current_slice(mc)-1, mc.s.greens_temp)
    # display(G .- mc.s.greens)
    @assert G ≈ mc.s.greens
    old_weight = prod(mc.s.Dl)

    # weight = w_C' / w_C = det|G'| / det|G| = det|G' G^-1|
    # det(new_G * inv(old_G))
    new_greens = calculate_greens(mc, current_slice(mc)-1, mc.s.greens_temp, conf)
    new_weight = prod(mc.s.Dl)
    # inverted = inv!(mc.s.tmp1, mc.s.greens)
    # copyto!(mc.s.tmp1, mc.s.greens_temp)
    # try 
    #     inverted = LinearAlgebra.inv!(lu!(mc.s.tmp1))
    #     mul!(mc.s.tmp2, new_greens, inverted)
    # catch e
    #     println(mc.s.greens_temp)
    #     rethrow(e)
    # end
    # ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    # return det(mc.s.tmp2), ΔE_Boson, new_greens
    
    # detratio = exp(tr(log(new_greens)) - tr(log(mc.s.greens)))
    detratio = old_weight/new_weight
    ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    # @info old_weight, new_weight, detratio
    return detratio, ΔE_Boson, new_greens
end

# function accept_global!(mc::DQMC, m::Model, conf)
#     copyto!(mc.conf, conf)
#     # Need a full stack rebuild
#     build_stack(mc, mc.s)
#     # This calculates greens
#     propagate(mc)
#     # which should match new_G
#     # @assert new_G ≈ mc.s.greens
#     nothing
# end

function accept_global!(mc::DQMC, m::Model, conf, new_G)
    copyto!(mc.conf, conf)
    # Need a full stack rebuild
    build_stack(mc, mc.s)
    # This calculates greens
    propagate(mc)

    # @info mc.s.current_slice, mc.s.direction
    # which should match new_G
    # display(new_G .- mc.s.greens)
    @assert new_G ≈ mc.s.greens
    nothing
end
=#


################################################################################
### Global update (experimental)
################################################################################



@bm function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray)
    # I don't think we need this...
    @assert mc.s.current_slice == 1
    @assert mc.s.direction == 1

    # This should be just after calculating greens, so mc.s.Dl is from the UDT
    # decomposed G
    D = copy(mc.s.Dl)

    # -1?
    # TODO: 
    # this is essentially reverse_build_stack + partial calculate_greens
    # we need to do this to get a weight, so maybe we should
    # accept & deny_global
    # instead of just accept?
    inv_det(mc, current_slice(mc)-1, conf)

    # This may help with stability
    detratio = 1.0
    for i in eachindex(D)
        detratio *= D[i] * mc.s.Dr[i]
    end
    ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    
    # @info detratio
    return detratio, ΔE_Boson, nothing
end

@bm function accept_global!(mc::DQMC, m::Model, conf, passthrough)
    new_G = finish_calculate_greens(
        mc.s.Ul, mc.s.Dl, mc.s.Tl, mc.s.Ur, mc.s.Dr, mc.s.Tr,
        mc.s.greens_temp, mc.s.pivot, mc.s.tempv
    )

    copyto!(mc.conf, conf)
    # Need a full stack rebuild
    reverse_build_stack(mc, mc.s)
    # This calculates greens
    propagate(mc)

    # @info mc.s.current_slice, mc.s.direction
    # which should match new_G
    # display(new_G .- mc.s.greens)
    @assert new_G ≈ mc.s.greens
    nothing
end


# Alt
#=
function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray)
    # I don't think we need this...
    @assert mc.s.current_slice == 1
    @assert mc.s.direction == 1

    # This should be just after calculating greens, so mc.s.Dl is from the UDT
    # decomposed G
    D = copy(mc.s.Dl)

    # well this but with a conf given
    reverse_build_stack(mc, mc.s)
    propagate(mc)
    # e.g. rewrite inv_det above to overwrite stack?
    # can probably use the split calculate_greens

    # This may help with stability
    detratio = 1.0
    for i in eachindex(D)
        detratio *= D[i] * mc.s.Dr[i]
    end
    ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    
    @info detratio
    return detratio, ΔE_Boson, nothing
end

function accept_global!(mc::DQMC, m::Model, conf, passthrough)
    copyto!(mc.conf, conf)
    # maybe move propagate here?
    nothing
end

function deny_global!(mc::DQMC, m::Model, conf, passthrough)
    new_greens = finish_calculate_greens(
        mc.s.Ul, mc.s.Dl, mc.s.Tl, mc.s.Ur, mc.s.Dr, mc.s.Tr,
        mc.s.greens_temp, mc.s.pivot, mc.s.tempv
    )
    # Need a full stack rebuild
    reverse_build_stack(mc, mc.s)
    # This calculates greens
    propagate(mc)
    nothing
end
=#


################################################################################
### Global flip
################################################################################


# TODO
using Random
@bm function global_move(mc, model)
    conf = shuffle(mc.conf)
    detratio, ΔE_boson, new_greens = propose_global_from_conf(mc, model, conf)

    if mc.p.check_sign_problem
        if abs(imag(detratio)) > 1e-6
            push!(mc.a.imaginary_probability, abs(imag(detratio)))
            mc.p.silent || @printf(
                "Did you expect a sign problem? imag. detratio:  %.9e\n", 
                abs(imag(detratio))
            )
        end
        if real(detratio) < 0.0
            push!(mc.a.negative_probability, real(detratio))
            mc.p.silent || @printf(
                "Did you expect a sign problem? negative detratio %.9e\n",
                real(detratio)
            )
        end
    end
    p = real(exp(- ΔE_boson) * detratio)
    # @info p

    # Gibbs/Heat bath
    # p = p / (1.0 + p)
    # Metropolis
    if p > 1 || rand() < p
        accept_global!(mc, model, conf, new_greens)
        return 1
    end

    return 0
end


################################################################################
### Replica Exchange
################################################################################

#=
mutable struct ReplicaExchange
    parent::DQMC

    needs_recalculation::Bool
    connected_ids::Vector{Int}  # All process IDs to exchange data with
    cycle_idx::Int              # idx in connected_ids - we wanna speak to one

    updated::Bool
    new_greens::AbstractArray
    conf::AbstractArray

    ReplicaExchange() = new()
end

const replica_exchange = ReplicaExchange()

function ReplicaExchange(mc::DQMC, recalculate=true)
    replica_exchange.parent = mc
    replica_exchange.needs_recalculation = recalculate
    replica_exchange.connected = 0

    replica_exchange.updated = false
    replica_exchange.greens = copy(dqmc.s.greens)
    replica_exchange.conf = copy(dqmc.conf)

    replica_exchange
end

# Tell remote that local is a valid target
function connect!(dqmc, target)
    remotecall(_connect!, target, myid())
end
function _connect!(target)
    push!(replica_exchange.connected_ids, target)
end

# Tell remote that local is no longer a valid target
function disconnect!()
    for target in replica_exchange.needs_recalculation
        remotecall(_disconnect!, target, myid())
    end
    return
end
function _disconnect!(target)
    delete!(replica_exchange.connected_ids, target)
    return
end

# Give remote some local data
function send!(mc::DQMC)
    re = replica_exchange
    if re.needs_recalculation
        send!(mc.conf)
    else
        send!(mc.conf, mc.s.greens)
    end
end

function send!(conf::AbstractArray)
    re = replica_exchange
    if !isempty(re.connected_ids)
        target = re.connected_ids[re.cycle_idx]
        re.cycle_idx = mod1(re.cycle_idx + 1, length(re.connected_ids))
        remotecall(_set_replica_conf!, target, conf)
        return true
    end
    return false
end

function send!(conf::AbstractArray, greens::AbstractArray)
    re = replica_exchange
    if !isempty(re.connected_ids)
        target = re.connected_ids[re.cycle_idx]
        re.cycle_idx = mod1(re.cycle_idx + 1, length(re.connected_ids))
        remotecall(_set_replica_conf_and_greens!, target, conf, greens)
        return true
    end
    return false
end

function _set_replica_conf!(conf)
    copyto!(replica_exchange.conf, conf)
    replica_exchange.updated = true
    return
end

function _set_replica_conf_and_greens!(conf, greens)
    # Pretty sure this doesn't do anything but better save than sorry
    @sync begin
        copyto!(replica_exchange.conf, conf)
        copyto!(replica_exchange.greens, greens)
    end
    replica_exchange.updated = true
    return
end

function replica_exchange_update(mc::DQMC)
    re = replica_exchange
    re.updated || return
    p = if re.needs_recalculation
        propose_global_from_conf(mc, mc.model, re.conf)
    else
        propose_global_from_green(mc, mc.model, re.greens)
    end
    if p > 1.0 || p > rand()
        if re.needs_recalculation
            accept_global!(mc, mc.model, re.conf)
        else
            accept_global!(mc, mc.model, re.greens, re.conf)
        end
    end
    return
end
=#


# Oh dear Oh boy
# Big fucking nope
#=
function compute_ratio(mc, conf)
    # U D T
    copyto!(mc.s.u_stack[1], I)
    mc.s.d_stack[1] .= one(eltype(mc.s.d_stack[1]))
    copyto!(mc.s.t_stack[1], I)

    @inbounds for idx in 1:length(mc.s.ranges)
        copyto!(mc.s.curr_U, mc.s.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
        for slice in mc.s.ranges[idx]
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, mc.s.curr_U)
        end

        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.d_stack[idx]))
        udt_AVX_pivot!(
            mc.s.u_stack[idx + 1], mc.s.d_stack[idx + 1], mc.s.tmp1, 
            mc.s.pivot, mc.s.tempv
        )
        vmul!(mc.s.t_stack[idx + 1], mc.s.tmp1, mc.s.t_stack[idx])
    end
    U = copy(mc.s.u_stack[end])
    D = copy(mc.s.d_stack[end])
    T = copy(mc.s.t_stack[end])

    # Ul Dl Tl is the same
    Ul = copy(mc.s.u_stack[end])
    Dl = copy(mc.s.d_stack[end])
    Tl = copy(mc.s.t_stack[end])

    # Ur Dr Tr from new greens
    mc.conf .= conf
    copyto!(mc.s.u_stack[1], I)
    mc.s.d_stack[1] .= one(eltype(mc.s.d_stack[1]))
    copyto!(mc.s.t_stack[1], I)

    @inbounds for idx in 1:length(mc.s.ranges)
        copyto!(mc.s.curr_U, mc.s.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.s.ranges[idx])
        for slice in mc.s.ranges[idx]
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.s.curr_U)
        end

        vmul!(mc.s.tmp1, mc.s.curr_U, Diagonal(mc.s.d_stack[idx]))
        udt_AVX_pivot!(
            mc.s.u_stack[idx + 1], mc.s.d_stack[idx + 1], mc.s.tmp1, 
            mc.s.pivot, mc.s.tempv
        )
        vmul!(mc.s.t_stack[idx + 1], mc.s.tmp1, mc.s.t_stack[idx])
    end
    Ur = copy(mc.s.u_stack[end])
    Dr = copy(mc.s.d_stack[end])
    Tr = copy(mc.s.t_stack[end])

    # new G
    build_stack(mc, mc.s)
    propagate(mc)

    mc.s.greens + compute_ratio(U, D, T, Ul, Dl, Tl, Ur, Dr, Tr)
end

function compute_ratio(
        U, D, T, Ul, Dl, Tl, Ur, Dr, Tr, 
        greens = similar(U), tmp1 = similar(greens), 
        pivot = zeros(Int, length(D)), tempv = similar(D)
    )
    @bm "compute G" begin
        # [B_{l+1}^-1 B_{l+2}^-1 ⋯ B_k^-1 + B_l ⋯ B_1 B_N ⋯ B_{k+1}]^-1
        # [U D T + Ul (Dl Tl Tr^† Dr) Ur^†]^-1
        @bm "B1" begin
            vmul!(greens, Tl, adjoint(Tr))
            vmul!(tmp1, greens, Diagonal(Dr))
            vmul!(greens, Diagonal(Dl), tmp1)
        end
        # [U D T + Ul (G) Ur^†]^-1
        @bm "udt" begin
            udt_AVX_pivot!(Tr, Dr, greens, pivot, tempv, Val(false))
        end
        # [U D T + (Ul Tr) Dr (G Ur^†)]^-1
        @bm "B2" begin
            vmul!(Tl, Ul, Tr)
            # (G Ur^†) = (Ur / G)^-1
            # Ur := Ur / G
            rdivp!(Ur, greens, Ul, pivot) 
        end
        # [U D T + Tl Dr Ur^-1]^-1
        # [U (D T Ur + U^† Tl Dr) Ur^-1]^-1
        # [U D_max (D_min T Ur 1/Dr_max + 1/D_max U^† Tl Dr_min) Dr_max Ur^-1]^-1
        @bm "B3" begin
            # 1/D_max U^† Tl Dr_min
            vmul!(Tr, adjoint(U), Tl)
            vmaxinv!(Dl, D) # Dl .= 1.0 ./ max.(1.0, D)
            vmul!(tmp1, Diagonal(Dl), Tr)
            vmin!(Dl, Dr) # Dl .= min.(1.0, Dr)
            vmul!(Tr, tmp1, Diagonal(Dl))
        end
        # [U D_max (D_min T Ur 1/Dr_max + Tr) Dr_max Ur^-1]^-1
        @bm "B4" begin
            # D_min T Ur 1/Dr_max
            vmul!(Tl, T, Ur)
            vmin!(Dl, D) # Dl .= min.(1.0, D)
            vmul!(tmp1, Diagonal(Dl), Tl)
            vmaxinv!(Dl, Dr) # Dl .= 1.0 ./ max.(1.0, Dr)
            vmul!(Tl, tmp1, Diagonal(Dl))
        end
        # [U D_max (Tl + Tr) Dr_max Ur^-1]^-1
        @bm "sum, UDT" begin
            rvadd!(Tl, Tr)
            udt_AVX_pivot!(Tr, Dl, Tl, pivot, tempv, Val(false))
        end
        # [U D_max (Tr Dl Tl) Dr_max Ur^-1]^-1
        # Ur 1/Dr_max Tl^-1 1/Dl Tr^† D_max U^†
        @bm "B5" begin
            # [[((1/Dr_max) / Tl) 1/Dl] Tr^†] D_max
            vmaxinv!(Dr, Dr) # Dr .= 1.0 ./ max.(1.0, Dr)
            copyto!(Ul, Diagonal(Dr))
            rdivp!(Ul, Tl, tmp1, pivot)
            vinv!(Dl) # Dl[i] .= 1.0 ./ Dl[i]
            vmul!(tmp1, Ul, Diagonal(Dl))
            vmul!(Ul, tmp1, adjoint(Tr))
            vmaxinv!(Dl, D) # Dl .= 1.0 ./ max.(1.0, D)
            vmul!(greens, Ul, Diagonal(Dl))
        end
        # Ur G U^†
        @bm "B6" begin
            vmul!(Tr, greens, adjoint(U))
            vmul!(greens, Ur, Tr)
        end
    end
    greens
end
=#