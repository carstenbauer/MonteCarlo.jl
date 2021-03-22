# Design
# call stack:
# global_update(scheduler, mc, model)
#   - figures out which update to use
#   > global_update(update, mc, model, temp_conf, temp_vec)
#       - generates a conf
#       > global_update(mc, model, temp_conf, temp_vec)
#           > propose_global
#           - do the Metropolis
#           > accept_global

# types:
# <: AbstractUpdateScheduler
# Holds a bunch of updates and iterates through them in some way
# <: AbstractGlobalUpdate
# Generates a new configuration

# Note:
# ReplicaExchange fits here as an AbstractGlobalUpdate

# TODO
# - Should each update collect statistics? Should DQMC not?
#   - Since we can mix updates it might be nice to have the global stats as well
# - FileIO



################################################################################
### Scheduler
################################################################################



abstract type AbstractUpdateScheduler end



"""
    EmptyScheduler()

Schedules no global updates.
"""
struct EmptyScheduler <: AbstractUpdateScheduler end
global_update(::EmptyScheduler, args...) = 0



"""
    SimpleScheduler(mc, updates...)

Schedules global updates in the order specified through `updates...`. For example
`SimpleScheduler(mc, GlobalFlip(), GlobalFlip(), GlobalShuffle())` would generate
a sequence `GlobalFlip() > GlobalFlip() > GlobalShuffle() > repeat`.
"""
mutable struct SimpleScheduler{CT, T <: Tuple} <: AbstractUpdateScheduler
    # temporary
    conf::CT
    temp::Vector{Float64}
    sequence::T
    idx::Int64
end

function SimpleScheduler(mc::DQMC, updates...)
    N = length(lattice(mc))
    flv = nflavors(mc.model)
    SimpleScheduler(copy(conf(dqmc)), zeros(Float64, flv*N), updates, 0)
end

function global_update(s::SimpleScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    global_update(s.sequence[s.idx], mc, model, s.conf, s.temp)
end



"""
    AdaptiveScheduler

DO NOT USE!

The idea with this is to adaptive adjust how frequently global updates are used.
With this you could just throw a bunch of global updates at a problem and let 
the simulation figure out which ones are worth doing.

Notes:
- low acceptance rate <=> bad update => do it less
- this needs a "grace period" to collect stats before starting to adjust rates
- this should probably adjust asymptotically?
- one should be able to exclude updates (e.g. ReplicaExchange)

Draft:
Add wrapper for updates that should have an adaptive sampling rate
    mutable struct AdaptiveUpdate{T}
        accepted::Int
        total::Int
        sampling_rate::Float64
        update::T
    end

everyone starts with a sampling_rate of 1.0 (or whatever is specified)
pick update according to sampling rates:
    p = (sr1 + sr2 + ... srN) * rand()
    cumsum = 0.0
    for update in adaptive_updates
        cumsum += update.sampling_rate
        if p < cumsum
            # perform update
            break
        end
    end

The sequence and the updates we do are somewhat disconnected... We probably want 
something like
    AdaptiveScheduler(
        mc, 
        (Adaptive(), Adaptive(), ReplicaExchange()), 
        (StaticUpdate(NoUpdate(), 0.1), GlobalShuffle(), GlobalFlip())
    )
where we need something like "StaticUpdate" specifically for NoUpdate...? Or 
maybe we can special-case this in the first place and give it a sampling rate?
"""
mutable struct AdaptiveScheduler{CT, T1, T2} <: AbstractUpdateScheduler
    # below this, the sampling rate is set to 0. Basing this on sampling rate
    # should allow us to make things a bit smoother and avoid bad luck.
    minimum_sampling_rate::Float64 # 0.01?
    # some number of sweeps where we just collect statistics
    grace_period::Int # 100?
    # Weight for how slowly sampling_rate changes.
    # sampling_rate = (adaptive_rate * sampling_rate + accepted/total) / (adaptive_rate+1)
    # 0 = use current probability as sampling rate
    adaptive_rate::Float64 # 9.0?

    conf::CT
    temp::Vector{Float64}

    adaptive_pool::T1
    sequence::T2
    idx::Int
end
struct Adaptive end

function AdaptiveScheduler(
        mc, sequence, pool;
        minimum_sampling_rate = 0.01, grace_period = 99, adaptive_rate = 9.0
    )

    adaptive_pool = map(pool) do update
        if update isa NoUpdate || update isa AdaptiveUpdate
            return update
        else
            AdaptiveUpdate(update)
        end
    end

    if !(NoUpdate() in adaptive_pool)
        adaptive_pool = tuple(adaptive_pool..., NoUpdate())
    end

    N = length(lattice(mc))
    flv = nflavors(mc.model)

    AdaptiveScheduler(
        minimum_sampling_rate, grace_period, adaptive_rate,
        similar(conf(mc)), zeros(Float64, flv*N),
        Tuple(adaptive_pool), Tuple(sequence), 0
    )
end

function global_update(s::AdaptiveScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    
    if s.sequence[s.idx] === Adaptive()
        # Find appropriate (i.e. with probability matching sampling rates) 
        # update from adaptive pool
        total_weight = mapreduce(x -> x.sampling_rate, +, s.adaptive_pool)
        p = total_weight * rand()
        sum = 0.0

        for update in s.adaptive_pool
            sum += update.sampling_rate
            
            # if sum goes beyond p we have an appropriate update
            if p < sum
                accepted = global_update(update, mc, model, s.conf, s.temp)

                # start adjusting sampling rate if we have sampled more than 
                # <grace_period> times
                if update.total > s.grace_period
                    update.sampling_rate = (
                        s.adaptive_rate * update.sampling_rate + 
                        update.accepted / update.total
                    ) / (s.adaptive_rate + 1)
                    
                    # Hit miniomum threshold - this can no longer be accepted.
                    if update.sampling_rate < s.minimum_sampling_rate
                        update.sampling_rate = 0.0
                    end
                end

                return accepted
            end
        end
    else
        # Some sort of non-adaptive update, just perform it.
        return global_update(s.sequence[s.idx], mc, model, s.conf, s.temp)
    end
end



################################################################################
### Updates
################################################################################



abstract type AbstractGlobalUpdate end


"""
    AdaptiveUpdate(update[, expected_acceptance_rate = 0.5])

Wraps any generic global update, keeps track of the number of update attempts 
and the number of successes for adaptive sampling. This should be used with the
adaptive scheduler but shouldn't create any problems otherwise.
"""
struct AdaptiveUpdate{T}
    accepted::Int
    total::Int
    sampling_rate::Float64
    update::T
end
function AdaptiveUpdate(update::AbstractGlobalUpdate, sampling_rate=0.5)
    AdaptiveUpdate(0, 0, sampling_rate, update)
end
@inline function global_update(u::AdaptiveUpdate, mc, m, tc, tv)
    u.total += 1
    accepted = global_update(u.update, mc, m, tc, tv)
    u.accepted += accepted
    return accepted
end



"""
    NoUpdate([mc, model], [sampling_rate = 1e-10])

A global update that does nothing. Mostly used internally to keep the adaptive 
scheduler running if all (other) updates are ignored.
"""
mutable struct NoUpdate <: AbstractGlobalUpdate
    accepted::Int
    total::Int
    sampling_rate::Float64
end
NoUpdate(sampling_rate = 1e-10) = NoUpdate(0, -1, sampling_rate)
NoUpdate(mc, model, sampling_rate=1e-10) = NoUpdate(sampling_rate)
function global_update(u::NoUpdate, args...)
    # We don't want to update total because that'll eventually trigger the
    # adaptive process. But we may still want to know how often this was used...
    # So let's keep track of that in accepted instead
    u.accepted += 1
    # we count this as "denied global update
    return 0
end


"""
    GlobalFlip([mc, model])

A global update that flips the configuration (±1 -> ∓1).
"""
struct GlobalFlip <: AbstractGlobalUpdate end
GlobalFlip(mc, model) = GlobalFlip()

function global_update(::GlobalFlip, mc, model, temp_conf, temp_vec)
    @. temp_conf = -conf(mc)
    global_update(mc, model, temp_conf, temp_vec)
end



"""
    GlobalShuffle([mc, model])

A global update that shuffles the current configuration. Note that this is not 
local to a time slice.
"""
struct GlobalShuffle <: AbstractGlobalUpdate end
GlobalShuffle(mc, model) = GlobalShuffle()

function global_update(::GlobalShuffle, mc, model, temp_conf, temp_vec)
    copyto!(temp_conf, conf(mc))
    shuffle!(temp_conf)
    global_update(mc, model, temp_conf, temp_vec)
end



# I.e. we trade or not
# struct ReplicaExchange <: AbstractGlobalUpdate end 

# I.e. we give confs to each other and do things independently
# struct ReplicaGift <: AbstractGlobalUpdate



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



@bm function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray, temp)
    # I don't think we need this...
    @assert mc.s.current_slice == 1
    @assert mc.s.direction == 1

    # This should be just after calculating greens, so mc.s.Dl is from the UDT
    # decomposed G
    # We need an independent temp vector here as inv_det changes Dl, Dr and tempv
    temp .= mc.s.Dl

    # This is essentially reverse_build_stack + partial calculate_greens
    # after this: G = Ur Tr^-1 Dr^-1 Ul^† Tl^†
    # where Ur, Ul, Tl only contribute complex phases and Tr contributes 1
    # Since we are already checking for sign problems in local updates we ignore
    # the phases here and set det(G) = 1 / det(Dr)
    inv_det(mc, current_slice(mc)-1, conf)

    # This loop helps with stability - it multiplies large and small numbers
    # whihc avoid reaching extremely large or small (typemin/max) floats
    detratio = 1.0
    for i in eachindex(temp)
        detratio *= temp[i] * mc.s.Dr[i]
    end
    ΔE_Boson = energy_boson(mc, m, conf) - energy_boson(mc, m)
    
    # @info detratio
    return detratio, ΔE_Boson, nothing
end

@bm function accept_global!(mc::DQMC, m::Model, conf, passthrough)
    # for checking
    # new_G = finish_calculate_greens(
    #     mc.s.Ul, mc.s.Dl, mc.s.Tl, mc.s.Ur, mc.s.Dr, mc.s.Tr,
    #     mc.s.greens_temp, mc.s.pivot, mc.s.tempv
    # )

    copyto!(mc.conf, conf)
    # Need a full stack rebuild
    reverse_build_stack(mc, mc.s)
    # This calculates greens
    propagate(mc)

    # @info mc.s.current_slice, mc.s.direction
    # which should match new_G
    # display(new_G .- mc.s.greens)
    # @assert new_G ≈ mc.s.greens
    nothing
end



# This does a MC update with the given temp_conf as the proposed new_conf
function global_update(mc::DQMC, model::Model, temp_conf::AbstractArray, temp_vec::Vector{Float64})
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, temp_conf, temp_vec)

    p = exp(- ΔE_boson) * detratio
    @assert imag(p) == 0.0 "p = $p should always be real because ΔE_boson = $ΔE_boson and detratio = $detratio should always be real..."

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
### to be deleted
################################################################################



@bm function global_move(mc, model)
    conf = shuffle(mc.conf)
    detratio, ΔE_boson, new_greens = propose_global_from_conf(mc, model, conf, similar(mc.s.Dr))

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