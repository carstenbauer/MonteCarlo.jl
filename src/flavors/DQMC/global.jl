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


# parallel
# I think we can make good use of the "trick" TimerOutput uses for debug 
# timings - redefining a function.
# In this case we can have a send/receive that does nothing originally, and gets
# changed when a global updater is created, referencing its conf.
# We don't need this for yield(), that can be based on a simple variable...


################################################################################
### Parallel
################################################################################

# Notes (E - everywhere/every worker, L - local/this worker)
# [E] @everywhere using MonteCarlo (this is blocking)
# [E] => creates constants
# [L] create MC, updates, scheduler
# [L] => scheduler generates get_conf
# [L] => notify connected workers (write to constants)
# [L] yield to recieve notfications
# [L] run simulation, yield once per sweep

# Look through these to find a partner
const connected_ids = Vector{Int}(undef, 0)

# process local
add_worker!(id) = begin in(id, connected_ids) || push!(connected_ids, id); true end
remove_worker!(id) = begin filter!(x -> x != id, connected_ids); true end
# this should be replaced when a parallel update is created
get_conf(args...) = false

# multi-process communication
# don't care if they return, we're just saying "call me maybe?"
connect(targets...; wait = false) = connect(collect(targets), wait = wait)
function connect(targets::Vector; wait = false)
    if wait
        @sync for target in targets
            @async remotecall_wait(add_worker!, target, myid())
        end
    else
        for target in targets
            remotecall(add_worker!, target, myid())
        end
    end
    nothing
end
# here we do care, because otherwise remote might be requesting from us when we're gone
function disconnect(targets...)
    # maybe this could just be a @sync...
    # I'm not sure if that would block other tasks though, like a remote saying "Hey diconnect me"
    done = zeros(Bool, length(targets))
    for (i, target) in enumerate(targets)
        @async done[i] = remotecall_fetch(remove_worker!, target, myid())
    end
    while !all(done)
        yield()
    end
    nothing
end
pull_conf_from_remote(trg) = remotecall_fetch(get_conf, trg)


const weight_probability = Channel{Tuple{Int, Float64, Float64}}(1)
function put_weight_prob!(id, weight, probability)
    put!(weight_probability, (id, weight, probability))
    nothing
end


# 2 process barrier
const wait_for = Channel{Int}(1)
waiting_on() = isready(wait_for) ? take!(wait_for) : -1
function wait_for_remote(id)
    put!(wait_for, id)
    maybe_me = remotecall_fetch(waiting_on, id)

    if maybe_me != myid()
        while !isempty(wait_for)
            yield()
        end
    else
        yield()
        isready(wait_for) && take!(wait_for)
    end

    nothing
end

function generate_communication_functions(local_conf)
    @eval MonteCarlo begin
        get_conf() = $local_conf
    end
end



# Ok so...
# - since parallel and non parallel global updates maybe mixed it's not reasonable
#   to write to the conf in scheduler
# - since we could have multiple parallel updates replacing receive_conf! is also
#   not valid
# so I guess everything needs to be pull based, originating from the parallel
# update...

# replica exchange
# - pull_conf
# - compute weight
# - pull weight, random
# - compute full weight and compare to random from smaller id (or larger, whatever)

# weight and random requires a global Channel{Tuple{Float64, Float64}(1) (or maybe 0?)
# push on remote first, then wait of take!()


################################################################################
### Scheduler
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
    EmptyScheduler()

Schedules no global updates.
"""
struct EmptyScheduler <: AbstractUpdateScheduler
    EmptyScheduler(args...; kwargs...) = new()
end
global_update(::EmptyScheduler, args...) = 0
show_statistics(::EmptyScheduler, prefix="") = nothing
function save_scheduler(file::JLDFile, s::EmptyScheduler, entryname::String="/Scheduler")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(s))
    nothing
end
_load(data, ::Type{<: EmptyScheduler}, ::Type, model) = EmptyScheduler()



"""
    SimpleScheduler(::Type{<: DQMC}, model, updates...)

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

    # For loading
    function SimpleScheduler(conf::T1, temp, sequence::T2, idx) where {T1, T2}
        new{T1, T2}(conf, temp, sequence, idx)
    end

    function SimpleScheduler(::Type{<: DQMC}, model::Model, updates::AbstractGlobalUpdate...)
        dummy = rand(DQMC, model, 1)
        updates = map(AcceptanceStatistics, updates)
        obj = new{typeof(dummy), typeof(updates)}()
        obj.sequence = updates
        obj.idx = 0
        obj
    end
end


function init!(s::SimpleScheduler, mc::DQMC, model::Model)
    N = length(lattice(mc))
    flv = nflavors(mc.model)
    s.conf = copy(conf(mc))
    s.temp = zeros(Float64, flv*N)
    generate_communication_functions(mc.conf)
    s
end

function global_update(s::SimpleScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    global_update(s.sequence[s.idx], mc, model, s.conf, s.temp)
end

function show_statistics(s::SimpleScheduler, prefix="")
    println(prefix, "Global update statistics (since start):")
    cum_accepted = 0
    cum_total = 0
    for update in s.sequence
        update isa AcceptanceStatistics{NoUpdate} && continue
        @printf(
            "%s\t%s %0.1f%s accepted (%i / %i)\n",
            prefix, rpad(name(update), 20), 100update.accepted/max(1, update.total), 
            "%", update.accepted, update.total
        )
        cum_accepted += update.accepted
        cum_total += update.total
    end
    @printf(
        "%s\t%s %0.1f%s accepted (%i / %i)\n",
        prefix, rpad("Total", 20), 100cum_accepted/cum_total, 
        "%", cum_accepted, cum_total
    )
    nothing
end

function Base.show(io::IO, s::SimpleScheduler)
    sequence = mapreduce(
        i -> name(s.sequence[mod1(i, length(s.sequence))]),
        (a, b) -> "$a -> $b",
        s.idx+1 : s.idx+length(s.sequence)
    )
    print(io, "SimpleScheduler(): $sequence -> (repeat)")
end

function save_scheduler(file::JLDFile, s::SimpleScheduler, entryname::String="/Scheduler")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(s))
    write(file, entryname * "/sequence", s.sequence)
    write(file, entryname * "/idx", s.idx)
    nothing
end
function _load(data, ::Type{<: SimpleScheduler}, ::Type{<: DQMC}, model)
    s = SimpleScheduler(DQMC, model, data["sequence"], data["pool"])
    s.idx = data["idx"]
    s
end



"""
    Adaptive()

A placeholder for adaptive updates in the `AdaptiveScheduler`.
"""
struct Adaptive end
name(::Adaptive) = "Adaptive"



"""
    AdaptiveScheduler(::Type{<: DQMC}, model::Model, sequence::Tuple, pool::Tuple; kwargs...])

Creates an adaptive scheduler which prioritizes updates that succeed more 
frequently. 

`sequence` defines the static sequence of global updates, with 
`Adaptive()` as a proxy for adaptive updates. When polling the scheduler 
`Adaptive()` will be replaced by one of the updates in pool based on their 
success rate.

Example:
```
AdaptiveScheduler(
    DQMC, model, 
    (Adaptive(), ReplicaExchange()), 
    (GlobalFlip(), GlobalShuffle())
)
```
This setup will perform one of `GlobalFlip()` and `GlobalShuffle()` followed by 
a `ReplicaExchange()` update, repeating after that. The latter will always be a 
`ReplicaExchange()` while the former will happen based on the relative 
sampling rates which are sluggishly derived from acceptance rates. 

If those sampling rates fall below a certain threshhold they may be set to 0. 
If all updates drop to 0 acceptance rate, `NoUpdate()` is performed instead. 
(`NoUpdate()` is always added and has a static probability of `1e-10`.)

### Keyword Arguments:
- `minimum_sampling_rate = 0.01`: This defines the threshhold under which the 
sampling rate is set to 0.
- `grace_period = 99`: This sets a minimum number of times an update needs to 
be called before its sampling rate is adjusted. 
- `adaptive_rate = 9.0`: Controls how fast the sampling rate is adjusted to the 
acceptance rate. More means slower. This follows the formula 
`(adaptive_rate * sampling_rate + accepted/total) / (adaptive_rate + 1)`
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
    sampling_rates::Vector{Float64}
    sequence::T2
    idx::Int

    # for loading
    function AdaptiveScheduler(msr, gp, ar, c::CT, t, ap::T1, sr, s::T2, i) where {CT, T1, T2}
        new{CT, T1, T2}(msr, gp, ar, c, t, ap, sr, s, i)
    end

    function AdaptiveScheduler(
            ::Type{<: DQMC}, model::Model, sequence, pool;
            minimum_sampling_rate = 0.01, grace_period = 99, adaptive_rate = 9.0
        )

        # sampling rate x after i steps with constant acceptance rate p:
        # x_i = (c/c+1)^i x_0 + a * \sum_{i} c^{i-1} / (c+1)^i
        # where c is the adaptive rate and x_0 is the initial samplign rate.
        # Minimum time to discard:
        i_min = ceil(Int, log(
            adaptive_rate / (adaptive_rate+1), 
            minimum_sampling_rate / 0.5
        ))
        @info("Minimum number of samples for discard: $grace_period + $i_min")

        if !any(x -> x isa NoUpdate, pool)
            pool = tuple(pool..., NoUpdate())
        end

        # Wrap in AcceptanceStatistics
        adaptive_pool = map(AcceptanceStatistics, pool)
        sequence = map(AcceptanceStatistics, sequence)

        adaptive_pool = Tuple(adaptive_pool)
        sequence = Tuple(sequence)
        dummy = rand(DQMC, model, 1)

        obj = new{typeof(dummy), typeof(adaptive_pool), typeof(sequence)}()
        obj.minimum_sampling_rate = minimum_sampling_rate
        obj.grace_period = grace_period
        obj.adaptive_rate = adaptive_rate
        obj.adaptive_pool = adaptive_pool
        obj.sampling_rates = [u isa AcceptanceStatistics{NoUpdate} ? 1e-10 : 0.5 for u in adaptive_pool]
        obj.sequence = sequence
        obj.idx = 0

        obj
    end
end

function init!(s::AdaptiveScheduler, mc::DQMC, model::Model)
    N = length(lattice(mc))
    flv = nflavors(mc.model)
    s.conf = copy(conf(mc))
    s.temp = zeros(Float64, flv*N)
    generate_communication_functions(mc.conf)
    s
end

function global_update(s::AdaptiveScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    
    if s.sequence[s.idx] === Adaptive()
        # Find appropriate (i.e. with probability matching sampling rates) 
        # update from adaptive pool
        total_weight = sum(s.sampling_rates)
        target = rand() * total_weight
        current_weight = 0.0
        idx = 1
        while idx < length(s.sampling_rates)
            current_weight += s.sampling_rates[idx]
            if target <= current_weight
                break
            else
                idx += 1
            end
        end

        # Apply the update and adjust sampling rate
        update = s.adaptive_pool[idx]
        accepted = global_update(update, mc, model, s.conf, s.temp)
        if !(update isa AcceptanceStatistics{NoUpdate}) && update.total > s.grace_period
            s.sampling_rates[idx] = (
                s.adaptive_rate * s.sampling_rates[idx] + 
                update.accepted / update.total
            ) / (s.adaptive_rate + 1)
            
            # Hit miniomum threshold - this can no longer be accepted.
            if s.sampling_rates[idx] < s.minimum_sampling_rate
                s.sampling_rates[idx] = 0.0
            end
        end

        return accepted
    else
        # Some sort of non-adaptive update, just perform it.
        return global_update(s.sequence[s.idx], mc, model, s.conf, s.temp)
    end
end

function show_statistics(s::AdaptiveScheduler, prefix="")
    println(prefix, "Global update statistics (since start):")
    cum_accepted = 0
    cum_total = 0
    for update in s.sequence
        # here NoUpdate()'s are static, so they represent "skip global here"
        # Adaptive()'s are give by updates from adaptive pool
        update isa AcceptanceStatistics{NoUpdate} && continue
        update isa Adaptive && continue
        @printf(
            "%s\t%s %2.1f%s accepted (%i / %i)\n",
            prefix, rpad(name(update), 20), 100update.accepted/max(1, update.total), 
            "%", update.accepted, update.total
        )
        cum_accepted += update.accepted
        cum_total += update.total
    end
    for update in s.adaptive_pool
        # here NoUpdate()'s may replace a global update, so they kinda count
        @printf(
            "%s\t%s %2.1f%s accepted (%i / %i)\n",
            prefix, rpad(name(update), 20), 100update.accepted/max(1, update.total), 
            "%", update.accepted, update.total
        )
        cum_accepted += update.accepted
        cum_total += update.total
    end
    @printf(
        "%s\t%s %2.1f%s accepted (%i / %i)\n",
        prefix, rpad("Total", 20), 100cum_accepted/cum_total, 
        "%", cum_accepted, cum_total
    )
    nothing
end

function Base.show(io::IO, s::AdaptiveScheduler)
    sequence = mapreduce(
        i -> name(s.sequence[mod1(i, length(s.sequence))]),
        (a, b) -> "$a -> $b",
        s.idx+1 : s.idx+length(s.sequence)
    )
    total_weight = sum(s.sampling_rates)
    pool = mapreduce((a, b) -> "$a, $b", s.sampling_rates, s.adaptive_pool) do sr, u
        @sprintf("%0.0f%s %s", 100sr / total_weight, "%", name(u))
    end
    println(io, "AdaptiveScheduler():")
    println(io, "\t$sequence -> (repeat)")
    print(io, "\twith Adaptive() = ($pool)")
end

function save_scheduler(file::JLDFile, s::AdaptiveScheduler, entryname::String="/Scheduler")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(s))
    write(file, entryname * "/sequence", s.sequence)
    write(file, entryname * "/sampling_rates", s.sampling_rates)
    write(file, entryname * "/pool", s.adaptive_pool)
    write(file, entryname * "/minimum_sampling_rate", s.minimum_sampling_rate)
    write(file, entryname * "/grace_period", s.grace_period)
    write(file, entryname * "/adaptive_rate", s.adaptive_rate)
    write(file, entryname * "/idx", s.idx)
    nothing
end
function _load(data, ::Type{<: AdaptiveScheduler}, ::Type{<: DQMC}, model)
    s = AdaptiveScheduler(DQMC, model, data["sequence"], data["pool"])
    s.sampling_rates = data["sampling_rates"]
    s.minimum_sampling_rate = data["minimum_sampling_rate"]
    s.grace_period = data["grace_period"]
    s.adaptive_rate = data["adaptive_rate"]
    s.idx = data["idx"]
    s
end




################################################################################
### Updates
################################################################################



# Should this inherit from AbstractGlobalUpdate?
mutable struct AcceptanceStatistics{Update <: AbstractGlobalUpdate}
    accepted::Int
    total::Int
    update::Update
end
AcceptanceStatistics(update) = AcceptanceStatistics(0, 0, update)
AcceptanceStatistics(wrapped::AcceptanceStatistics) = wrapped
AcceptanceStatistics(proxy::Adaptive) = proxy
name(w::AcceptanceStatistics) = name(w.update)
function global_update(w::AcceptanceStatistics, mc, m, tc, tv)
    accepted = global_update(w.update, mc, m, tc, tv)
    w.total += 1
    w.accepted += accepted
    return accepted
end
function Base.show(io::IO, u::AcceptanceStatistics)
    @printf(io,
        "%s with %i/%i = %0.1f%c accepted", 
        name(u), u.accepted, u.total, u.accepted/max(1, u.accepted), '%'
    )
end



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

function global_update(u::GlobalFlip, mc, model, temp_conf, temp_vec)
    c = conf(mc)
    @. temp_conf = -c
    return global_update(mc, model, temp_conf, temp_vec)
end



"""
    GlobalShuffle([mc, model, [sampling_rate = 0.5])

A global update that shuffles the current configuration. Note that this is not 
local to a time slice.
"""
struct GlobalShuffle <: AbstractGlobalUpdate end
GlobalShuffle(mc, model) = GlobalShuffle()
name(::GlobalShuffle) = "GlobalShuffle"


function global_update(u::GlobalShuffle, mc, model, temp_conf, temp_vec)
    copyto!(temp_conf, conf(mc))
    shuffle!(temp_conf)
    return global_update(mc, model, temp_conf, temp_vec)
end



abstract type AbstractParallelUpdate <: AbstractGlobalUpdate end


# I.e. we >trade< or not
# struct ReplicaExchange <: AbstractGlobalUpdate end 
# TODO: It would be better to use the temp_conf in the scheduler...
struct ReplicaExchange <: AbstractParallelUpdate
    target::Int
end
ReplicaExchange(mc, model, target) = ReplicaExchange(target)
name(::ReplicaExchange) = "ReplicaExchange"

@bm function global_update(u::ReplicaExchange, mc, model, temp_conf, temp_vec)
    # Need to sync at the start here because else th weights might be based on different confs
    # barrier
    wait_for_remote(u.target)

    # swap conf
    conf = pull_conf_from_remote(u.target)
    temp_conf .= conf

    # compute weight
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, temp_conf, temp_vec)
    local_weight = exp(- ΔE_boson) * detratio
    local_prob = rand()
    
    # swap weight, probability
    remotecall(put_weight_prob!, u.target, myid(), local_weight, local_prob)
    while !isready(weight_probability)
        yield()
    end
    remote, remote_weight, remote_prob = take!(weight_probability)
    @assert remote == u.target

    # accept/deny
    # We need to pick one of the random values and do so consistently across 
    # processes. 
    w = local_weight * remote_weight
    if ifelse(myid() < u.target, local_prob, remote_prob) < w
        accept_global!(mc, model, temp_conf, passthrough)
        return 1
    end

    return 0
end



# I.e. we gift/push confs to each other and accept them independently.
# struct ReplicaPush <: AbstractGlobalUpdate
mutable struct ReplicaPull <: AbstractParallelUpdate
    cycle_idx::Int
end
ReplicaPull() = ReplicaPull(1)
ReplicaPull(mc::MonteCarloFlavor, model::Model) = ReplicaPull(1)
name(::ReplicaPull) = "ReplicaPull"

@bm function global_update(u::ReplicaPull, mc, model, temp_conf, temp_vec)
    # cycle first to make sure the idx is in bounds
    @sync if !isempty(connected_ids)
        idx = mod1(u.cycle_idx, length(connected_ids))
        conf = pull_conf_from_remote(connected_ids[idx])
        temp_conf .= conf
    end
    return global_update(mc, model, temp_conf, temp_vec)
end



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



@bm function propose_global_from_conf(mc::DQMC, m::Model, conf::AbstractArray, temp)
    # I don't think we need this...
    @assert mc.stack.current_slice == 1
    @assert mc.stack.direction == 1

    # This should be just after calculating greens, so mc.stack.Dl is from the UDT
    # decomposed G
    # We need an independent temp vector here as inv_det changes Dl, Dr and tempv
    temp .= mc.stack.Dl

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
        detratio *= temp[i] * mc.stack.Dr[i]
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


# TODO remove
@bm function global_move(mc, model)
    conf = shuffle(mc.conf)
    detratio, ΔE_boson, new_greens = propose_global_from_conf(mc, model, conf, similar(mc.stack.Dr))

    if mc.parameters.check_sign_problem
        if abs(imag(detratio)) > 1e-6
            push!(mc.analysis.imaginary_probability, abs(imag(detratio)))
            mc.parameters.silent || @printf(
                "Did you expect a sign problem? imag. detratio:  %.9e\n", 
                abs(imag(detratio))
            )
        end
        if real(detratio) < 0.0
            push!(mc.analysis.negative_probability, real(detratio))
            mc.parameters.silent || @printf(
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
    replica_exchange.greens = copy(dqmc.stack.greens)
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
        send!(mc.conf, mc.stack.greens)
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
    copyto!(mc.stack.u_stack[1], I)
    mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
    copyto!(mc.stack.t_stack[1], I)

    @inbounds for idx in 1:length(mc.stack.ranges)
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.stack.ranges[idx])
        for slice in mc.stack.ranges[idx]
            multiply_slice_matrix_inv_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx]))
        udt_AVX_pivot!(
            mc.stack.u_stack[idx + 1], mc.stack.d_stack[idx + 1], mc.stack.tmp1, 
            mc.stack.pivot, mc.stack.tempv
        )
        vmul!(mc.stack.t_stack[idx + 1], mc.stack.tmp1, mc.stack.t_stack[idx])
    end
    U = copy(mc.stack.u_stack[end])
    D = copy(mc.stack.d_stack[end])
    T = copy(mc.stack.t_stack[end])

    # Ul Dl Tl is the same
    Ul = copy(mc.stack.u_stack[end])
    Dl = copy(mc.stack.d_stack[end])
    Tl = copy(mc.stack.t_stack[end])

    # Ur Dr Tr from new greens
    mc.conf .= conf
    copyto!(mc.stack.u_stack[1], I)
    mc.stack.d_stack[1] .= one(eltype(mc.stack.d_stack[1]))
    copyto!(mc.stack.t_stack[1], I)

    @inbounds for idx in 1:length(mc.stack.ranges)
        copyto!(mc.stack.curr_U, mc.stack.u_stack[idx])

        # println("Adding slice seq left $idx = ", mc.stack.ranges[idx])
        for slice in mc.stack.ranges[idx]
            multiply_daggered_slice_matrix_left!(mc, mc.model, slice, mc.stack.curr_U)
        end

        vmul!(mc.stack.tmp1, mc.stack.curr_U, Diagonal(mc.stack.d_stack[idx]))
        udt_AVX_pivot!(
            mc.stack.u_stack[idx + 1], mc.stack.d_stack[idx + 1], mc.stack.tmp1, 
            mc.stack.pivot, mc.stack.tempv
        )
        vmul!(mc.stack.t_stack[idx + 1], mc.stack.tmp1, mc.stack.t_stack[idx])
    end
    Ur = copy(mc.stack.u_stack[end])
    Dr = copy(mc.stack.d_stack[end])
    Tr = copy(mc.stack.t_stack[end])

    # new G
    build_stack(mc, mc.stack)
    propagate(mc)

    mc.stack.greens + compute_ratio(U, D, T, Ul, Dl, Tl, Ur, Dr, Tr)
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