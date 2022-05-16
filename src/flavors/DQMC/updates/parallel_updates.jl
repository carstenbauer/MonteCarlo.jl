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
disconnect(targets...; wait = false) = disconnect(collect(targets), wait = wait)
function disconnect(targets; wait=false)
    if wait
        @sync for target in targets
            @async remotecall_wait(remove_worker!, target, myid())
        end
    else
        for target in targets
            remotecall(remove_worker!, target, myid())
        end
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
const barrier_counter = Ref(1)
const waiting_list = Set{Tuple{Int, Int}}()

function wait_for(id, counter)
    push!(waiting_list, (id, counter))
    filter!(x -> x[2] >= barrier_counter[], waiting_list)
    nothing
end

function wait_for_remote(ids::Int...; timeout=Inf)
    t0 = time()
    for trg in ids
        remotecall(wait_for, trg, myid(), barrier_counter[])
    end
    
    while !all(trg -> (trg, barrier_counter[]) in waiting_list, ids)
        yield()
        if time() - t0 > timeout
            barrier_counter[] += 1
            return false
        end
    end

    barrier_counter[] += 1
    return true
end

function generate_communication_functions(local_conf)
    @eval MonteCarlo begin
        get_conf() = $local_conf
    end
end



################################################################################
### Updates
################################################################################



abstract type AbstractParallelUpdate <: AbstractGlobalUpdate end
init!(mc, ::AbstractParallelUpdate) = generate_communication_functions(conf(field(mc)))

"""
    ReplicaExchange([mc, model], target[, timeout = 600.0])

Represents a replica exchange update with a given `target` worker.

Note that this update is blocking, i.e. a simulation will wait on its partner to
perform the update. You can define a `timeout` in seconds to avoid the simulation
getting stuck for too long.
"""
struct ReplicaExchange <: AbstractParallelUpdate
    target::Int
    timeout::Float64
end
ReplicaExchange(target) = ReplicaExchange(target, 600.0)
ReplicaExchange(mc, model, target, timeout=600.0) = ReplicaExchange(target, timeout)
name(::ReplicaExchange) = "ReplicaExchange"
function _save(f::FileLike, name::String, update::ReplicaExchange)
    write(f, "$name/tag", :ReplicaExchange)
    write(f, "$name/target", update.target)
    write(f, "$name/timeout", update.timeout)
end
_load(f::FileLike, ::Val{:ReplicaExchange}) = ReplicaExchange(f["target"], f["timeout"])

@bm function update(u::ReplicaExchange, mc, model, field)
    tc = temp_conf(field)
    # Need to sync at the start here because else th weights might be based on different confs
    # barrier
    if !wait_for_remote(u.target, timeout = u.timeout)
        return 0
    end

    # swap conf
    conf = pull_conf_from_remote(u.target)
    tc .= conf

    # compute weight
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, tc)
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
        accept_global!(mc, model, tc, passthrough)
        return 1
    end

    return 0
end



"""
    ReplicaPull([mc, model])

This update will pull a configuration from a connected worker and attempt a 
global update with it. The target worker will be cycled as the simulation 
progresses.

To connect workers for this update, the remote worker has to call 
`connect(this_worker)`.

This update only blocks locally.
"""
mutable struct ReplicaPull <: AbstractParallelUpdate
    cycle_idx::Int
end
ReplicaPull() = ReplicaPull(1)
ReplicaPull(mc::MonteCarloFlavor, model::Model) = ReplicaPull(1)
name(::ReplicaPull) = "ReplicaPull"
Base.:(==)(a::ReplicaPull, b::ReplicaPull) = a.cycle_idx == b.cycle_idx
function _save(f::FileLike, name::String, update::ReplicaPull)
    write(f, "$name/tag", :ReplicaPull)
    write(f, "$name/idx", update.cycle_idx)
end
_load(f::FileLike, ::Val{:ReplicaPull}) = ReplicaPull(f["idx"])

@bm function update(u::ReplicaPull, mc, model, field)
    tc = temp_conf(field)
    # cycle first to make sure the idx is in bounds
    @sync if !isempty(connected_ids)
        idx = mod1(u.cycle_idx, length(connected_ids))
        conf = pull_conf_from_remote(connected_ids[idx])
        tc .= conf
        return global_update(mc, model, tc)
    end
    return 0
end