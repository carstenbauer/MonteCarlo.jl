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



################################################################################
### Updates
################################################################################



abstract type AbstractParallelUpdate <: AbstractGlobalUpdate end


# I.e. we >trade< or not
struct ReplicaExchange <: AbstractParallelUpdate
    target::Int
end
ReplicaExchange(mc, model, target) = ReplicaExchange(target)
name(::ReplicaExchange) = "ReplicaExchange"

@bm function update(u::ReplicaExchange, mc, model)
    # Need to sync at the start here because else th weights might be based on different confs
    # barrier
    wait_for_remote(u.target)

    # swap conf
    conf = pull_conf_from_remote(u.target)
    mc.temp_conf .= conf

    # compute weight
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, mc.temp_conf)
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
        accept_global!(mc, model, mc.temp_conf, passthrough)
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

@bm function update(u::ReplicaPull, mc, model)
    # cycle first to make sure the idx is in bounds
    @sync if !isempty(connected_ids)
        idx = mod1(u.cycle_idx, length(connected_ids))
        conf = pull_conf_from_remote(connected_ids[idx])
        mc.temp_conf .= conf
    end
    return global_update(mc, model, mc.temp_conf)
end