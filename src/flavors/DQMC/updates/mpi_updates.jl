################################################################################
### Updates
################################################################################
import .MPI


abstract type AbstractMPIUpdate <: AbstractGlobalUpdate end
function init!(mc, ::AbstractMPIUpdate)
    MPI.Initialized() || MPI.Init()
end


"""
    MPIReplicaExchange([mc, model], target)

Represents a replica exchange update with a given `target` MPI rank.

Note that this update is blocking, i.e. a simulation will wait on its partner to
perform the update.
"""
struct MPIReplicaExchange <: AbstractMPIUpdate
    target::Int
end
MPIReplicaExchange(target) = MPIReplicaExchange(target)
MPIReplicaExchange(mc, model, target) = MPIReplicaExchange(target)
name(::MPIReplicaExchange) = "MPIReplicaExchange"
function _save(f::FileLike, name::String, update::MPIReplicaExchange)
    write(f, "$name/tag", :MPIReplicaExchange)
    write(f, "$name/target", update.target)
end
_load(f::FileLike, ::Val{:MPIReplicaExchange}) = MPIReplicaExchange(f["target"])

@bm function update(u::MPIReplicaExchange, mc, model, field)
    c = conf(field); tc = temp_conf(field)
    # swap conf
    MPI.Isend(c,  u.target, 0, MPI.COMM_WORLD)
    MPI.Recv!(tc, u.target, 0, MPI.COMM_WORLD)

    # compute weight
    detratio, ΔE_boson, passthrough = propose_global_from_conf(mc, model, tc)
    local_weight = exp(- ΔE_boson) * detratio
    local_prob = rand()
    
    # swap weight, probability
    req = MPI.Isend((local_weight, local_prob), u.target, 1, MPI.COMM_WORLD)
    data, _ = MPI.Recv(Tuple{Float64, Float64}, u.target, 1, MPI.COMM_WORLD)
    remote_weight, remote_prob = data

    # accept/deny
    # We need to pick one of the random values and do so consistently across 
    # processes. 
    w = local_weight * remote_weight
    if ifelse(myid() < u.target, local_prob, remote_prob) < w
        accept_global!(mc, model, tc, passthrough)
        MPI.Wait!(req)
        return 1
    else
        MPI.Wait!(req)
        return 0
    end
end



# """
#     MPIReplicaPull([mc, model])

# This update will pull a configuration from a connected worker and attempt a 
# global update with it. The target worker will be cycled as the simulation 
# progresses.

# To connect workers for this update, the remote worker has to call 
# `connect(this_worker)`.

# This update only blocks locally.
# """
# mutable struct MPIReplicaPull <: AbstractMPIUpdate
#     cycle_idx::Int
#     window::MPI.Win

#     MPIReplicaPull(idx) = new(idx)
# end
# MPIReplicaPull() = MPIReplicaPull(1)
# MPIReplicaPull(mc::MonteCarloFlavor, model::Model) = MPIReplicaPull(1)
# name(::MPIReplicaPull) = "MPIReplicaPull"
# Base.:(==)(a::MPIReplicaPull, b::MPIReplicaPull) = a.cycle_idx == b.cycle_idx
# function init!(mc, update::MPIReplicaPull)
#     MPI.Initialized() || MPI.Init()
#     update.window = MPI.Win_create(mc.conf, MPI.COMM_WORLD)
#     nothing
# end

# @bm function update(u::MPIReplicaPull, mc, model)
#     # TODO
#     # - this needs to call
#     #   MPI.Win_create(mc.conf, MPI.COMM_WORLD)
#     #   and make the result available here?
#     # - needs testing
#     # - needs its own connected_ids management

#     # cycle first to make sure the idx is in bounds
#     @sync if !isempty(connected_ids)
#         idx = mod1(u.cycle_idx, length(connected_ids))
#         MPI.Get(mc.temp_conf, connected_ids[idx], u.window)
#         return global_update(mc, model, mc.temp_conf)
#     end
#     return 0
# end