# This file should probably be moved to the general directory and used by 
# classical MC as well



################################################################################
### Interface Utility updates
################################################################################



"""
    AbstractUpdate

An update should be a struct inhereting from AbstractUpdate or one of its 
abstract children `AbstractLocalUpdate`, `AbstractGlobalUpdate` or 
`AbstractParallelUpdate`. 

```
struct MyGlobalUpdate <: MonteCarlo.AbstractGlobalUpdate
    ...
end
```

It must implement: 
- `update(update, mc, model, field_or_conf)` to perform the update
- `_load(f::FileLike, ::Val{name})` to load the update from a file

It can implement
- `init!(mc, update)` to perform simulation specific initialization
- `is_full_sweep(update) = false` to disable sweep incrementation
- `requires_temp_conf(update) = true` to request allocation of a temporery configuration
- `can_replace(update1, update2)` to identify updates that can replace each other
- `_save(f::FileLike, name, update)` to provide a custom saving method
- `name(update)` to provide a custom name for printing

See the individual docstrings for more information.

Behind the scenes, the scheduler will wrap updates in `AcceptanceStatistics` 
which collect the number of requested and accepted updates. It is expected that 
you return `0/false` if the update is denied or `1/true` if it is accepted
for this purpose.
"""
abstract type AbstractUpdate end
abstract type AbstractLocalUpdate <: AbstractUpdate end


########################################
### General Interface
########################################


"""
    init!(mc, update::AbstractUpdate)

Initializes an update with an otherwise initialized simulation `mc`. This is 
called just before the main Simulation loop.
"""
init!(mc, update::AbstractUpdate) = nothing

"""
    is_full_sweep(update::AbstractUpdate)

If this returns `true` for a given `update` the sweep counter is incremented 
after executing it. Defaults to `true`.
"""
is_full_sweep(update::AbstractUpdate) = true

"""
    requires_temp_conf(update::AbstractUpdate)

If this returns `true` for a given `update` the simulation will allocate a 
second configuration as a temporary buffer for the update. This configuration
is reused between all updates and should be asumed to change between calls to 
`update!(...)`
"""
requires_temp_conf(update::AbstractUpdate) = false
# If this returns true updates with the same key will be combined into a single instance

"""
    can_replace(update1, update2)

If this returns `true` the scheduler is allowed to replace one of the updates 
by the other. Defaults to false.

This can be useful to save memory when an update needs some temporary storage 
that can be shared between different instances.
"""
can_replace(u1::AbstractUpdate, u2::AbstractUpdate) = false

"""
    _save(file::FileLike, name::String, update::AbstractUpdate)

Saves an update to a file. By default, writes only `nameof(typeof(update))`.
"""
function _save(file::FileLike, name::String, update::AbstractUpdate)
    write(file, "$name/tag", nameof(typeof(update)))
end

"""
    update(update::AbstractUpdate, mc, model, field_or_conf)

Performs the `update` with the given simulation `mc`, `model` and 
`field_or_conf`. Should return `true` if the update has been successfull.

This function must to be implemented for each update.
"""
function update(update::AbstractUpdate, args...)
    throw(MethodError(update, (update, args...)))
end

"""
    name(update::AbstractUpdate)

Returns a string corresponding to the name of the update. Defaults to 
`string(nameof(typeof(update)))`. Used in printing.
"""
name(update::AbstractUpdate) = update |> typeof |> nameof |> string



"""
AbstractUtilityUpdate <: AbstractUpdate

An `AbstractUtilityUpdate` is one that doesn't update the configuration. It may
be a placeholder, like `NoUpdate` and `Adaptive`, or modify the simulation in 
some other way like `ChemicalPotentialTuning`
"""
abstract type AbstractUtilityUpdate <: AbstractUpdate end

"""
    NoUpdate([mc, model])

An update that does nothing. Used internally to keep the adaptive scheduler 
running if all (other) adaptive updates are discarded.
"""
struct NoUpdate <: AbstractUtilityUpdate end
NoUpdate(mc, model) = NoUpdate()
function update(u::NoUpdate, args...)
    # we count this as "denied" global update
    return 0
end
is_full_sweep(update::NoUpdate) = false
_load(f::FileLike, ::Val{:NoUpdate}) = NoUpdate()




"""
    Adaptive()

A placeholder for adaptive updates in the `AdaptiveScheduler`.
"""
struct Adaptive <: AbstractUtilityUpdate end
name(::Adaptive) = "Adaptive"
_load(f::FileLike, ::Val{:Adaptive}) = Adaptive()




# Should this inherit from AbstractUpdate?
# This is required for AdaptiveScheduler, used by both
mutable struct AcceptanceStatistics{Update <: AbstractUpdate} <: AbstractUpdate
    accepted::Float64
    total::Int
    update::Update
end
AcceptanceStatistics(update) = AcceptanceStatistics(0.0, 0, update)
AcceptanceStatistics(wrapped::AcceptanceStatistics) = wrapped
AcceptanceStatistics(proxy::Adaptive) = proxy
name(w::AcceptanceStatistics) = name(w.update)
requires_temp_conf(update::AcceptanceStatistics) = requires_temp_conf(update.update)
function can_replace(u1::AcceptanceStatistics, u2::AcceptanceStatistics)
    can_replace(u1.update, u2.update) && (u1.accepted == u2.accepted) && (u1.total == u2.total)
end
function update(w::AcceptanceStatistics, mc, m, field)
    accepted = update(w.update, mc, m, field)
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
init!(mc, update::AcceptanceStatistics) = init!(mc, update.update)
is_full_sweep(update::AcceptanceStatistics) = is_full_sweep(update.update)
function _save(file::FileLike, name::String, update::AcceptanceStatistics)
    write(file, "$name/tag", nameof(typeof(update)))
    write(file, "$name/accepted", update.accepted)
    write(file, "$name/total", update.total)
    _save(file, "$name/update", update.update)
    return
end
function _load(f::FileLike, ::Val{:AcceptanceStatistics})
    AcceptanceStatistics(f["accepted"], f["total"], _load(f["update"]))
end



################################################################################
### Generic
################################################################################



updates(s::AbstractUpdateScheduler) = s.sequence
requires_temp_conf(s::AbstractUpdateScheduler) = any(requires_temp_conf, updates(s))
function init_scheduler!(mc, scheduler::AbstractUpdateScheduler)
    for update in updates(scheduler)
        init!(mc, update)
    end
    nothing
end

function make_unique(updates::Union{Tuple, Vector})
    cache = AbstractUpdate[]
    return map(updates) do update
        for cached_update in cache
            can_replace(update, cached_update) && return cached_update
        end
        push!(cache, update)
        return update
    end
end


################################################################################
### SimpleScheduler
################################################################################



"""
    SimpleScheduler(updates...)

Schedules updates in the order specified through `updates...`. Note that local 
updates (`LocalSweep([N=1])`) always needs to be part of those updates, as they
define what a sweep is.

Example:
```
scheduler = SimpleScheduler(
    LocalSweep(10), 
    GlobalFlip(), 
    GlobalShuffle(), 
    LocalSweep(10), 
    ReplicaExchange(3)
)
```
This defines a scheduler that performs 10 local sweeps, then attempts a global 
flip of the configuration, then a global shuffle, another 10 local sweeps and
a replica exchange with worker 3. After the replica exchange the sequence 
repeats from the start. 
"""
mutable struct SimpleScheduler{ST} <: AbstractUpdateScheduler
    sequence::ST
    idx::Int64

    # For loading
    function SimpleScheduler(sequence::ST, idx::Int) where {ST}
        new{ST}(sequence, idx)
    end

    function SimpleScheduler(updates...)
        # Without local sweeps we don't have a sweep increment
        updates = make_unique(AcceptanceStatistics.(Tuple(vcat(updates...))))
        if !any(u -> u isa AcceptanceStatistics{<: AbstractLocalUpdate}, updates)
            error("The scheduler requires local updates, but none were passed.")
        end
        any(isa.(updates, Adaptive)) && error("SimpleScheduler cannot process Adaptive. Use AdaptiveScheduler.")
        obj = new{typeof(updates)}()
        obj.sequence = updates
        obj.idx = 0
        obj
    end
end

function update(s::SimpleScheduler, mc::DQMC, model, field = field(mc))
    while true
        s.idx = mod1(s.idx + 1, length(s.sequence))
        update(s.sequence[s.idx], mc, model, field)
        is_full_sweep(s.sequence[s.idx]) && break
    end
    mc.last_sweep += 1
    return nothing
end

function max_acceptance(s::SimpleScheduler)
    output = 0.0
    for update in s.sequence
        if update isa AcceptanceStatistics
            output = max(output, update.accepted / max(1, update.total))
        end
    end
    output
end

function show_statistics(io::IO, s::SimpleScheduler, prefix="")
    println(io, prefix, "Update statistics (since start):")

    cum_accepted  = mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.sequence)
    cum_total  = mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.sequence)

    accumulated = flatten_sequence_statistics(s.sequence)
    show_accumulated_sequence(io, accumulated, prefix * "\t", cum_total)
        
    p = @sprintf("%2.1f", 100cum_accepted/max(1, cum_total))
    s = rpad("Total", 20) * " " * lpad(p, 5) * "% accepted" *
        "   (" * string(round(Int, cum_accepted)) * " / $cum_total)"

    println(io, prefix, '\t', '-' ^ length(s))
    println(io, "$prefix\t", s)

    nothing
end

function Base.show(io::IO, s::SimpleScheduler)
    sequence = combine_sequence_to_string(s.sequence)
    print(io, "SimpleScheduler(): $sequence -> (repeat)")
end

function _save(file::FileLike, entryname::String, s::SimpleScheduler)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "SimpleScheduler")
    _save_collection(file, "$entryname/sequence", s.sequence)
    write(file, entryname * "/idx", s.idx)
    nothing
end
function _load(data, ::Val{:SimpleScheduler})
    sequence = make_unique(_load_collection(data["sequence"]))
    SimpleScheduler(Tuple(sequence), data["idx"])
end

to_tag(::Type{<: SimpleScheduler}) = Val(:SimpleScheduler)



################################################################################
### AdaptiveScheduler
################################################################################



"""
    AdaptiveScheduler(sequence::Tuple, pool::Tuple; kwargs...])

Creates an adaptive scheduler which prioritizes updates that succeed more 
frequently. 

`sequence` defines the static sequence of global updates, with 
`Adaptive()` as a proxy for adaptive updates. When polling the scheduler 
`Adaptive()` will be replaced by one of the updates in `pool` based on their 
success rate.

Example:
```
scheduler = AdaptiveScheduler(
    (Adaptive(), LocalSweep(10), ReplicaExchange(3)), 
    (GlobalFlip(), GlobalShuffle())
)
```
This setup will perform one of `GlobalFlip()` and `GlobalShuffle()` followed by 
10 local sweeps, a `ReplicaExchange(3)` update, repeating after that. The 
replica exchange will always be a `ReplicaExchange()` while the ealier global
update will be picked based on their relative sampling rates which are sluggishly 
derived from acceptance rates. 

If those sampling rates fall below a certain threshhold they may be set to 0. 
If all updates drop to 0 sampling rate, `NoUpdate()` is performed instead. 
(`NoUpdate()` is always added and has a static probability of `1e-10`.)

### Keyword Arguments:
- `minimum_sampling_rate = 0.01`: This defines the threshold under which the 
sampling rate is set to 0.
- `grace_period = 99`: This sets a minimum number of times an update needs to 
be called before its sampling rate is adjusted. 
- `adaptive_rate = 9.0`: Controls how fast the sampling rate is adjusted to the 
acceptance rate. More means slower. This follows the formula 
`(adaptive_rate * sampling_rate + accepted/total) / (adaptive_rate + 1)`
"""
mutable struct AdaptiveScheduler{PT, ST} <: AbstractUpdateScheduler
    minimum_sampling_rate::Float64
    grace_period::Int
    adaptive_rate::Float64

    adaptive_pool::PT
    sampling_rates::Vector{Float64}
    sequence::ST
    idx::Int

    # for loading
    function AdaptiveScheduler(msr, gp, ar, ap::PT, sr, s::ST, i) where {PT, ST}
        new{PT, ST}(msr, gp, ar, ap, sr, s, i)
    end

    function AdaptiveScheduler(sequence, pool;
            minimum_sampling_rate = 0.01, grace_period = 99, adaptive_rate = 9.0
        )
        # Without local updates we have no sweep increment
        sequence = make_unique(AcceptanceStatistics.(vcat(sequence...)))
        if !any(u -> u isa AcceptanceStatistics{<: AbstractLocalUpdate}, sequence)
            error("The scheduler requires local updates, but none were passed.")
        end

        # sampling rate x after i steps with constant acceptance rate p:
        # x_i = (c/c+1)^i x_0 + a * \sum_{i} c^{i-1} / (c+1)^i
        # where c is the adaptive rate and x_0 is the initial sampling rate.
        # Minimum number of samples to discard:
        i_min = ceil(Int, log(
            adaptive_rate / (adaptive_rate+1), 
            minimum_sampling_rate / 0.5
        ))
        @debug("Minimum number of samples for discard: $grace_period + $i_min")

        if !any(x -> x isa NoUpdate, pool)
            pool = tuple(vcat(pool...)..., NoUpdate())
        end

        # Wrap in AcceptanceStatistics
        adaptive_pool = make_unique(AcceptanceStatistics.(pool))

        adaptive_pool = Tuple(adaptive_pool)
        sequence = Tuple(sequence)

        obj = new{typeof(adaptive_pool), typeof(sequence)}()
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

function update(s::AdaptiveScheduler, mc::DQMC, model, field = field(mc))
    while true
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
            updater = s.adaptive_pool[idx]
            update(updater, mc, model, field)
            if !(updater isa AcceptanceStatistics{NoUpdate}) && updater.total > s.grace_period
                s.sampling_rates[idx] = (
                    s.adaptive_rate * s.sampling_rates[idx] + 
                    updater.accepted / updater.total
                ) / (s.adaptive_rate + 1)
                
                # Hit miniomum threshold - this can no longer be accepted.
                if s.sampling_rates[idx] < s.minimum_sampling_rate
                    s.sampling_rates[idx] = 0.0
                end
            end

            
        else
            # Some sort of non-adaptive update, just perform it.
            updater = s.sequence[s.idx]
            update(updater, mc, model, field)
        end

        is_full_sweep(updater) && break
    end

    mc.last_sweep += 1

    return nothing
end

function max_acceptance(s::AdaptiveScheduler)
    output = 0.0
    for update in s.sequence
        if update isa AcceptanceStatistics
            output = max(output, update.accepted / max(1, update.total))
        end
    end
    for update in s.adaptive_pool
        if update isa AcceptanceStatistics
            output = max(output, update.accepted / max(1, update.total))
        end
    end
    output
end

function show_statistics(io::IO, s::AdaptiveScheduler, prefix="")
    println(io, prefix, "Update statistics (since start):")

    cum_accepted  = mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.sequence)
    cum_accepted += mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.adaptive_pool)
    cum_total  = mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.sequence)
    cum_total += mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.adaptive_pool)

    accumulated = flatten_sequence_statistics(s.adaptive_pool)
    flatten_sequence_statistics(s.sequence, accumulated)
    show_accumulated_sequence(io, accumulated, prefix * "\t", cum_total)

    p = @sprintf("%2.1f", 100cum_accepted/max(1, cum_total))
    s = rpad("Total", 20) * " " * lpad(p, 5) * "% accepted" *
        "   (" * string(round(Int, cum_accepted)) * " / $cum_total)"

    println(io, prefix, '\t', '-' ^ length(s))
    println(io, "$prefix\t", s)

    nothing
end

function Base.show(io::IO, s::AdaptiveScheduler)
    sequence = combine_sequence_to_string(s.sequence)
    total_weight = sum(s.sampling_rates)
    pool = mapreduce((a, b) -> "$a, $b", s.sampling_rates, s.adaptive_pool) do sr, u
        @sprintf("%0.0f%s %s", 100sr / total_weight, "%", name(u))
    end
    println(io, "AdaptiveScheduler():")
    println(io, "\t$sequence -> (repeat)")
    print(io, "\twith Adaptive() = ($pool)")
end

function _save(file::FileLike, entryname::String, s::AdaptiveScheduler)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/tag", "AdaptiveScheduler")
    _save_collection(file, "$entryname/sequence", s.sequence)
    write(file, entryname * "/sampling_rates", s.sampling_rates)
    _save_collection(file, "entryname/pool", s.adaptive_pool)
    write(file, entryname * "/minimum_sampling_rate", s.minimum_sampling_rate)
    write(file, entryname * "/grace_period", s.grace_period)
    write(file, entryname * "/adaptive_rate", s.adaptive_rate)
    write(file, entryname * "/idx", s.idx)
    nothing
end
function _load(data, ::Val{:AdaptiveScheduler})
    sequence = make_unique(_load_collection(data["sequence"]))
    pool = make_unique(_load_collection(data["pool"]))
    s = AdaptiveScheduler(Tuple(sequence), Tuple(pool))
    s.sampling_rates = data["sampling_rates"]
    s.minimum_sampling_rate = data["minimum_sampling_rate"]
    s.grace_period = data["grace_period"]
    s.adaptive_rate = data["adaptive_rate"]
    s.idx = data["idx"]
    s
end

to_tag(::Type{<: AdaptiveScheduler}) = Val(:AdaptiveScheduler)


updates(s::AdaptiveScheduler) = (s.sequence..., s.adaptive_pool...)



################################################################################
### Printing utillities
################################################################################



function flatten_sequence_statistics(sequence, accumulated::Dict = Dict{String, Tuple{Float64, Int}}())
    for update in sequence
        if update isa Adaptive || !(update isa AcceptanceStatistics)
            continue
        end

        key = name(update)
        acc = update.accepted
        total = update.total

        if haskey(accumulated, key)
            accumulated[key] = accumulated[key] .+ (acc, total)
        else
            accumulated[key] = (acc, total)
        end
    end

    accumulated
end

function show_accumulated_sequence(io::IO, accumulated, prefix = "", max_total=0)
    N = length(string(max_total))
    sorted = sort!(collect(accumulated), by = x -> x[2][1] / max(1, x[2][2]))
    for (key, (acc, total)) in sorted
        total == 0 && continue
        p = @sprintf("%2.1f", 100acc / max(1, total))
        println(
            io, prefix, rpad(key, 20), " ", lpad(p, 5), "% accepted",  "   (", 
            lpad(string(round(Int, acc)), N), " / ", lpad(string(total), N), ")"
        )
    end

    nothing
end

function show_sequence(io::IO, sequence, prefix = "", max_total=0)
    N = length(string(max_total))
    for update in sequence
        if update isa AcceptanceStatistics{NoUpdate} ||
            update isa Adaptive || !(update isa AcceptanceStatistics)
            continue
        end

        p = @sprintf("%2.1f", 100update.accepted/max(1, update.total))
        println(
            io, prefix, rpad(name(update), 20), " ", lpad(p, 5), "% accepted", "(", 
            lpad(string(round(Int, update.accepted)), N), " / ", 
            lpad(string(update.total), N), ")"
        )
    end

    nothing
end

function combine_sequence_to_string(sequence)
    last_update = first(sequence)
    counter = 1
    output = ""
    for update in sequence[2:end]
        if update == last_update
            counter += 1
        else
            _name = name(last_update)
            if counter == 1
                output *= " -> $_name"
            else
                output *= " -> $(_name)($counter)"
            end
            counter = 1
            last_update = update
        end
    end

    _name = name(last(sequence))
    if counter == 1
        output *= " -> $_name"
    else
        output *= " -> $(_name)($counter)"
    end

    output
end

# Literally just for tests
for T in (SimpleScheduler, AdaptiveScheduler, AcceptanceStatistics)
    @eval begin
        function Base.:(==)(a::$T, b::$T)
            for field in fieldnames($T)
                getfield(a, field) == getfield(b, field) || return false
            end
            return true
        end
    end
end