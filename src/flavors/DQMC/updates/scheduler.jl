# This file should probably be moved to the general directory and used by 
# classical MC as well


################################################################################
### EmptyScheduler
################################################################################



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



################################################################################
### SimpleScheduler
################################################################################



"""
    SimpleScheduler(::Type{<: DQMC}, model, updates...)

Schedules global updates in the order specified through `updates...`. For example
`SimpleScheduler(mc, GlobalFlip(), GlobalFlip(), GlobalShuffle())` would generate
a sequence `GlobalFlip() > GlobalFlip() > GlobalShuffle() > repeat`.
"""
mutable struct SimpleScheduler{CT, T <: Tuple} <: AbstractUpdateScheduler
    # temporary
    conf::CT
    sequence::T
    idx::Int64

    # For loading
    function SimpleScheduler(conf::T1, temp, sequence::T2, idx) where {T1, T2}
        new{T1, T2}(conf, temp, sequence, idx)
    end

    function SimpleScheduler(::Type{<: DQMC}, model::Model, updates...)
        dummy = rand(DQMC, model, 1)
        updates = map(AcceptanceStatistics, updates)
        obj = new{typeof(dummy), typeof(updates)}()
        obj.sequence = updates
        obj.idx = 0
        obj
    end
end


function init!(s::SimpleScheduler, mc::DQMC, model::Model)
    s.conf = copy(conf(mc))
    generate_communication_functions(mc.conf)
    s
end

function global_update(s::SimpleScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    global_update(s.sequence[s.idx], mc, model, s.conf)
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



################################################################################
### AdaptiveScheduler
################################################################################



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
    s.conf = copy(conf(mc))
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
        accepted = global_update(update, mc, model, s.conf)
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
        return global_update(s.sequence[s.idx], mc, model, s.conf)
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



# Should this inherit from AbstractGlobalUpdate?
# This is required for AdaptiveScheduler
mutable struct AcceptanceStatistics{Update}
    accepted::Int
    total::Int
    update::Update
end
AcceptanceStatistics(update) = AcceptanceStatistics(0, 0, update)
AcceptanceStatistics(wrapped::AcceptanceStatistics) = wrapped
AcceptanceStatistics(proxy::Adaptive) = proxy
name(w::AcceptanceStatistics) = name(w.update)
function global_update(w::AcceptanceStatistics, mc, m, tc)
    accepted = global_update(w.update, mc, m, tc)
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