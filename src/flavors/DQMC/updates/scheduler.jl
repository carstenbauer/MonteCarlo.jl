# This file should probably be moved to the general directory and used by 
# classical MC as well


abstract type AbstractUpdate end
abstract type AbstractLocalUpdate <: AbstractUpdate end



"""
    Adaptive()

A placeholder for adaptive updates in the `AdaptiveScheduler`.
"""
struct Adaptive <: AbstractUpdate end
name(::Adaptive) = "Adaptive"



################################################################################
### SimpleScheduler
################################################################################



"""
    SimpleScheduler(updates...)

Schedules global updates in the order specified through `updates...`. For example
`SimpleScheduler(mc, GlobalFlip(), GlobalFlip(), GlobalShuffle())` would generate
a sequence `GlobalFlip() > GlobalFlip() > GlobalShuffle() > repeat`.
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
        updates = map(AcceptanceStatistics, Tuple(vcat(updates...)))
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

function update(s::SimpleScheduler, mc::DQMC, model)
    s.idx = mod1(s.idx + 1, length(s.sequence))
    update(s.sequence[s.idx], mc, model)
end

function show_statistics(s::SimpleScheduler, prefix="")
    println(prefix, "Global update statistics (since start):")

    cum_accepted  = mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.sequence)
    cum_total  = mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.sequence)

    accumulated = flatten_sequence_statistics(s.sequence)
    show_accumulated_sequence(accumulated, prefix * "\t", cum_total)
        
    p = @sprintf("%2.1f", 100cum_accepted/max(1, cum_total))
    s = rpad("Total", 20) * " " * lpad(p, 5) * "% accepted" *
        "   (" * string(round(Int, cum_accepted)) * " / $cum_total)"

    println(prefix, '\t', '-' ^ length(s))
    println("$prefix\t", s)

    nothing
end

function Base.show(io::IO, s::SimpleScheduler)
    sequence = combine_sequence_to_string(s.sequence)
    print(io, "SimpleScheduler(): $sequence -> (repeat)")
end

function save_scheduler(file::JLDFile, s::SimpleScheduler, entryname::String="/Scheduler")
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(s))
    write(file, entryname * "/sequence", s.sequence)
    write(file, entryname * "/idx", s.idx)
    nothing
end
function _load(data, ::Type{<: SimpleScheduler})
    SimpleScheduler(data["sequence"], data["idx"])
end



################################################################################
### AdaptiveScheduler
################################################################################



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
mutable struct AdaptiveScheduler{PT, ST} <: AbstractUpdateScheduler
    # below this, the sampling rate is set to 0. Basing this on sampling rate
    # should allow us to make things a bit smoother and avoid bad luck.
    minimum_sampling_rate::Float64 # 0.01?
    # some number of sweeps where we just collect statistics
    grace_period::Int # 100?
    # Weight for how slowly sampling_rate changes.
    # sampling_rate = (adaptive_rate * sampling_rate + accepted/total) / (adaptive_rate+1)
    # 0 = use current probability as sampling rate
    adaptive_rate::Float64 # 9.0?

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
        sequence = map(AcceptanceStatistics, vcat(sequence...))
        if !any(u -> u isa AcceptanceStatistics{<: AbstractLocalUpdate}, sequence)
            error("The scheduler requires local updates, but none were passed.")
        end

        # sampling rate x after i steps with constant acceptance rate p:
        # x_i = (c/c+1)^i x_0 + a * \sum_{i} c^{i-1} / (c+1)^i
        # where c is the adaptive rate and x_0 is the initial samplign rate.
        # Minimum number of samples to discard:
        i_min = ceil(Int, log(
            adaptive_rate / (adaptive_rate+1), 
            minimum_sampling_rate / 0.5
        ))
        @info("Minimum number of samples for discard: $grace_period + $i_min")

        if !any(x -> x isa NoUpdate, pool)
            pool = tuple(vcat(pool...)..., NoUpdate())
        end

        # Wrap in AcceptanceStatistics
        adaptive_pool = map(AcceptanceStatistics, pool)

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

function update(s::AdaptiveScheduler, mc::DQMC, model)
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
        accepted = update(updater, mc, model)
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

        return accepted
    else
        # Some sort of non-adaptive update, just perform it.
        return update(s.sequence[s.idx], mc, model)
    end
end

function show_statistics(s::AdaptiveScheduler, prefix="")
    println(prefix, "Update statistics (since start):")

    cum_accepted  = mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.sequence)
    cum_accepted += mapreduce(u -> u isa AcceptanceStatistics ? u.accepted : 0, +, s.adaptive_pool)
    cum_total  = mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.sequence)
    cum_total += mapreduce(u -> u isa AcceptanceStatistics ? u.total : 0, +, s.adaptive_pool)

    accumulated = flatten_sequence_statistics(s.adaptive_pool)
    flatten_sequence_statistics(s.sequence, accumulated)
    show_accumulated_sequence(accumulated, prefix * "\t", cum_total)

    p = @sprintf("%2.1f", 100cum_accepted/max(1, cum_total))
    s = rpad("Total", 20) * " " * lpad(p, 5) * "% accepted" *
        "   (" * string(round(Int, cum_accepted)) * " / $cum_total)"

    println(prefix, '\t', '-' ^ length(s))
    println("$prefix\t", s)

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
function _load(data, ::Type{<: AdaptiveScheduler})
    s = AdaptiveScheduler(data["sequence"], data["pool"])
    s.sampling_rates = data["sampling_rates"]
    s.minimum_sampling_rate = data["minimum_sampling_rate"]
    s.grace_period = data["grace_period"]
    s.adaptive_rate = data["adaptive_rate"]
    s.idx = data["idx"]
    s
end



# Should this inherit from AbstractGlobalUpdate?
# This is required for AdaptiveScheduler
mutable struct AcceptanceStatistics{Update <: AbstractUpdate}
    accepted::Float64
    total::Int
    update::Update
end
AcceptanceStatistics(update) = AcceptanceStatistics(0.0, 0, update)
AcceptanceStatistics(wrapped::AcceptanceStatistics) = wrapped
AcceptanceStatistics(proxy::Adaptive) = proxy
name(w::AcceptanceStatistics) = name(w.update)
function update(w::AcceptanceStatistics, mc, m)
    accepted = update(w.update, mc, m)
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



################################################################################
### Utillities
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

function show_accumulated_sequence(accumulated, prefix = "", max_total=0)
    N = length(string(max_total))
    sorted = sort!(collect(accumulated), by = x -> x[2][1] / max(1, x[2][2]))
    for (key, (acc, total)) in sorted
        total == 0 && continue
        p = @sprintf("%2.1f", 100acc / max(1, total))
        println(
            prefix, rpad(key, 20), " ", lpad(p, 5), "% accepted",  "   (", 
            lpad(string(round(Int, acc)), N), " / ", lpad(string(total), N), ")"
        )
    end

    nothing
end

function show_sequence(sequence, prefix = "", max_total=0)
    N = length(string(max_total))
    for update in sequence
        if update isa AcceptanceStatistics{NoUpdate} ||
            update isa Adaptive || !(update isa AcceptanceStatistics)
            continue
        end

        p = @sprintf("%2.1f", 100update.accepted/max(1, update.total))
        println(
            prefix, rpad(name(update), 20), " ", lpad(p, 5), "% accepted", "(", 
            lpad(string(round(Int, update.accepted)), N), " / ", 
            lpad(string(update.total), N), ")"
        )
    end

    nothing
end

function combine_sequence_to_string(sequence)
    _name = name(first(sequence))
    counter = 1
    output = ""
    for update in sequence[2:end]
        if name(update) == _name
            counter += 1
        else
            if counter == 1
                output *= " -> $_name"
            else
                output *= " -> $(_name)($counter)"
            end
            counter = 1
            _name = name(update)
        end
    end
    
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