# For recording error information
#

# This type is for recording the minimum and maximum of some online sequence 
# alongside the average magnitude of the input. This is specifically for 
# recording propagation errors and negative or imagine probabilities. In that 
# case it's much more interesting to know the average magnitude instead of true
# average.
mutable struct MagnitudeStats
    max::Float64
    min::Float64
    sum::Float64
    count::Int64
end

MagnitudeStats() = MagnitudeStats(-Inf, +Inf, 0.0, 0)

function Base.push!(stat::MagnitudeStats, value)
    v = log10(abs(value))
    stat.max = max(stat.max, v)
    stat.min = min(stat.min, v)
    stat.sum += v
    stat.count += 1
end

Base.min(s::MagnitudeStats) = s.count > 0 ? 10.0^(s.min) : 0.0
Base.max(s::MagnitudeStats) = s.count > 0 ? 10.0^(s.max) : 0.0
Statistics.mean(s::MagnitudeStats) = s.count > 0 ? 10.0^(s.sum / s.count) : 0.0
Base.length(s::MagnitudeStats) = s.count

function Base.show(io::IO, s::MagnitudeStats)
    println(io, "MagnitudeStats: ($(s.count) Values)")
    println(io, "\tmin = $(min(s))")
    println(io, "\tmean = $(mean(s))")
    println(io, "\tmax = $(max(s))")
end



"""
    DQMCAnalysis

Analysis data of determinant quantum Monte Carlo (DQMC) simulations.
"""
@with_kw mutable struct DQMCAnalysis
    th_runtime::Float64 = 0.0
    me_runtime::Float64 = 0.0
    imaginary_probability::MagnitudeStats = MagnitudeStats()
    negative_probability::MagnitudeStats = MagnitudeStats()
    propagation_error::MagnitudeStats = MagnitudeStats()
end