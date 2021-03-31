# Statistics for the DQMC simulation runtime. This has nothin to do with 
# measurements/observables.
include("statistics.jl")

# Some tapes, checkerboard stuff (which has been neglected a lot)
include("abstract.jl")

# contains `DQMCParameters` which holds parameters relevant to DQMC and the 
# general simulation. 
include("parameters.jl")


# The DQMC implementation requires a bit of splitting. Many methods and structs
# use the DQMC type in some capacity, so we need to define it first. However 
# methods connected closely to DQMC need those structs and methods as well.
# Therefore we'll put the type definition here for now.

# TODO
# - Is there a better way to make this concretely typed?


mutable struct DQMC{
        M <: Model, CB <: Checkerboard, ConfType <: Any, RT <: AbstractRecorder, 
        Stack <: AbstractDQMCStack, UTStack <: AbstractDQMCStack
    } <: MonteCarloFlavor

    model::M
    conf::ConfType
    last_sweep::Int

    stack::Stack # s -> stack 
    ut_stack::UTStack
    # scheduler::UST
    parameters::DQMCParameters # p -> parameters
    analysis::DQMCAnalysis # a -> analysis

    recorder::RT # configs -> recorder
    thermalization_measurements::Dict{Symbol, AbstractMeasurement}
    measurements::Dict{Symbol, AbstractMeasurement}


    function DQMC{M, CB, ConfType, RT, Stack, UTStack}(; kwargs...) where {
            M <: Model, CB <: Checkerboard, ConfType <: Any, 
            RT <: AbstractRecorder, 
            Stack <: AbstractDQMCStack, UTStack <: AbstractDQMCStack
        }
        complete!(new{M, CB, ConfType, RT, Stack, UTStack}(), kwargs)
    end
    function DQMC(CB::Type{<:Checkerboard}; kwargs...)
        DQMC{
            Model, CB, Any, AbstractRecorder, 
            AbstractDQMCStack, AbstractDQMCStack
        }(; kwargs...)
    end
    function DQMC{CB}(
            model::M,
            conf::ConfType,
            last_sweep::Int,
            s::Stack,
            ut_stack::UTStack,
            p::DQMCParameters,
            a::DQMCAnalysis,
            configs::RT,
            thermalization_measurements::Dict{Symbol, AbstractMeasurement},
            measurements::Dict{Symbol, AbstractMeasurement}
        ) where {
            M <: Model, CB <: Checkerboard, ConfType <: Any, 
            RT <: AbstractRecorder, 
            Stack <: AbstractDQMCStack, UTStack <: AbstractDQMCStack
        }
        new{M, CB, ConfType, RT, Stack, UTStack}(
            model, conf, last_sweep, s, ut_stack, p, a, configs, 
            thermalization_measurements, measurements
        )
    end
    function DQMC{M, CB, C, RT, S, UTS}(args...) where {M, CB, C, RT, S, UTS}
        DQMC{CB}(args...)
    end
end


function complete!(a::DQMC, kwargs)
    for (field, val) in kwargs
        setfield!(a, field, val)
    end
    make_concrete!(a)
end

function make_concrete!(a::DQMC{M, CB, C, RT, S, UTS}) where {M, CB, C, RT, S, UTS}
    Ts = (
        isdefined(a, :model) ? typeof(a.model) : M, 
        CB, 
        isdefined(a, :conf) ? typeof(a.conf) : C, 
        isdefined(a, :configs) ? typeof(a.configs) : RT, 
        isdefined(a, :s) ? typeof(a.s) : S,
        isdefined(a, :ut_stack) ? typeof(a.ut_stack) : UTS
    )
    all(Ts .== (M, CB, C, RT, S, UTS)) && return a
    data = [(f, getfield(a, f)) for f in fieldnames(DQMC) if isdefined(a, f)]
    DQMC{Ts...}(; data...)
end


# Contains `DQMCStack`, `propagate` and related functions. The stack keeps track
# of all the matrices that go into calculating the path integral. `propagate` 
# moves imaginary time along, between `l = 1` and `l = slices`.
# This also contains some functions for calculating greens functions.
include("stack.jl")

# Contains the `UnequalTimeStack`, related greens calculation methods and 
# iterators (i.e. CombinedGreens). This is used for unequal time measurements.
include("unequal_time_stack.jl")

# Contains functions for computations using the matrices representing a time 
# slice. Also contains special methods for Checkerboard decomposed matrices.
include("slice_matrices.jl")

# Contains global update schedulers, the updates themselves as well as some 
# functions necessary to calculate and perform those updates.
include("global.jl")

# Contains the "main/top-level" simulation code. Specifically construction and
# initialization of the DQMC type, convenience functions, `run!` and `replay!`, 
# and local updates.
include("DQMC.jl")

# Contains mandatory and optional method definitions expected to exist for a 
# given model.
include("DQMC_mandatory.jl")
include("DQMC_optional.jl")

# saving and loading of DQMC
include("FileIO.jl")

# Contains some greens methods
# some others are in stack.jl and unequal_time_stack.jl
include("greens.jl")

# Contains code related to make measurements. Specifically:
# The overall structure
include("measurements/generic.jl")
# Quick constructers and measurement kernels (i.e. applied Wicks theorem)
include("measurements/measurements.jl")
# Contains some post processing tools
include("measurements/extensions.jl")
# structs and conversions from the old system
include("measurements/deprecated.jl")