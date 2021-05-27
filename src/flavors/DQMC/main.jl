# Statistics for the DQMC simulation runtime. This has nothin to do with 
# measurements/observables.
include("statistics.jl")

# Some tapes, checkerboard stuff (which has been neglected a lot)
include("abstract.jl")

# contains `DQMCParameters` which holds parameters relevant to DQMC and the 
# general simulation. 
include("parameters.jl")


# There are a bunch of functions that require the DQMC type, but logically fit 
# with one of its constituents. For example `initialize_stack` requires 
# a ::DQMC to figure out how large a bunch of matrices need to be, but 
# logically fits in `stack.jl`.
mutable struct DQMC{
        M <: Model, CB <: Checkerboard, ConfType <: Any, RT <: AbstractRecorder, 
        Stack <: AbstractDQMCStack, UTStack <: AbstractDQMCStack,
        US <: AbstractUpdateScheduler
    } <: MonteCarloFlavor

    model::M
    conf::ConfType
    temp_conf::ConfType
    last_sweep::Int

    stack::Stack # s -> stack 
    ut_stack::UTStack
    scheduler::US
    parameters::DQMCParameters # p -> parameters
    analysis::DQMCAnalysis # a -> analysis

    recorder::RT # configs -> recorder
    thermalization_measurements::Dict{Symbol, AbstractMeasurement}
    measurements::Dict{Symbol, AbstractMeasurement}

    function DQMC{M, CB, ConfType, RT, Stack, UTStack, US}(args...) where {
            M <: Model, CB <: Checkerboard, ConfType <: Any, 
            RT <: AbstractRecorder, 
            Stack <: AbstractDQMCStack, UTStack <: AbstractDQMCStack,
            US <: AbstractUpdateScheduler
        }
        
        @assert isconcretetype(M)
        @assert isconcretetype(ConfType)
        @assert isconcretetype(Stack)
        @assert isconcretetype(UTStack)
        @assert isconcretetype(RT)
        @assert isconcretetype(US)
        
        new{M, CB, ConfType, RT, Stack, UTStack, US}(args...)
    end
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
# include("global.jl")
include("updates/scheduler.jl")
include("updates/local_updates.jl")
include("updates/global_updates.jl")
include("updates/parallel_updates.jl")

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