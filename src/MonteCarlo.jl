module MonteCarlo

using Reexport
# Loading the RNG will fail if Random is nto exported
@reexport using MonteCarloObservable, Random
import MonteCarloObservable.AbstractObservable
using Parameters, Requires
using TimerOutputs, LoopVectorization
using Printf, SparseArrays, LinearAlgebra, Dates, Statistics

import JLD, JLD2
# To allow switching between JLD and JLD2:
const UnknownType = Union{JLD.UnsupportedType, JLD2.UnknownType}
const JLDFile = Union{JLD.JldFile, JLD2.JLDFile}



include("helpers.jl")
include("inplace_udt.jl")
export enable_benchmarks, disable_benchmarks, print_timer, reset_timer!
include("flavors/abstract.jl")
include("models/abstract.jl")
include("lattices/abstract.jl")

include("configurations.jl")
export Discarder, ConfigRecorder
include("Measurements.jl")
export measurements, observables

include("lattices/masks.jl")
include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/honeycomb.jl")
include("lattices/triangular.jl")
include("lattices/ALPS.jl")
# export DistanceMask, RawMask # maybe?
export directions

include("flavors/MC/MC.jl")
include("flavors/DQMC/DQMC.jl")
export GreensMeasurement, BosonEnergyMeasurement, OccupationMeasurement,
        ChargeDensityCorrelationMeasurement, SpinDensityCorrelationMeasurement,
        MagnetizationMeasurement, PairingCorrelationMeasurement
export mask, uniform_fourier, structure_factor, SymmetryWrapped

include("models/Ising/IsingModel.jl")
include("models/HubbardAttractive/HubbardModelAttractive.jl")

include("FileIO.jl")
export save, load, resume!
# include("../test/testfunctions.jl")

export reset!
export run!, resume!, replay!
export IsingModel, HubbardModelAttractive
export MC, DQMC
export greens

# For extending
export AbstractMeasurement, Model


function __init__()
    @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
    @require LightXML = "9c8b4983-aa76-5018-a973-4c85ecc9e179" include("lattices/ALPS.jl")
end

end # module
