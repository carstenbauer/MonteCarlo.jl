module MonteCarlo

using Reexport
@reexport using MonteCarloObservable
import MonteCarloObservable.AbstractObservable
using Parameters, Requires
using JLD, TimerOutputs
using LoopVectorization, RecursiveFactorization

using Printf, SparseArrays, LinearAlgebra, Dates, Random, Statistics

include("helpers.jl")
include("inplace_udt.jl")
export enable_benchmarks, disable_benchmarks, print_timer, reset_timer!
include("flavors/abstract.jl")
include("models/abstract.jl")
include("lattices/abstract.jl")

include("Measurements.jl")
export measurements, observables

include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/honeycomb.jl")
include("lattices/ALPS.jl")

include("flavors/MC/MC.jl")
include("flavors/DQMC/DQMC.jl")

include("models/Ising/IsingModel.jl")
include("models/HubbardAttractive/HubbardModelAttractive.jl")

include("configurations.jl")
include("FileIO.jl")
export save, load, resume!
# include("../test/testfunctions.jl")

export reset!
export run!, resume!, replay!
export IsingModel
export HubbardModelAttractive
export MC
export DQMC
export greens

function __init__()
    @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
    @require LightXML = "9c8b4983-aa76-5018-a973-4c85ecc9e179" include("lattices/ALPS.jl")
end

end # module
