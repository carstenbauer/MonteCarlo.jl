module MonteCarlo

using Reexport
@reexport using MonteCarloObservable
import MonteCarloObservable.AbstractObservable
using StableDQMC, LightXML, Parameters, Requires

using Printf, SparseArrays, LinearAlgebra, Dates, Random


include("flavors/abstract.jl")
include("models/abstract.jl")
include("lattices/abstract.jl")

include("helpers.jl")
include("Measurements.jl")
export measurements, observables, save_measurements!

include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/honeycomb.jl")
include("lattices/ALPS.jl")

include("flavors/MC/MC.jl")
include("flavors/DQMC/DQMC.jl")

include("models/Ising/IsingModel.jl")
include("models/HubbardAttractive/HubbardModelAttractive.jl")

include("../test/testfunctions.jl")

export reset!
export run!
export IsingModel
export HubbardModelAttractive
export MC
export DQMC
export greens

function __init__()
    @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
end

end # module
