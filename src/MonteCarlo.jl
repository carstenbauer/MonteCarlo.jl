module MonteCarlo

using Reexport
@reexport using MonteCarloObservable
using StableDQMC, LightXML, Parameters, Requires
import LatPhysBase
import LatPhysBase.numSites

using Printf, SparseArrays, LinearAlgebra, Dates, Random

include("helpers.jl")
include("abstract.jl")

include("lattices/abstract.jl")
include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/ALPS.jl")
include("lattices/Honeycomb.jl")

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
export observables

function __init__()
    @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
end

end # module
