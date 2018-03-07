module MonteCarlo
using MonteCarloObservable
using LightXML
using Parameters

include("abstract.jl")

include("flavors/MC/MC.jl")
include("flavors/DQMC/DQMC.jl")

include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/ALPS.jl")

include("models/Ising/IsingModel.jl")
include("models/Hubbard/HubbardModel.jl")

export reset!
export run!
export IsingModel
export HubbardModel
export MC
export DQMC
export observables

end # module
