module MonteCarlo
using MonteCarloObservable

include("abstract.jl")

# include("flavors/MC/MC.jl")
include("flavors/Integrator/Integrator.jl")

include("lattices/square.jl")
include("lattices/ALPS.jl")

# include("models/Ising/IsingModel.jl")
include("models/GaussianFunction/GaussianFunction.jl")
# include("models/Hubbard/HubbardModel.jl")

export init!
export run!
# export IsingModel
# export MC
export observables

export Integrator
export GaussianFunction

end # module
