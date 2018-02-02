module MonteCarlo
using MonteCarloObservable

include("abstract.jl")

include("flavors/MC/MC.jl")
include("flavors/Integrator/Integrator.jl")

include("lattices/square.jl")
include("lattices/ALPS.jl")

include("models/Ising/IsingModel.jl")
include("models/GaussianFunction/GaussianFunction.jl")

# general methods
export init!
export run!
export observables

# models
export GaussianFunction
export IsingModel

# flavors
export MC
export Integrator

end # module
