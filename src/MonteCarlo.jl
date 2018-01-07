module MonteCarlo
using MonteCarloObservable

include("abstract.jl")

include("flavors/MC/MC.jl")

include("lattices/square.jl")
include("lattices/ALPS.jl")

include("models/Ising/IsingModel.jl")
# include("models/Hubbard/HubbardModel.jl")

export init!
export run!
export IsingModel
export MC
export observables

end # module
