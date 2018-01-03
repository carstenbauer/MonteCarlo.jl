module MonteCarlo
using MonteCarloObservable

include("abstract/abstract_model.jl")
include("abstract/abstract_flavor.jl")
include("abstract/abstract_lattice.jl")
include("abstract/abstract_functions.jl")

# # include all lattices
# for l in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"lattices"))))
#     include(joinpath(@__DIR__,"lattices",l))
# end

# # include all models
# for m in filter(x->isdir(joinpath(@__DIR__,"models",x)),
#                 sort!(readdir(joinpath(@__DIR__,"models"))))
#     for jlf in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"models",m))))
#         include(joinpath(@__DIR__,"models",m,jlf))
#     end
# end

# # include all mc flavors
# for flv in filter(x->isdir(joinpath(@__DIR__,"flavors",x)),
#                 sort!(readdir(joinpath(@__DIR__,"flavors"))))
#     for jlf in filter(x->endswith(x,".jl"), sort!(readdir(joinpath(@__DIR__,"flavors",flv))))
#         include(joinpath(@__DIR__,"flavors",flv,jlf))
#     end
# end

include("lattices/square.jl")
include("lattices/ALPS.jl")

include("models/Ising/IsingModel.jl")

include("flavors/MC/mc.jl")

export init!
export run!
export IsingModel
export MC
export observables

end # module
