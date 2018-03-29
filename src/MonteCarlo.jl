module MonteCarlo
using MonteCarloObservable
using Parameters

include("helpers.jl")
include("abstract.jl")

module Lattices
	using LightXML
	using Parameters
	include("lattices/abstract.jl")
	export Lattice
	export AbstractCubicLattice
	include("lattices/square.jl")
	export SquareLattice
	include("lattices/chain.jl")
	export Chain
	include("lattices/cubic.jl")
	export CubicLattice
	include("lattices/ALPS.jl")
	export ALPSLattice
end
using .Lattices

module MCm
	using MonteCarloObservable
	using Parameters
	using ..Lattices
	include("helpers.jl")
	include("abstract.jl")

	include("flavors/MC/MC.jl")

	export reset!
	export run!
	export MC
end
using .MCm

module DQMCm
	using MonteCarloObservable
	using Parameters
	using ..Lattices
	using ..MonteCarlo: Model, MonteCarloFlavor
	include("helpers.jl")

	include("flavors/DQMC/DQMC.jl")

	export reset!
	export run!
	export DQMC

	export conftype
end
using .DQMCm

include("models/Ising/IsingModel.jl")
include("models/HubbardAttractive/HubbardModelAttractive.jl")

include("../test/testfunctions.jl")

export reset!
export run!
export IsingModel
export HubbardModelAttractive
export MC
export DQMC
export observables

end # module
