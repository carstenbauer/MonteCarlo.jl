module MonteCarlo

using LinearAlgebra: AbstractMatrix
using Reexport
# Loading the RNG will fail if Random is nto exported
@reexport using MonteCarloObservable, Random
import MonteCarloObservable.AbstractObservable
using Parameters, Requires
using TimerOutputs, LoopVectorization, StructArrays
using Printf, SparseArrays, LinearAlgebra, Dates, Statistics, Random, Distributed

import JLD, JLD2
# To allow switching between JLD and JLD2:
const UnknownType = Union{JLD.UnsupportedType, JLD2.UnknownType}
const JLDFile = Union{JLD.JldFile, JLD2.JLDFile}



include("helpers.jl")
export enable_benchmarks, disable_benchmarks, print_timer, reset_timer!
include("linalg/general.jl")
include("linalg/UDT.jl")
# include("linalg/complex.jl") # TODO
include("linalg/blockdiagonal.jl")
include("flavors/abstract.jl")
include("models/abstract.jl")
include("lattices/abstract.jl")

include("configurations.jl")
export Discarder, ConfigRecorder
include("Measurements.jl")
export measurements, observables

include("lattices/lattice_iterators.jl")
include("lattices/square.jl")
include("lattices/chain.jl")
include("lattices/cubic.jl")
include("lattices/honeycomb.jl")
include("lattices/triangular.jl")
include("lattices/ALPS.jl")
include("lattices/deprecated.jl")
# export directions, RawMask, DistanceMask # maybe getorder?
export AbstractLattice, Chain, SquareLattice, CubicLattice, TriangularLattice, ALPSLattice
export EachSite, EachSiteAndFlavor, OnSite, EachSitePair, EachSitePairByDistance, 
        EachLocalQuadByDistance, EachLocalQuadBySyncedDistance, 
        Sum, ApplySymmetries
export neighbors, directions

include("flavors/MC/MC.jl")
include("flavors/DQMC/main.jl")
export Greens, GreensAt, CombinedGreensIterator
export GreensMatrix, dagger
export boson_energy_measurement, greens_measurement, occupation, magnetization
export charge_density, charge_density_correlation, charge_density_susceptibility
export spin_density, spin_density_correlation, spin_density_susceptibility
export pairing, pairing_correlation, pairing_susceptibility
export current_current_susceptibility, superfluid_density
export noninteracting_energy, interacting_energy, total_energy

export EmptyScheduler, SimpleScheduler, AdaptiveScheduler
export Adaptive, NoUpdate, LocalSweep, GlobalFlip, GlobalShuffle
export ReplicaExchange, ReplicaPull, connect, disconnect
# export mask, uniform_fourier, structure_factor, SymmetryWrapped, swave, eswave

include("models/Ising/IsingModel.jl")
include("models/HubbardModel/HubbardModel.jl")
export IsingEnergyMeasurement, IsingMagnetizationMeasurement

include("FileIO.jl")
export save, load, resume!


export reset!
export run!, resume!, replay!
export Model, IsingModel, HubbardModel, HubbardModelAttractive, HubbardModelRepulsive
export MonteCarloFlavor, MC, DQMC
export greens, lattice, model, parameters

# For extending
export AbstractMeasurement, Model


function __init__()
    # @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
    @require LatticePhysics = "53011200-ee7a-11e8-39f1-5f3e57afe4fd" include("lattices/LatPhys.jl")
    @require LightXML = "9c8b4983-aa76-5018-a973-4c85ecc9e179" include("lattices/ALPS.jl")
end

end # module
