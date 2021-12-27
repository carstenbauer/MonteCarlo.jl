module MonteCarlo

using LinearAlgebra: AbstractMatrix
using Reexport
# Loading the RNG will fail if Random is nto exported
@reexport using MonteCarloObservable, Random
import MonteCarloObservable.AbstractObservable
using Parameters, Requires
using TimerOutputs, StructArrays
using Printf, SparseArrays, LinearAlgebra, Dates, Statistics, Random, Distributed
import ProgressMeter

if get(ENV, "MONTECARLO_USE_LOOPVECTORIZATION", "true") == "true"
    import LoopVectorization
    using LoopVectorization: @turbo
else
    printstyled(
        "Using MonteCarlo.jl without LoopVectorization. This should only be done for tests.",
        color = :red
    )
    macro turbo(code)
        esc(quote @inbounds @fastmath $code end)
    end
end

import JLD, JLD2
using CodecZlib

# Because we fully load all data directly for JLD we lose access to the path
# This is supposed to keep track of path information
struct FileWrapper{T}
    file::T
    path::String

    FileWrapper(file::T, path) where T = new{T}(file, abspath(path))
end

function Base.getindex(fw::FileWrapper{T}, k) where T
    out = getindex(fw.file, k)
    if out isa Union{JLD2.JLDFile, JLD2.Group, Dict} # JLD generates a nested dict :(
        return FileWrapper(out, fw.path)
    else
        out
    end
end
Base.setindex!(fw::FileWrapper, k, v) = setindex!(fw.file, k, v)
Base.haskey(fw::FileWrapper, k) = haskey(fw.file, k)
Base.write(fw::FileWrapper, x) = write(fw.file, x)
Base.write(fw::FileWrapper, k, x) = write(fw.file, k, x)
Base.close(fw::FileWrapper) = close(fw.file)
Base.get(fw::FileWrapper, k, default) = haskey(fw, k) ? fw[k] : default
Base.keys(fw::FileWrapper) = keys(fw.file)

# To allow switching between JLD and JLD2:
const UnknownType = Union{JLD.UnsupportedType, JLD2.UnknownType}
const JLDFile = Union{FileWrapper{<: JLD.JldFile}, FileWrapper{<: JLD2.JLDFile}, JLD.JldFile, JLD2.JLDFile}



include("helpers.jl")
export enable_benchmarks, disable_benchmarks, print_timer, reset_timer!
include("linalg/general.jl")
include("linalg/UDT.jl")
include("linalg/complex.jl") # TODO
include("linalg/blockdiagonal.jl")
include("flavors/abstract.jl")
include("models/abstract.jl")
include("lattices/abstract.jl")

include("configurations.jl")
export Discarder, ConfigRecorder, BufferedConfigRecorder, RelativePath, AbsolutePath
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
export GreensMatrix, swapop
export boson_energy_measurement, greens_measurement, occupation, magnetization
export charge_density, charge_density_correlation, charge_density_susceptibility
export spin_density, spin_density_correlation, spin_density_susceptibility
export pairing, pairing_correlation, pairing_susceptibility
export current_current_susceptibility, superfluid_density
export noninteracting_energy, interacting_energy, total_energy

export EmptyScheduler, SimpleScheduler, AdaptiveScheduler
export Adaptive, NoUpdate, LocalSweep
export GlobalFlip, GlobalShuffle, SpatialShuffle, TemporalShuffle, 
        Denoise, DenoiseFlip, StaggeredDenoise
export ReplicaExchange, ReplicaPull, connect, disconnect
# export mask, uniform_fourier, structure_factor, SymmetryWrapped, swave, eswave

include("models/Ising/IsingModel.jl")
include("models/HubbardModel/HubbardModel.jl")
include("models/GaußHermite/GaußHermiteRepulsive.jl")
include("models/GaußHermite/GaußHermiteAttractive.jl")
export IsingEnergyMeasurement, IsingMagnetizationMeasurement

include("FileIO.jl")
export save, load, resume!


export reset!
export run!, resume!, replay!
export Model, IsingModel
export RepulsiveGHQHubbardModel, AttractiveGHQHubbardModel
export HubbardModel, HubbardModelAttractive, HubbardModelRepulsive
export MonteCarloFlavor, MC, DQMC
export greens, greens!, lattice, model, parameters

# For extending
export AbstractMeasurement, Model


function __init__()
    # @require LatPhysBase="eec5c15a-e8bd-11e8-0d23-6799ca40c963" include("lattices/LatPhys.jl")
    @require LatticePhysics = "53011200-ee7a-11e8-39f1-5f3e57afe4fd" include("lattices/LatPhys.jl")
    @require LightXML = "9c8b4983-aa76-5018-a973-4c85ecc9e179" include("lattices/ALPS.jl")
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        include("mpi.jl")
        export mpi_queue
        
        include("flavors/DQMC/updates/mpi_updates.jl")
        export MPIReplicaExchange, MPIReplicaPull
    end
    @require DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("DataFrames.jl")
end

end # module
